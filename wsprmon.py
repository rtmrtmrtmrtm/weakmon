#!/usr/local/bin/python

#
# WSPR receiver.
#
# switches among bands if weakcat.py understands the radio.
# reports to wsprnet if mycall/mygrid defined in weak.ini.
#
# Robert Morris, AB1HL
#

import wspr
import sys
import os
import time
import weakaudio
import numpy
import threading
import re
import random
import copy
import weakcat
from six.moves import urllib
#import urllib.request, urllib.parse, urllib.error
import weakutil
import weakargs

# look only at these bands.
plausible = [ "80", "40", "30", "20", "17" ]

b2f = { "80" : 3.568600, "40" : 7.038600, "30" : 10.138700, "20" : 14.095600,
        "17" : 18.104600, "15" : 21.094600, "12" : 24.924600,
        "10" : 28.124600, "6" : 50.293000, "2" : 144.489 }

def load_prefixes():
    d = { }
    f = open("jt65prefixes.dat")
    for ln in f:
        ln = re.sub(r'\t', ' ', ln)
        ln = re.sub(r'  *', ' ', ln)
        ln.strip()
        ln = re.sub(r' *\(.*\) *', '', ln)
        ln.strip()
        m = re.search(r'^([A-Z0-9]+) +(.*)', ln)
        if m != None:
            d[m.group(1)] = m.group(2)
    f.close()
    return d

def look_prefix(call, d):
    if len(call) == 5 and call[0:3] == "KG4":
        # KG4xx is Guantanamo, KG4x and KG4xxx are not.
        return "Guantanamo Bay"

    while len(call) > 0:
        if call in d:
            return d[call]
        call = call[0:-1]
    return None

# weighted choice (to pick bands).
# a[i] = [ value, weight ]
def wchoice(a, n):
    total = 0.0
    for e in a:
        total += e[1]

    ret = [ ]
    while len(ret) < n:
        x = random.random() * total
        for ai in range(0, len(a)):
            e = a[ai]
            if x <= e[1]:
                ret.append(e[0])
                total -= e[1]
                a = a[0:ai] + a[ai+1:]
                break
            x -= e[1]

    return ret

def wchoice_test():
    a = [ [ "a", .1 ], [ "b", .1 ], [ "c", .4 ], [ "d", .3 ], [ "e", .1 ] ]
    counts = { }
    for iter in range(0, 500):
        x = wchoice(a, 2)
        for e in x:
            counts[e] = counts.get(e, 0) + 1
    print(counts)

class WSPRMon:
    def __init__(self, incard, cat, oneband):
        self.mycall = weakutil.cfg("wsprmon", "mycall")
        self.mygrid = weakutil.cfg("wsprmon", "mygrid")

        self.running = True
        self.rate = 12000
        self.logname = "wspr-log.txt"
        self.bandname = "wspr-band.txt"
        self.jtname = "wspr"
        self.verbose = False

        self.incard = incard
        self.oneband = oneband

        if cat != None:
            self.cat = weakcat.open(cat)
            self.cat.sync()
            self.cat.set_usb_data()
        else:
            self.cat = None

        # for each band, count of received signals last time we
        # looked at it, to guess most profitable band.
        self.bandinfo = { }

        # for each two-minute interval, the band we were listening on.
        self.minband = { }

        # has readall() processed each interval?
        self.mindone = { }

        self.prefixes = load_prefixes()

    def start(self):
        self.r = wspr.WSPR()
        self.r.cardrate = self.rate
        self.r.opencard(self.incard)
        self.rth = threading.Thread(target=lambda : self.r.gocard())
        self.rth.daemon = True
        self.rth.start()

        if self.mycall == None or self.mygrid == None:
            print("not reporting to wsprnet because no mycall/mygrid in weak.cfg")
        elif True:
            self.nth = threading.Thread(target=lambda : self.gonet())
            self.nth.daemon = True
            self.nth.start()
            print("reporting to wsprnet as %s at %s." % (self.mycall, self.mygrid))
        else:
            print("not reporting to wsprnet.")

    def close(self):
        self.running = False
        self.r.close()
        self.rth.join()
        self.nth.join()
        self.pya.terminate()

    # thread to send to wsprnet.
    # hints from wsjtx wsprnet.cpp and
    # http://blog.marxy.org/2015/12/wsprnet-down-up-down.html
    def gonet(self):
        mi = 0
        while self.running:
            time.sleep(30)
            msgs = self.r.get_msgs()
            while mi < len(msgs):
                msg = msgs[mi]
                mi += 1
                # msg is a wspr.Decode.
                if not (msg.minute in self.minband):
                    continue
                band = self.minband[msg.minute]
                pp = self.parse(msg.msg)
                if pp == None:
                    continue
                [ call, grid, dbm ] = pp

                when = self.r.start_time + 60*msg.minute
                gm = time.gmtime(when)

                url = "http://wsprnet.org/post?"
                url += "function=wspr&"
                url += "rcall=%s&" % (self.mycall)
                url += "rgrid=%s&" % (self.mygrid)
                url += "rqrg=%.6f&" % (b2f[band]) # my frequency, mHz
                url += "date=%02d%02d%02d&" % (gm.tm_year-2000, gm.tm_mon, gm.tm_mday)
                url += "time=%02d%02d&" % (gm.tm_hour, gm.tm_min)
                url += "sig=%.0f&" % (msg.snr)
                url += "dt=%.1f&" % (msg.dt)
                url += "drift=%.1f&" % (msg.drift)
                url += "tqrg=%.6f&" % (b2f[band] + msg.hz()/1000000.0)
                url += "tcall=%s&" % (call)
                url += "tgrid=%s&" % (grid)
                url += "dbm=%s&" % (dbm)
                url += "version=weakmon-0.3&"
                url += "mode=2"

                try:
                    req = urllib.request.urlopen(url)
                    for junk in req:
                        pass
                    req.close()
                except:
                    print("wsprnet GET failed for %s" % (msg.msg))
                    pass

    # process messages from one cycle ago, i.e. the latest
    # cycle for which both reception and 
    # decoding have completed.
    def readall(self):
        now = time.time()
        nowmin = self.r.minute(now)
        for min in range(max(0, nowmin-6), nowmin, 2):
            if min in self.mindone:
                continue
            self.mindone[min] = True

            if not (min in self.minband):
                continue
            band = self.minband[min]

            bandcount = 0
            msgs = self.r.get_msgs()
            # each msg is a wspr.Decode.
            for m in msgs[len(msgs)-50:]:
                if m.minute == min:
                    bandcount += 1
                    self.log(self.r.start_time + 60*min, band, m.hz(), m.msg, m.snr, m.dt, m.drift)
            x = self.bandinfo.get(band, 0)
            self.bandinfo[band] = 0.5 * x + 0.5 * bandcount

    # turn "WB4HIR EM95 33" into ["WB4HIR", "EM95", "33"], or None.
    def parse(self, msg):
        msg = msg.strip()
        msg = re.sub(r'  *', ' ', msg)
        m = re.search(r'^([A-Z0-9\/]+) ([A-Z0-9]+) ([0-9]+)', msg)
        if m == None:
            print("wsprmon log could not parse %s" % (msg))
            return None

        call = m.group(1)
        grid = m.group(2)
        dbm = m.group(3)

        return [ call, grid, dbm ]

    def log(self, when, band, hz, msg, snr, dt, drift):
        pp = self.parse(msg)
        if pp == None:
            return
        [ call, grid, dbm ] = pp

        entity = look_prefix(call, self.prefixes)

        # b2f is mHz
        freq = b2f[band] + hz / 1000000.0

        ts = self.r.ts(when)
        ts = re.sub(r':[0-9][0-9]$', '', ts) # delete seconds

        info = "%s %9.6f %s %s %s %.0f %.1f %.1f %s" % (ts,
                                                        freq,
                                                        call,
                                                        grid,
                                                        dbm,
                                                        snr,
                                                        dt,
                                                        drift,
                                                        entity)
        print("%s" % (info))

        f = open(self.logname, "a")
        f.write("%s\n" % (info))
        f.close()

    # return a good band on which to listen.
    def rankbands(self):
        global plausible

        # are we missing bandinfo for any bands?
        missing = [ ]
        for b in plausible:
            if self.bandinfo.get(b) == None:
                missing.append(b)

        # always explore missing bands first.
        if len(missing) > 0:
            band = missing[0]
            # so we no longer count it as "missing".
            self.bandinfo[band] = 0
            return band

        # most profitable bands, best first.
        best = sorted(plausible, key = lambda b : -self.bandinfo.get(b, -1))

        if random.random() < 0.3 or self.bandinfo[best[0]] <= 0.1:
            band = random.choice(plausible)
        else:
            wa = [ [ b, self.bandinfo[b] ] for b in best ]
            band = wchoice(wa, 1)[0]

        return band

    def go(self):
        while self.running:
            # wait until we'are at the start of a two-minute interval.
            # that is, don't tell the radio to change bands in the
            # middle of an interval.
            while True:
                if self.running == False:
                    return
                second = self.r.second(time.time())
                if second >= 119 or second < 1:
                    break
                time.sleep(0.2)

            # choose a band.
            if self.oneband != None:
                band = self.oneband
            else:
                band = self.rankbands()
            if self.cat != None:
                self.cat.setf(0, int(b2f[band] * 1000000.0))

            now = time.time()
            if self.r.second(now) < 5:
                min = self.r.minute(now)
            else:
                min = self.r.minute(now + 5)

            # remember the band for this minute, for readall().
            self.minband[min] = band

            if self.verbose:
                sys.stdout.write("band %s ; " % (band))
                for b in self.bandinfo:
                    sys.stdout.write("%s %.1f, " % (b, self.bandinfo[b]))
                sys.stdout.write("\n")
                sys.stdout.flush()

            # make sure we get into the next minute
            time.sleep(5)

            # collect incoming message reports.
            while self.running:
                now = time.time()
                second = self.r.second(now)
                if second >= 118:
                    break
                self.readall()
                time.sleep(1)
                
def oldmain():
    incard = None
    cattype = None
    catdev = None
    oneband = None
    levels = False
    vflag = False
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-in":
            incard = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-cat":
            cattype = sys.argv[i+1]
            catdev = sys.argv[i+2]
            i += 3
        elif sys.argv[i] == "-band":
            oneband = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-levels":
            levels = True
            i += 1
        elif sys.argv[i] == "-v":
            vflag = True
            i += 1
        else:
            usage()

    if levels:
        # print sound card avg/peak once per second, to
        # adjust level.
        if incard == None:
            usage()
        c = weakaudio.new(incard, 12000)
        c.levels()
        sys.exit(0)

    if catdev == None and oneband == None:
        sys.stderr.write("wsprmon needs either -cat or -band\n")
        usage()
    
    if incard != None:
        w = WSPRMon(incard, cattype, catdev, oneband)
        w.verbose = vflag
        w.start()
        w.go()
        w.close()
    else:
        usage()

def main():
    parser = weakargs.stdparse('Decode WSPR.')
    parser.add_argument("-band")
    args = weakargs.parse_args(parser)
        
    if args.card == None:
        parser.error("wsprmon requires -card")
      
    if args.cat == None and args.band == None:
        parser.error("wsprmon needs either -cat or -band")

    w = WSPRMon(args.card, args.cat, args.band)
    w.verbose = args.v
    w.start()
    w.go()
    w.close()

    sys.exit(0)

main()
