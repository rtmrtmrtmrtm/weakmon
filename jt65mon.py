#!/usr/local/bin/python

#
# receive JT65.
#
# switches among bands if weakcat.py understands the radio.
# reports to pskreporter.info if mycall/mygrid defined in weak.ini.
#
# Robert Morris, AB1HL
#

import jt65
import sys
import os
import time
import numpy
import threading
import re
import random
import copy
import weakcat
import weakaudio
import weakutil
import weakargs
import pskreport

# look only at these bands.
plausible = [ "40", "30", "20", "17", "15" ]

b2f = { "80" : 3.576, "40" : 7.076, "30" : 10.138, "20" : 14.076,
        "17" : 18.102, "15" : 21.076, "12" : 24.917,
        "10" : 28.076, "6" : 50.276 }

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
    print counts

# listen for CQ, answer.
class JT65Mon:
    def __init__(self, desc1, desc2, cat, oneband):
        self.mycall = weakutil.cfg("jt65mon", "mycall")
        self.mygrid = weakutil.cfg("jt65mon", "mygrid")

        self.oneband = oneband
        self.verbose = False
        self.rate = 11025
        self.allname = "jt65-all.txt"
        self.bandname = "jt65-band.txt"
        self.jtname = "jt65"

        self.incards = [ ]
        self.incards.append(desc1)
        if desc2 != None:
            self.incards.append(desc2)

        if cat != None:
            self.cat = weakcat.open(cat)
            self.cat.sync()
            self.cat.set_usb_data()
        else:
            self.cat = None

        # for each band, count of received signals last time we
        # looked at it, to guess most profitable band.
        self.bandinfo = { }

        self.prefixes = load_prefixes()

        if self.mycall != None and self.mygrid != None:
            # talk to pskreporter.
            print "reporting to pskreporter as %s at %s" % (self.mycall, self.mygrid)
            self.pskr = pskreport.T(self.mycall, self.mygrid, "weakmon 0.2", False)
        else:
            print "not reporting to pskreporter since call/grid not in weak.cfg"
            self.pskr = None

    # read latest msgs from all cards.
    # each msg is [ minute, hz, msg, decode_time, nerrs, snr, card# ]
    def readall(self, now, bands):
        minute = self.r[0].minute(now)

        # for each card, new msgs
        all = [ ] # all messages, with duplicates
        d = { } # for each text, [ card_index, msg ]
        for ci in range(0, len(self.r)):
            bandcount = 0  # all msgs
            bandcount1 = 0 # msgs with lowist reed-solomon error counts
            msgs = self.r[ci].get_msgs()
            # each msg is [ minute, hz, msg, decode_time, nerrs, snr ]
            for m in msgs[self.msgs_index[ci]:]:
                m = m + [ ci ]
                if m[0] == minute:
                    bandcount += 1
                    if m[4] < 25:
                        bandcount1 += 1
                    all.append(m)
                    z = d.get(m[2], [])
                    z.append([ ci, m ])
                    d[m[2]] = z
                else:
                    print "LATE: %s %.1f %s" % (self.r[ci].ts(m[3]), m[1], m[2])
            x = self.bandinfo.get(bands[ci], 0)
            self.bandinfo[bands[ci]] = 0.5 * x + 0.5 * bandcount1

            f = open(self.bandname, "a")
            f.write("%s %s %d %2d %2d\n" % (self.r[ci].ts(now),
                                            bands[ci],
                                            ci,
                                            bandcount,
                                            bandcount1))
            f.close()

            self.msgs_index[ci] = len(msgs)

        # append each msg to jt65-all.txt.
        for txt in d:
            band = None
            got = ""
            hz = None
            snr = None
            nerrs = [ -1, -1 ] # for each antenna
            for [ ci, m ] in d[txt]:
                band = bands[ci]
                if not (str(ci) in got):
                    got += str(ci)
                hz = m[1]
                nerrs[ci] = m[4]
                if snr == None or m[5] > snr:
                    snr = m[5]

            info = "%s %s rcv %2s %2d %2d %3.0f %6.1f %s" % (self.r[ci].ts(m[3]),
                                                               band,
                                                               got,
                                                               nerrs[0],
                                                               nerrs[1],
                                                               snr,
                                                               m[1],
                                                               m[2])

            print info

            f = open(self.allname, "a")
            f.write(info + "\n")
            f.close()
            
            # send CQs that don't have too many errors to pskreporter.
            # the 30 here suppresses some good CQ receptions, but
            # perhaps better that than reporting erroneous decodes.
            if (nerrs[0] >= 0 and nerrs[0] < 30) or (nerrs[1] >= 0 and nerrs[1] < 30):
                txt = m[2]
                # normalize
                txt = re.sub(r'  *', ' ', txt)
                txt = re.sub(r'CQ DX ', 'CQ ', txt)
                mm = re.search(r'^CQ ([0-9A-Z/]+) ([A-R][A-R][0-9][0-9])$', txt)
                if mm != None and self.pskr != None:
                    hz = m[1] + int(b2f[band] * 1000000.0)
                    self.pskr.got(mm.group(1), hz, "JT65", mm.group(2), m[3])

        return all

    # return two bands to listen for CQs on.
    # specialized to two receivers. doesn't work for
    # just one receiver.
    def rankbands(self):
        global plausible

        if len(plausible) == 1:
            return [ plausible[0] ]

        # are we missing bandinfo for any bands?
        missing = [ ]
        for b in plausible:
            if self.bandinfo.get(b) == None:
                missing.append(b)

        # most profitable bands, best first.
        best = sorted(plausible, key = lambda b : -self.bandinfo.get(b, -1))

        # always explore missing bands first.
        if len(missing) >= len(self.r):
            return missing[0:len(self.r)]

        if len(missing) > 0:
            return [ best[0], missing[0] ]

        ret = [ ]
        if len(self.r) > 1:
            # two receivers.
            # always look at best band.
            ret.append(best[0])
            best = best[1:]

        if len(best) == 0:
            pass
        elif random.random() < 0.3 or self.bandinfo[best[0]] <= 0.1:
            band = random.choice(best)
            ret.append(band)
        else:
            wa = [ [ b, self.bandinfo[b] ] for b in best ]
            band = wchoice(wa, 1)[0]
            ret.append(band)

        return ret

    def one(self):
        if self.oneband != None:
            bands = [ self.oneband ] * len(self.r)
        else:
            # choose a band per receiver.
            bands = self.rankbands()

        bands = bands[0:len(self.r)]
        while len(bands) < len(self.r):
            bands.append(bands[0])

        # highest frequency on main receiver, so that low-pass ATU
        # will let us use main antenna on sub-receiver too.
        bands = sorted(bands, key = lambda b : int(b))

        if self.verbose:
            sys.stdout.write("band ")
            for b in bands:
                sys.stdout.write("%s " % (b))
            sys.stdout.write("; ")
            for b in self.bandinfo:
                sys.stdout.write("%s %.1f, " % (b, self.bandinfo[b]))
            sys.stdout.write("\n")
            sys.stdout.flush()

        if self.cat != None:
            for i in range(0, len(self.r)):
                self.cat.setf(i, int(b2f[bands[i]] * 1000000.0))

        time.sleep(55)

        # wait for the 59th second.
        while True:
            now = time.time()
            second = self.r[0].second(now)
            if second >= 59.5:
                break
            time.sleep(0.2)

        self.readall(now, bands)

    def go(self):
        # receive card(s)
        self.r = [ ]
        self.msgs_index = [ ]
        self.rth = [ ]
        for c in self.incards:
            r = jt65.JT65()
            self.r.append(r)
            r.cardrate = self.rate
            r.opencard(c)
            th = threading.Thread(target=lambda : r.gocard())
            th.daemon = True
            th.start()
            self.rth.append(th)
            self.msgs_index.append(0)

        while True:
            self.one()

    def close(self):
        for r in self.r:
            r.close()
        for th in self.rth:
            th.join()

def main():
    parser = weakargs.stdparse('Decode JT65A.')
    parser.add_argument("-band")
    parser.add_argument("-card2", nargs=2, metavar=('CARD', 'CHAN'))
    args = parser.parse_args()
    
    if args.levels == True:
        weakaudio.levels(args.card)
        
    if args.card == None:
        parser.error("jt65mon requires -card")
      
    if args.cat == None and args.band == None:
        parser.error("jt65mon needs either -cat or -band")

    jt65mon = JT65Mon(args.card, args.card2, args.cat, args.band)
    jt65mon.verbose = args.v
    jt65mon.go()
    jt65mon.close()
    sys.exit(0)

main()
