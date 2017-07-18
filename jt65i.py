#!/usr/local/bin/python

#
# interactive JT65.
# runs in a Linux or Mac terminal window.
# user can respond to CQs, but not send CQ.
# optional automatic band switching when not in QSO.
#
# I use jt65.pyi with a K3S via USB, automatically switching
# among bands:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23
#
# To switch among just a few bands:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -bands "30 20 17"
#
# To use on a single band without CAT (radio must be set up
# correctly already, and have VOX if you want to transmit):
# ./jt65i.py -card 2 0 -out 1 -band 30
#
# Select a CQ to reply to by typing the upper-case letter displayed
# next to the CQ. jt65i.py automates the rest of the exchange.
#
# jt65i.py can use multiple receivers, listening to a different band on
# each; when you reply to a CQ it automatically sets the transmitter to
# the correct band. This works for a K3 with a sub-receiver, or a K3
# with SDRs that jt65i.py knows how to control.
#
# For a K3 with a sub-receiver:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -bands "40 30 20"
#
# For a K3 without sub-receiver, and an RFSpace NetSDR/CloudIQ/SDR-IP:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 sdrip 192.168.3.130 -bands "40 30 20"
#
# For a K3 with sub-receiver and an RFSpace NetSDR/CloudIQ/SDR-IP (i.e. three receivers):
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -card3 sdrip 192.168.3.130 -bands "40 30 20"
#
# Robert Morris, AB1HL
#

#import fake65 as jt65
import jt65
import sys
import os
import time
import calendar
import numpy
import threading
import re
import random
import copy
import pyaudio # just for output, not input
import weakcat
import weakaudio
import pskreport
import weakutil
import weakargs
import tty
import fcntl
import termios
import struct

# automatically switch only among these bands.
# auto_bands = [ "160", "80", "60", "40", "30", "20", "17", "15", "12", "10" ]
auto_bands = [ "40", "30", "20", "17" ]

b2f = { "160" : 1.838, "80" : 3.576, "60" : 5.357, "40" : 7.076,
        "30" : 10.138, "20" : 14.076,
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
        assert len(x) == 2
        for e in x:
            counts[e] = counts.get(e, 0) + 1
    print(counts)

# load lotwreport.adi from LOTW, so we know what's been confirmed
# and thus who it makes sense to contact. returns a dictionary
# with band-call (and True for value).
#
# <APP_LoTW_OWNCALL:5>AB1HL
# <STATION_CALLSIGN:5>AB1HL
# <CALL:6>CE3CBM
# <BAND:3>20M
# <FREQ:8>14.07600
# <MODE:4>JT65
# <APP_LoTW_MODEGROUP:4>DATA
# <QSO_DATE:8>20160903
# <TIME_ON:6>234900
# <QSL_RCVD:1>Y
# <QSLRDATE:8>20160926
# <eor>
#
def read_lotw():
    try:
        f = open("lotwreport.adi", "r")
    except:
        return None

    d = { }
    call = None
    band = None
    mode = None
    qsl_rcvd = None

    while True:
        line = f.readline()
        if line == '':
            break
        line = line.upper()
        line = line.strip()
        m = re.match(r'^<([A-Z_]+):*[0-9]*>(.*)$', line)
        if m != None:
            k = m.group(1)
            v = m.group(2)
            if k == "CALL":
                call = v
            if k == "BAND":
                band = v
                band = re.sub(r'M$', '', band)
            if k == "MODE":
                mode = v
            if k == "QSL_RCVD":
                qsl_rcvd = v
        if line == "<EOR>":
            if call != None and band != None and mode[0:4] == "JT65" and qsl_rcvd == "Y":
                #print "+++ %s %s %s %s" % (call, band, mode, qsl_rcvd)
                d[band + "-" + call] = True
            #else:
            #    print "--- %s %s %s %s" % (call, band, mode, qsl_rcvd)
            call = None
            band = None
            mode = None
            qsl_rcvd = None
    f.close()
    return d

# returns [ rows, columns ] e.g. [ 24, 80 ]
# works on FreeBSD, Linux, and OSX.
def terminal_size(fd):
    cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
    return [ int(cr[0]), int(cr[1]) ]

class JT65I:
    def __init__(self, outcard, descs, cat, oneband):
        self.mycall = weakutil.cfg("jt65i", "mycall")
        self.mygrid = weakutil.cfg("jt65i", "mygrid")

        self.outcard = outcard
        self.oneband = oneband

        self.rate = 11025
        self.logname = "jt65-log.txt"
        self.allname = "jt65-all.txt"
        self.cqname = "jt65-cq.txt"
        self.jtname = "jt65"

        self.card_descs = descs

        if cat != None:
            self.cat = weakcat.open(cat)
            self.cat.sync()
            self.cat.set_usb_data()
        else:
            self.cat = None

        # self.bandinfo[band] is an array of [ minute, count ].
        # used to pick which band to listen to.
        self.bandinfo = { }

        # for each minute, for each card index, what band?
        self.minute_bands = { }

        self.prefixes = load_prefixes()

        # contacted call signs, grid counts, entity counts.
        # keys are band-call, band-grid, band-entity.
        # initialized from jt65-log.txt
        self.band_call = { }
        self.band_entity = { }
        self.all_entity = { }
        self.band_grid = { }
        self.wrote_cq = { }

        self.read_log()

        if False and self.mycall != None and self.mygrid != None:
            # talk to pskreporter.
            print("reporting to pskreporter as %s at %s" % (self.mycall, self.mygrid))
            self.pskr = pskreport.T(self.mycall, self.mygrid, "weakmon-0.3", False)
        else:
            print("not reporting to pskreporter since call/grid not in weak.cfg")
            self.pskr = None

        # all messages decoded.
        self.log = [ ] # log of all msgs received
        self.log_already = { } # to suppress duplicates in dlog, band-minute-txt

        # for display()
        self.dthis = [ ] # log of this conversation
        self.dhiscall = None # who we're trying to respond to
        self.dsending = False # TX vs RX
        self.dwanted = None # set to a call if user wants to respond
        self.dmessages = [ ] # messages to display to the user for a few seconds

    def decode_loop(self):
        while True:
            self.decode_new()
            time.sleep(0.5)

    # read latest msgs from all cards,
    # append to self.log[].
    def decode_new(self):
        minute = self.minute(time.time())
        oindex = len(self.log)

        # for each card, new msgs
        for ci in range(0, len(self.r)):
            msgs = self.r[ci].get_msgs()
            # each msg is a jt65.Decode.
            for dec in msgs:
                if self.r[ci].enabled:
                    dec.card = ci
                    dec.band = self.get_band(dec.minute, ci)
                    assert dec.band != None
                    ak = dec.band + "-" + str(dec.minute) + "-" + dec.msg
                    if not ak in self.log_already:
                        self.log_already[ak] = True
                        self.log.append(dec)
                        self.record_stat(dec)

        # append each msg to jt65-all.txt.
        # incorrectly omits some lines if a call transmits the same message
        # on multiple bands simultaneously, e.g. W5KDJ.
        for dec in self.log[oindex:]:
            f = open(self.allname, "a")
            ci = dec.card
            f.write("%s %s rcv %d %2d %3.0f %6.1f %s\n" % (self.r[ci].ts(dec.decode_time),
                                                          dec.band,
                                                          dec.card,
                                                          dec.nerrs,
                                                          dec.snr,
                                                          dec.hz(),
                                                          dec.msg))
            f.close()

            self.write_cq(dec.msg, dec.band, dec.snr, self.r[ci].ts(dec.decode_time))

            # send msgs that don't have too many errors to pskreporter.
            # the 30 here suppresses some good CQ receptions, but
            # perhaps better that than reporting erroneous decodes.
            if dec.nerrs >= 0 and dec.nerrs < 30:
                txt = dec.msg
                hz = dec.hz() + int(b2f[dec.band] * 1000000.0)
                tm = dec.decode_time
                self.maybe_pskr(txt, hz, tm)

    # if we've received a newish CQ, record it in jt65-cq.txt.
    def write_cq(self, txt, band, snr, ts):
        otxt = txt
        txt = txt.strip()
        txt = re.sub(r'  *', ' ', txt)
        txt = re.sub(r'CQ DX ', 'CQ ', txt)
        txt = re.sub(r'CQDX ', 'CQ ', txt)
        txt = re.sub(r'CQ NA ', 'CQ ', txt)
        txt = re.sub(r'CQ USA ', 'CQ ', txt)
        txt = re.sub(r'/QRP ', ' ', txt)

        call = None
        grid = None
        mm = re.search(r'^CQ ([0-9A-Z/]+) ([A-R][A-R][0-9][0-9])$', txt)
        if mm != None:
            call = mm.group(1)
            grid = mm.group(2)
        else:
            mm = re.search(r'^CQ ([0-9A-Z/]+) *([0-9A-Z/]*)', txt)
            if mm != None:
                if len(mm.group(1)) >= len(mm.group(2)):
                    call = mm.group(1)
                else:
                    call = mm.group(2)

        if call == None or not self.iscall(call):
            return

        k = band + "-" + call
        now = time.time()
        if k in self.wrote_cq and now - self.wrote_cq[k] < 6*3600:
            return
        self.wrote_cq[k] = now

        ng = 0
        if grid != None:
            ng = self.band_grid.get(self.gridkey(grid, band), 0)

        entity = look_prefix(call, self.prefixes)
        xentity = None
        ne = 0
        ae = 0
        if entity != None:
            ne = self.band_entity.get(self.entitykey(entity, band), 0)
            ae = self.all_entity.get(entity, 0)
            xentity = entity.replace(" ", "-")

        call_any = "N"
        for b in list(b2f.keys()):
            if (b + "-" + call) in self.band_call:
                call_any = "Y"

        if (band + "-" + call) in self.band_call:
            call_this = "Y"
        else:
            call_this = "N"

        f = open(self.cqname, "a")
        f.write("%s %s %4d %4d %3d %s%s %s %s\n" % (ts,
                                                    band,
                                                    ae,
                                                    ne,
                                                    ng,
                                                    call_any,
                                                    call_this,
                                                    xentity,
                                                    otxt))
        f.close()


    # report to pskreporter if we can figure out the originating
    # call and grid.
    def maybe_pskr(self, txt, hz, tm):
        txt = txt.strip()
        txt = re.sub(r'  *', ' ', txt)
        txt = re.sub(r'CQ DX ', 'CQ ', txt)
        txt = re.sub(r'CQDX ', 'CQ ', txt)
        mm = re.search(r'^CQ ([0-9A-Z/]+) ([A-R][A-R][0-9][0-9])$', txt)
        if mm != None and self.iscall(mm.group(1)):
            if self.pskr != None:
                self.pskr.got(mm.group(1), hz, "JT65", mm.group(2), tm)
            return
        mm = re.search(r'^([0-9A-Z/]+) ([0-9A-Z/]+) ([A-R][A-R][0-9][0-9])$', txt)
        if mm != None and self.iscall(mm.group(1)) and self.iscall(mm.group(2)):
            if self.pskr != None:
                self.pskr.got(mm.group(2), hz, "JT65", mm.group(3), tm)
            return

    # does call look syntactically correct?
    def iscall(self, call):
        if len(call) < 3:
            return False
        if re.search(r'[0-9][A-Z]', call) == None:
            # no digit
            return False
        if self.sender.packcall(call) == -1:
            # don't know how to encode this call, e.g. R870T
            return False
        return True

    # wait until the 59th second.
    # return array of Decodes from the almost-completed minute.
    # suppress=True means we sent during this minute, so
    # ignore any decodes.
    def wait59(self, suppress):
        while True:
            now = time.time()
            if self.seconds_left(now) <= 0.3:
                msgs = [ ]
                for dec in self.log[-30:]:
                    if dec.minute == self.minute(now):
                        msgs.append(dec)
                return msgs
            time.sleep(0.2)

    # is txt a CQ?
    # if yes, return [ call, grid ]
    # grid may be ""
    # otherwise return None
    def is_cq(self, txt):
        txt = txt.strip()
        txt = re.sub(r'CQ DX ', 'CQDX ', txt)
        txt = re.sub(r'CQDXDX ', 'CQDX ', txt)
        txt = re.sub(r'CQ NA ', 'CQ ', txt)
        txt = re.sub(r'CQ USA ', 'CQ ', txt)
        txt = re.sub(r'/QRP ', ' ', txt)
        txt = re.sub(r' DXDX$', ' DX', txt)
        txt = re.sub(r' DX DX$', ' DX', txt)

        # change CQ UR4UT DX to CQDX UR4UT
        mm = re.search(r'^CQ +([A-Z0-9]+) +DX$', txt)
        if mm != None:
            txt = "CQDX %s" % (mm.group(1))

        mm = re.search(r'^CQ +([A-Z0-9]+) *([A-Z0-9]*)$', txt)
        if mm != None and self.iscall(mm.group(1)):
            xc = mm.group(1) # his call
            xg = mm.group(2) # his grid, maybe ""
            return [ xc, xg ]

        mm = re.search(r'^CQDX +([A-Z0-9]+) *([A-Z0-9]*)$', txt)
        if mm != None and self.iscall(mm.group(1)):
            xc = mm.group(1) # his call
            xg = mm.group(2) # his grid, maybe ""
            return [ xc, xg ]

        return None

    # is the call in the our log, on any band?
    def call_in_log(self, call):
        for band in list(b2f.keys()):
            k = self.callkey(call, band)
            if k in self.band_call:
                return True
        return False

    # is call+band in our log?
    def band_call_in_log(self, call, band):
        k = self.callkey(call, band)
        return k in self.band_call

    # set up frequencies for a QSO minute during which we transmit.
    def qso_sending(self, band):
        minute = self.minute(time.time())
        for i in range(0, len(self.r)):
            self.set_band(minute, i, band)
            if i == 0:
                self.r[i].enabled = True
                thisband = band
            else:
                self.r[i].enabled = False
                # not band, to avoid receiver overload
                if band == "10":
                    thisband = "40"
                else:
                    thisband = "10"
            if self.rcat[i] != None:
                self.rcat[i].setf(i, int(b2f[thisband] * 1000000.0))
        if self.cat != None:
            self.cat.sync()

    # set up frequencies for a QSO minute during which we receive.
    def qso_receiving(self, band):
        minute = self.minute(time.time())
        for i in range(0, len(self.r)):
            self.set_band(minute, i, band)
            self.r[i].enabled = True
            if self.rcat[i] != None:
                self.rcat[i].setf(i, int(b2f[band] * 1000000.0))
        if self.cat != None:
            self.cat.sync()

    # what band was card ci tuned to during a particular minute?
    # returns a string, e.g. "20", or None.
    # returns info for most recent minute for which
    # any bands were set.
    def get_band(self, minute, ci):
        if minute in self.minute_bands and ci in self.minute_bands[minute]:
            return self.minute_bands[minute][ci]
        if len(self.minute_bands) == 0:
            return None
        for m in range(minute, minute-60, -1):
            if m in self.minute_bands:
                return self.minute_bands[m].get(ci, None)
        return None

    def set_band(self, minute, ci, band):
        if not minute in self.minute_bands:
            self.minute_bands[minute] = { }
        self.minute_bands[minute][ci] = band

    # incorporate a reception into per-band counts.
    def record_stat(self, dec):
        if dec.nerrs >= 25:
            return

        # self.bandinfo[band] is an array of [ minute, count ]

        if not dec.band in self.bandinfo:
            self.bandinfo[dec.band] = [ ]

        if len(self.bandinfo[dec.band]) == 0 or self.bandinfo[dec.band][-1][0] != dec.minute:
            self.bandinfo[dec.band].append( [ dec.minute, 0 ] )

        self.bandinfo[dec.band][-1][1] += 1

    # return a list of bands to listen to.
    def rankbands(self):
        # for each band, count of recent receptions.
        counts = { }
        for band in self.bandinfo:
            if len(self.bandinfo[band]) > 1:
                counts[band] = (self.bandinfo[band][-1][1] + self.bandinfo[band][-2][1]) / 2.0
            else:
                counts[band] = self.bandinfo[band][-1][1]

        # are we missing bandinfo stats for any bands?
        missing = [ ]
        for band in auto_bands:
            if counts.get(band) == None:
                missing.append(band)

        # most profitable bands, highest count first.
        best = sorted(auto_bands, key = lambda band : -counts.get(band, -1))

        # always explore missing bands first.
        if len(missing) >= len(self.r):
            return missing[0:len(self.r)]

        if len(missing) > 0:
            return best[0:len(self.r)-len(missing)] + missing

        ret = [ ]

        # choose weighted by activity, but
        # give a little weight even to dead bands.
        c = copy.copy(counts)
        for band in c:
            if c[band] > 5.0:
                # no band too high a weight
                c[band] = 5.0
        wsum = numpy.sum([ c[band] for band in c ])
        wsum = max(wsum, 1.0)
        minweight = wsum * 0.25 / len(c) # spend about 1/4 of the time on inactive bands
        wa = [ [ band, max(c[band], minweight*self.idleweight(band)) ] for band in c ]
        ret = wchoice(wa, len(self.r))

        return ret

    def idleweight(self, band):
        return 1.0

    def choose_bands(self, minute):
        if self.oneband != None:
            bands = [ self.oneband ]
        else:
            bands = self.rankbands()
            bands = bands[0:len(self.r)]

        while len(bands) < len(self.r):
            bands.append(bands[0])

        for band in bands:
            # trick rankbands() into counting each
            # band as missing just once.
            if not band in self.bandinfo:
                self.bandinfo[band] = [ [ 0, 0 ] ]

        for ci in range(0, len(self.r)):
            self.set_band(minute, ci, bands[ci])

        for ci in range(0, len(self.r)):
            band = bands[ci]
            if self.rcat[ci] != None:
                self.rcat[ci].setf(ci, int(b2f[band] * 1000000.0))
                self.rcat[ci].sync()

    # if the user has asked to respond to a CQ, do it.
    def one(self):
        self.dhiscall = None

        # wait for 59th second, get this minute's decodes,
        # also wait for user to choose a CQ to respond to.
        msgs = self.wait59(False)

        if self.dwanted == None:
            # user did not ask to respond to a CQ.
            # switch radio to new band.
            while self.second(time.time()) > 0.5:
                time.sleep(0.2)
            self.choose_bands(self.minute(time.time()))
            return

        cq = None
        for dec in msgs:
            cqinfo = self.is_cq(dec.msg)
            if cqinfo != None and cqinfo[0] == self.dwanted:
                dec.hiscall = cqinfo[0]
                dec.hisgrid = cqinfo[1]
                cq = dec

        if cq == None:
            self.show_message("No CQ from %s." % (self.dwanted))
            self.dwanted = None
            time.sleep(2)
            return

        self.dwanted = None

        self.show_message("Replying to %s." % (cq.hiscall))

        entity = look_prefix(cq.hiscall, self.prefixes)

        # reply off-frequency in case others reply too.
        myhz = cq.hz() + (random.random() * 300) - 150
        if myhz < 211:
            myhz = 211
        if myhz > 2200:
            myhz = 2200

        self.dhiscall = cq.hiscall
        self.dthis = [ ]
        self.dthis.append("<<< %s" % (cq.msg))
        
        # listen for his signal report.
        # if we hear nothing from him (including no msg to other call),
        # then sleep a minute and listen again. this situation comes
        # up fairly frequently.
        hissig = None
        olen = len(self.log)
        for tries in range(0, 2):
            if tries > 0:
                self.show_message("No reply from %s, trying once more." % (cq.hiscall))

            self.qso_sending(cq.band) # disable receivers
            self.send(cq.band, myhz, "%s %s %s" % (cq.hiscall,
                                                self.mycall,
                                                self.mygrid))

            self.wait59(True)
            self.qso_receiving(cq.band) # enable receivers
            time.sleep(3)

            # wait for 59th second, look for AB1HL HISCALL -YY
            while self.seconds_left(time.time()) > 0.3:
                if self.dwanted != None:
                    # user wants to switch to another CQ.
                    self.show_message("Terminating contact with %s." % (self.dhiscall))
                    return
                while olen < len(self.log):
                    m = self.log[olen]
                    olen += 1
                    if m.minute != self.minute(time.time()):
                        continue
                    txt = m.msg.strip()
                    txt = re.sub(r'  *', ' ', txt)
                    txt = re.sub(r'CQ DX ', 'CQ ', txt)
                    txt = re.sub(r'CQDX ', 'CQ ', txt)
                    txt = re.sub(r'CQ NA ', 'CQ ', txt)
                    txt = re.sub(r'CQ USA ', 'CQ ', txt)
                    txt = re.sub(r'/QRP ', ' ', txt)
                    a = txt.split(' ')
                    if len(a) == 3 and a[0] == self.mycall and a[1] == cq.hiscall:
                        # AB1HL HISCALL -YY
                        if hissig == None:
                            self.dthis.append("<<< %s" % (m.msg))
                            self.show_message("%s replied." % (cq.hiscall))
                        hissig = a[2]
                    elif len(a) == 2 and a[0] == self.mycall and ("-" in a[1]):
                        # AB1HL -YY
                        if hissig == None:
                            self.dthis.append("<<< %s" % (m.msg))
                            self.show_message("%s replied." % (cq.hiscall))
                        hissig = a[1]
                    if hissig == None and len(a) >= 2 and a[0] != self.mycall and a[1] == cq.hiscall:
                        # he responded to someone else, or he CQ'd again.
                        # OK to try again soon.
                        if a[0] == "CQ":
                            # he CQd again.
                            self.show_message("%s CQ'd again." % (cq.hiscall))
                        else:
                            self.show_message("%s responded to %s." % (cq.hiscall, a[0]))
                            return
                time.sleep(0.2)

            if hissig != None:
                break

        if hissig == None:
            self.show_message("%s did not respond." % (cq.hiscall))
            time.sleep(2)
            return

        snr = int(cq.snr)
        if snr > -1:
            snr = -1

        for tries in range(0, 3):
            self.qso_sending(cq.band) # disable receivers
            self.send(cq.band, myhz, "%s %s R-%02d" % (cq.hiscall, self.mycall, -snr))
            self.wait59(True)
            self.qso_receiving(cq.band) # enable receivers
            time.sleep(3)
            
            # wait for AB1HL HISCALL RRR or 73.
            # or AB1HL RRR 73
            # but try once more if we don't see RRR/73,
            # e.g. if he (re)sends AB1HL XXX -04.
            msgs = self.wait59(False)
            if self.dwanted != None:
                # user wants to switch to another CQ.
                self.show_message("Terminating contact with %s." % (self.dhiscall))
                return
            found = False
            got = None
            for m in msgs:
                txt = m.msg.strip()
                txt = re.sub(r'  *', ' ', txt)
                has73 = ("73" in txt or "RRR" in txt or "TU" in txt)
                if has73 and (self.mycall in txt):
                    found = True
                    got = m.msg
                if has73 and (abs(m.hz() - myhz) < 10 or abs(m.hz() - cq.hz()) < 10):
                    found = True
                    got = m.msg
            if got != None:
                self.dthis.append("<<< %s" % (got))
            if found:
                break

        self.show_message("Logging %s on %s meters." % (cq.hiscall, cq.band))
        self.write_log(cq.band, cq.hz(), cq.hiscall, hissig, cq.hisgrid, snr)
        self.qso_sending(cq.band) # disable receivers
        self.send(cq.band, myhz, "%s %s 73" % (cq.hiscall, self.mycall))
        self.wait59(True)
        self.qso_receiving(cq.band) # enable receivers

        return

    # generate audio signal for sending.
    def audio(self, hz, msg):
        twelve = self.sender.pack(msg)
        x1 = self.sender.send12(twelve, hz, self.snd_rate, self.snd_symsamples)

        # keep levels well below max sound card output.
        # but RigBlaster needs this to be at least 0.3 to
        # trigger VOX, maybe 0.4.
        # actually what the RigBlaster needs depends on audio
        # frequency; to work down to 200 hz you need 0.9 here.
        x1 *= 0.3

        # sound card expects 16-bit signed ints.
        x1 *= 32767.0
        x1 = x1.astype(numpy.int16)

        return x1

    # write audio samples to sound card for output.
    def tocard(self, x1):
        if self.cycle_length() < 60:
            # fake the send
            self.show_message("tocard() not actually sending.")
            time.sleep(4)
            return

        # write in smallish chunks so control-C works.
        blocksize = self.snd_rate
        i = 0
        while i < len(x1):
            x2 = x1[i:i+blocksize]
            x2 = x2.tostring()
            self.snd.write(x2)
            i += blocksize

    # msg is e.g. "CQ AB1HL FN42".
    # send out the sound card.
    # returns after 49 seconds (or just the time to send
    # the samples, not the whole 60 seconds).
    def send(self, band, hz, msg):
        self.dthis.append(">>> %s" % (msg))

        # this stuff takes a noticeable amount of CPU, so do it
        # before waiting until one second after the minute.
        x1 = self.audio(hz, msg)

        # wait for 1 second after the minute
        # XXX if after 1 second already, should start
        # right away at the appropriate place in x1.
        while True:
            s = self.second(time.time())
            if s >= 0.9 and s < 2:
                break
            lf = self.seconds_left(time.time())
            if lf > 2 and s > 2:
                print("too late seconds=%.2f" % (s))
                return
            time.sleep(0.1)

        f = open(self.allname, "a")
        f.write("%s %s snd %6.1f %s\n" % (self.ts(time.time()), band, hz, msg))
        f.close()
        
        self.dsending = True
        self.tocard(x1)
        self.dsending = False

    def write_log(self, band, hz, call, sig, grid, snr):
        key = self.callkey(call, band)
        self.band_call[key] = time.time()

        entity = look_prefix(call, self.prefixes)
        if entity != None:
            ek = self.entitykey(entity, band)
            self.band_entity[ek] = self.band_entity.get(ek, 0) + 1
            self.all_entity[entity] = self.all_entity.get(entity, 0) + 1
        if grid != "":
            gk = self.gridkey(grid, band)
            self.band_grid[gk] = self.band_grid.get(gk, 0) + 1

        freq = b2f[band]

        ts = self.ts(time.time() - 90)
        ts = re.sub(r':[0-9][0-9]$', '', ts) # delete seconds

        f = open(self.logname, "a")
        f.write("%-9s %s %6.3f 599 JT X     %s, %s, %s, %.0f, %s\n" % (call,
                                                                    ts,
                                                                    freq,
                                                                    entity,
                                                                    grid,
                                                                    sig,
                                                                    snr,
                                                                    self.jtname))
        f.close()

    # turn a frequency in MHz into a band like "40".
    def f2b(self, freq):
        try:
            mhz = int(float(freq))
        except:
            return None
        if mhz == 1:
            return "160"
        if mhz == 3:
            return "80"
        if mhz == 5:
            return "60"
        if mhz == 7:
            return "40"
        if mhz == 10:
            return "30"
        if mhz == 14:
            return "20"
        if mhz == 18:
            return "17"
        if mhz == 21:
            return "15"
        if mhz == 24:
            return "12"
        if mhz == 28:
            return "10"
        if mhz == 50:
            return "6"
        sys.stderr.write("jt65i f2b(%s) no band\n" % (freq))
        return None

    # derive key for self.band_call[].
    # band-call or just call.
    def callkey(self, call, band):
        if band != None:
            return band + "-" + call
        else:
            return call

    def gridkey(self, grid, band):
        if band != None:
            return band + "-" + grid
        else:
            return grid

    def entitykey(self, entity, band):
        if band != None:
            return band + "-" + entity
        else:
            return entity

    # populate self.band_call[] &c from log file.
    # only increment grid/entity counts for
    # lotw-confirmed contacts.
    def read_log(self):
        # either None or dict of band-call
        lotw = None
        #lotw = read_lotw()
        #if lotw == None:
        #    print("could not read lotwreport.adi")

        inlotw = 0
        notinlotw = 0

        f = open(self.logname, "r")
        while True:
            l = f.readline()
            if l == "":
                break
            l = re.sub(r'  *', ' ', l)
            a = l.split(" ")

            band = self.f2b(a[3])

            # parse the time, 15/02/17 17:51
            da = a[1].split("/")
            ta = a[2].split(":")
            if len(da) == 3 and len(ta) >= 2:
                tm = time.struct_time([
                     int(da[2]) + 2000, # year
                     int(da[1]), # mon
                     int(da[0]), # mday
                     int(ta[0]), # hour
                     int(ta[1]), # min
                     0, # sec
                     0, # wday
                     0, # yday
                     -1]) # isdst
                secs = calendar.timegm(tm)
            else:
                secs = time.time()

            key = self.callkey(a[0], band)
            self.band_call[key] = secs

            if lotw == None or key in lotw:
                inlotw += 1
                entity = look_prefix(a[0], self.prefixes)
                if entity != None:
                    ek = self.entitykey(entity, band)
                    self.band_entity[ek] = self.band_entity.get(ek, 0) + 1
                    self.all_entity[entity] = self.all_entity.get(entity, 0) + 1
                m = re.search(r', ([A-Z][A-Z][0-9][0-9]), ', l)
                if m != None:
                    grid = m.group(1)
                    gk = self.gridkey(grid, band)
                    self.band_grid[gk] = self.band_grid.get(gk, 0) + 1
            else:
                notinlotw += 1
        sys.stdout.flush()
        f.close()
        if lotw != None:
            print("%d in lotw, %d not in lotw" % (inlotw, notinlotw))

    # open sound card, create jt65 instance.
    def soundsetup(self):
        # receive card(s)
        self.r = [ ]
        self.rcat = [ ]
        self.rth = [ ]
        for ci in range(len(self.card_descs)):
            desc = self.card_descs[ci]
            r = jt65.JT65()
            self.r.append(r)
            r.cardrate = self.rate
            r.opencard(desc)
            th = threading.Thread(target=lambda r=r: r.gocard())
            th.daemon = True
            th.start()
            self.rth.append(th)
            if ci == 0:
                self.rcat.append(self.cat)
            elif ci == 1 and self.cat != None and self.cat.type == "k3" and desc[0][0].isdigit():
                # assume -card2 is the K3 sub-receiver.
                self.rcat.append(self.cat)
            else:
                self.rcat.append(weakcat.open(desc))

        # send card
        if self.outcard != None:
            # we want 11025, but some cards don't support it.
            outcard = int(self.outcard)
            self.snd_rate = weakaudio.pya_output_rate(outcard, 11025)
            if self.snd_rate == 44100:
                self.snd_rate = 48000
                print("for rigblaster, output rate %d rather than 44100 or 11025" % (self.snd_rate))
            elif self.snd_rate != 11025:
                print("using rate %d rather than 11025" % (self.snd_rate))
            self.snd_symsamples = int(4096 * (self.snd_rate / 11025.0))
            self.snd = weakaudio.pya().open(format=pyaudio.paInt16,
                                            output_device_index=outcard,
                                            channels=1,
                                            rate=self.snd_rate,
                                            output=True)
        self.sender = jt65.JT65Send()

    def go(self):
        self.soundsetup()

        # buffer of keystrokes from user input.
        self.keybuf = ""
        self.keybuf_lock = threading.Lock()

        # show information to user in terminal.
        th = threading.Thread(target=self.display_loop)
        th.daemon = True
        th.start()

        # accept keyboard input.
        th = threading.Thread(target=self.keyboard_loop)
        th.daemon = True
        th.start()

        # absorb decodes from radio into self.log.
        th = threading.Thread(target=self.decode_loop)
        th.daemon = True
        th.start()

        while True:
            self.one()

    # generate near-continuous sound-card output,
    # mimicing send().
    def test_send(self):
        self.soundsetup()

        msg = "T3ST AB1HL FN42"
        x1 = self.audio(910, msg)
        while True:
            sys.stdout.write(">>> " + msg)
            sys.stdout.flush()
            self.tocard(x1)
            sys.stdout.write("\n")
            time.sleep(2)

    def close(self):
        for r in self.r:
            r.close()
        for th in self.rth:
            th.join()

    # show a message to the user at the bottom of the screen for a few seconds.
    def show_message(self, txt):
        self.dmessages.append([ None, txt ])

    def keyboard_loop(self):
        tty.setcbreak(sys.stdin)
        while True:
            ch = sys.stdin.read(1)
            self.keybuf_lock.acquire()
            self.keybuf = self.keybuf + ch
            self.keybuf_lock.release()

    def display_loop(self):
        while True:
            self.display()
            time.sleep(1)

    # dd/mm/yy hh:mm:ss
    def ts(self, t):
        return self.r[0].ts(t)

    # HH:MM:SS
    # UTC
    def sts(self, t):
      gm = time.gmtime(t)
      return "%02d:%02d:%02d" % (
          gm.tm_hour,
          gm.tm_min,
          gm.tm_sec)

    # really integer cycle number.
    # t is UNIX time in seconds.
    def minute(self, t):
        return self.r[0].minute(t)

    def second(self, t):
        return self.r[0].second(t)

    def seconds_left(self, t):
        return self.r[0].seconds_left(t)

    def cycle_length(self):
        return self.r[0].cycle_length()

    # redraw display for user.
    def display(self):
        now = time.time()
        minute = self.minute(now)
        [ rows, cols ] = terminal_size(1)
        cols -= 1 # so that full-width line doesn't wrap.

        # clear the screen
        sys.stdout.write("\033[H") # home
        sys.stdout.write("\033[2J") # clear

        # status line
        if self.dhiscall != None:
            st = "Contacting: %s" % (self.dhiscall)
        else:
            st = "Contacting: ----"
        
        if self.dsending:
            txrx = "%s TX" % (self.get_band(minute, 0))
        else:
            txrx = ""
            for ci in range(0, len(self.r)):
                txrx += "%s " % (self.get_band(minute, ci))
            txrx += "RX"

        ts = "%s %s" % (txrx, self.sts(now))
        half1 = int(cols / 2)
        half2 = half1
        if half1+half2 == cols-1:
            half1 += 1
        print "%-*.*s%*.*s" % (half1, half1, st, half2, half2, ts)
        print "-" * cols

        # recent msgs from this conversation
        conv_lines = 6
        n = 0
        for e in self.dthis[-conv_lines:]:
            print e
            n += 1
        while n < conv_lines:
            print
            n += 1
        print "-" * cols

        # all recent msgs.
        # mark CQs.
        all_lines = rows - conv_lines - 5
        cqi = 0
        cqcalls = { } # map A..Z to CQ'ing call, for user keystroke
        n = 0
        for dec in self.log[-all_lines:]:
            start = ""
            entity = ""

            cqinfo = self.is_cq(dec.msg)
            if cqinfo != None:
                call = cqinfo[0]
                if dec.minute == minute:
                    # display a letter beside each fresh CQ,
                    # so user can select one with a keystroke.
                    cqch = chr(ord('A') + cqi)
                    start += cqch
                    cqcalls[cqch] = call
                    cqi += 1
                    if not self.call_in_log(call):
                        # mark CQs not already in log with *.
                        start += "*"
                entity = look_prefix(call, self.prefixes)
                if entity == None:
                    entity = ""
            print("%-3s %s %-3s %-20s %s" % (start, self.sts(dec.decode_time), dec.band, dec.msg, entity))
            n += 1
        while n < all_lines:
            print
            n += 1
        print "-" * cols

        # one line of general message display.
        # "persist" seconds per message.
        if len(self.dmessages) > 1:
            persist = 5
        else:
            persist = 10
        if len(self.dmessages) > 0 and self.dmessages[0][0] != None and now-self.dmessages[0][0] > persist:
            self.dmessages = self.dmessages[1:]
        if len(self.dmessages) > 0:
            if self.dmessages[0][0] == None:
                # remember when we first started displaying it,
                # so we can erase it after "persist" seconds.
                self.dmessages[0][0] = now
            sys.stdout.write(self.dmessages[0][1])

        # look for a keystroke, which asks us to
        # respond to one of the CQs displayed above.
        self.keybuf_lock.acquire()
        while self.keybuf != "":
            ch = self.keybuf[0]
            self.keybuf = self.keybuf[1:]
            if ch in cqcalls:
                if self.outcard == None:
                    self.show_message("You must specify a sound card with -out.")
                elif self.mycall == None:
                    self.show_message("You must set mycall in weak.cfg.")
                elif self.mygrid == None:
                    self.show_message("You must set mygrid in weak.cfg.")
                else:
                    # ask one() to respond to this call.
                    self.dwanted = cqcalls[ch]
        self.keybuf_lock.release()
        if self.dwanted != None and self.dhiscall == None:
            self.dhiscall = self.dwanted

        sys.stdout.flush()

def main():
    global auto_bands

    parser = weakargs.stdparse('JT65A.')
    parser.add_argument("-band")
    parser.add_argument("-bands")
    parser.add_argument("-card2", nargs=2, metavar=('CARD', 'CHAN'))
    parser.add_argument("-card3", nargs=2, metavar=('CARD', 'CHAN'))
    parser.add_argument("-card4", nargs=2, metavar=('CARD', 'CHAN'))
    parser.add_argument("-out", metavar="CARD")
    parser.add_argument("-test", action='store_true')
    args = weakargs.parse_args(parser)

    descs = [ ]
    if args.card != None:
        descs.append(args.card)
        if args.card2 != None:
            descs.append(args.card2)
            if args.card3 != None:
                descs.append(args.card3)
                if args.card4 != None:
                    descs.append(args.card4)

    if args.test:
        if args.out == None:
            parser.error("-test requires -out")
        jt65i = JT65I(args.out, descs,
                      args.cat, args.band)
        jt65i.test_send()
        sys.exit(0)
        
    if args.card == None:
        parser.error("jt65i requires -card")
      
    if args.cat == None and args.band == None:
        parser.error("jt65i needs either -cat or -band")

    # -band "40 30 20"
    # sets the list of bands among which to automatically switch.
    if args.bands != None:
        bands = args.bands.strip()
        bands = re.sub(r'  *', ' ', bands)
        a = bands.split(" ")
        for band in a:
            if not band in b2f:
                parser.error("band %s not recognized" % (band))
        auto_bands = a

    jt65i = JT65I(args.out, descs,
                  args.cat, args.band)

    jt65i.go()
    jt65i.close()
    sys.exit(0)

main()

