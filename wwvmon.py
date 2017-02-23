#!/usr/local/bin/python

#
# decode WWWV transmissions.
# Robert Morris, AB1HL
#
# set radio to AM, 2.5/5/10/15/20 MHz.
#

import weakargs
import weakutil
import weakaudio
import sys
import time
import wave
import numpy
import scipy

# optimizable tuning parameters.
filterwidth = 20 # Hz, 20.
filterorder = 3 # 100 hz bandpass filter, 3.
slicewin = 0 # seconds, 0=>off, 4.
slicesmooth = 0.2 # as a function of slicewin, 0=off, 0.0.
slicethresh = 0.25 # frac of min..max, 0.25.
smoothwin = 0.02 # smooth rectified samples, in seconds, 0.05.
topsecs = 4 # look at the best N seconds per 30 seconds, 2.
votewin = 3 # vote per bit over a window of this many minutes

# there are some tones that are constant; look for them
# to sync to minute.
# 0 means a constant zero bit,
# 1 means a constant 1 bit,
# 2 means a constant marker,
# -1 means zero or one, but unpredictable (but not marker).
# -2 means no tone at all (first second).
sync = [ -2, 0, -1, -1, -1, -1, -1, -1, 0, # 00..08
         2, -1, -1, -1, -1, -1, -1, -1, -1, 0, # 09..18
         2, -1, -1, -1, -1, -1, -1, -1, 0, 0, # 19..28
         2, -1, -1, -1, -1, -1, -1, -1, -1, -1, # 29..38
         2, -1, -1, 0, 0, 0, 0, 0, 0, 0, # 39..48
         2, -1, -1, -1, -1, -1, -1, -1, -1, -1, # 49..58
         2 ] # 59

# http://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
def smooth(values, window):
    oavg = numpy.mean(abs(values))
    #weights = numpy.repeat(1.0, window)/window
    weights = numpy.hamming(window)
    sma = numpy.convolve(values, weights, 'valid')
    sma = sma[0:len(values)]
    navg = numpy.mean(abs(sma))
    sma = sma * (oavg / navg)
    return sma

class Decode:
    def __init__(self):
        self.bits = None # save the 60 bits
        self.strengths = None
        self.dst1 = None
        self.leap = None
        self.year = None # since 0 C.E., e.g. 2017
        self.minute = None
        self.hour = None
        self.day_of_year = None # 1 = Jan 1
        self.dut = None # UT1 - UTC in seconds
        self.dst2 = None
        self.cardtime = None # seconds, relative to sound card or start of .wav file

    def s(self):
        x = "%02d %03d %02d:%02d" % (self.year,
                                     self.day_of_year,
                                     self.hour,
                                     self.minute)
        return x

    # minutes since year 0.
    # pretty bogus since doesn't know e.g. leap years.
    def tomin(self):
        x = 0
        x += self.year * 365 * 24 * 60
        x += (self.day_of_year - 1) * 24 * 60
        x += self.hour * 60
        x += self.minute
        return x

    # encode into 60 bits; returns an array of 0/1/2.
    def encode(self):
        bits = ( [] +
                 [ 0 ] +
                 [ 0 ] +
                 ebool(self.dst1) +
                 ebool(self.leap) +
                 e4((self.year - 2000) % 10) +
                 [ 0 ] +
                 [ 2 ] +
                 e4(self.minute % 10) +
                 [ 0 ] +
                 e3(self.minute / 10) +
                 [ 0 ] +
                 [ 2 ] +
                 e4(self.hour % 10) +
                 [ 0 ] +
                 e2(self.hour / 10) +
                 [ 0, 0 ] +
                 [ 2 ] +
                 e4(self.day_of_year % 10) +
                 [ 0 ] +
                 e4((self.day_of_year / 10) % 10) +
                 [ 2 ] +
                 e2(self.day_of_year / 100) +
                 [ 0, 0, 0, 0, 0, 0, 0 ] +
                 [ 2 ] +
                 ebool(self.dut >= 0) +
                 e4((self.year - 2000) / 10) +
                 ebool(self.dst2) +
                 e3(int(abs(self.dut*10))) +
                 [ 2 ]
                 )

        assert len(bits) == 60
        return bits

def ebool(b):
    if b:
        return [ 1 ]
    else:
        return [ 0 ]

def e1(x):
    assert x >= 0 and x <= 1
    return [ x & 1 ]

def e2(x):
    assert x >= 0 and x < 4
    return [ (x >> 0) & 1,
             (x >> 1) & 1 ]

def e3(x):
    assert x >= 0 and x < 8
    return [ (x >> 0) & 1,
             (x >> 1) & 1,
             (x >> 2) & 1 ]

def e4(x):
    assert x >= 0 and x < 16
    return [ (x >> 0) & 1,
             (x >> 1) & 1,
             (x >> 2) & 1,
             (x >> 3) & 1 ]

# turn four bits into a BCD digit 0..9.
def d4(bits):
    assert len(bits) == 4
    x = (bits[0] * 1 +
         bits[1] * 2 +
         bits[2] * 4 +
         bits[3] * 8)
    if x > 9:
        return 9
    return x

def d3(bits):
    assert len(bits) == 3
    x = (bits[0] * 1 +
         bits[1] * 2 +
         bits[2] * 4)
    if x > 7:
        return 7
    return x

def d2(bits):
    assert len(bits) == 2
    x = (bits[0] * 1 +
         bits[1] * 2)
    if x > 3:
        return 3
    return x

# convert minutes since year 0 back to a Decode with year/day/hour/minute.
def frommin(m):
    savem = m
    d = Decode()
    d.year = m / (365 * 24 * 60)
    if d.year < 2000:
        d.year = 2000
    m = m % (365 * 24 * 60)
    d.day_of_year = 1 + m / (24 * 60)
    m = m % (24 * 60)
    d.hour = m / 60
    m = m % 60
    d.minute = m
    assert m < 60
    #assert d.tomin() == savem
    return d

class WWV:

    def __init__(self):
        global filterwidth
        self.center = 100 # tone at 100 Hz
        self.filterwidth = filterwidth
        self.lorate = 315 # downsample to this, samples/second
                          # 300 works, but 200 does not!
                          # 441 is 11025/25, so resample() is fast.
                          # 315 is 11025/35, so resample() is fast.

        self.ssamples = [ ]
        self.ssampleslen = 0
        self.cb = None

        assert len(sync) == 60

    # tm0 is the time in seconds of buf[0].
    # if from a sound card, it's UNIX tim.
    # if from a .wav file, it's relative to start of file.
    def process(self, buf, eof, tm0):
        global filterorder, votewin

        # correct back to start of self.ssamples[]
        tm0 -= self.ssampleslen / float(self.inrate)

        self.ssamples.append(buf)
        self.ssampleslen += len(buf)

        while True:
            if self.ssampleslen < 60 * self.inrate:
                break

            if eof == False and self.ssampleslen < (votewin+1)*60*self.inrate:
                break

            samples = numpy.concatenate(self.ssamples)
            self.ssamples = None
            self.ssampleslen = None

            filter = weakutil.butter_bandpass(self.center - self.filterwidth/2,
                                              self.center + self.filterwidth/2,
                                              self.inrate, filterorder)
            filtered = scipy.signal.lfilter(filter[0], filter[1], samples)

            # down-sampling makes everything run much faster.
            # XXX perhaps sacrificing fine alignment?
            down = weakutil.resample(filtered, self.inrate, self.lorate)
            self.process1(down, tm0)

            trim = 60*self.inrate
            samples = samples[trim:]
            self.ssamples = [ samples ]
            self.ssampleslen = len(samples)
            tm0 += trim / float(self.inrate)

    def wt(self, filename, data):
        f = open(filename, "w")
        numpy.savetxt(f, data, "%f")
        f.close()

    # generate a bit of amplitude, to be used to
    # generate sync tone. highlen is length (in seconds)
    # of high-amplitude tone. 0.170, 0.470, or 0.770.
    def mksyncbit(self, highlen, high):
        sec = numpy.array([])
        sec = numpy.append(sec, numpy.repeat([-1.0], 0.030*self.lorate))
        sec = numpy.append(sec, numpy.repeat([1.0], 0.170*self.lorate))
        sec = numpy.append(sec, numpy.repeat([high], (highlen-0.170)*self.lorate))
        sec = numpy.append(sec, numpy.repeat([-1.0], (1.0-0.030-highlen)*self.lorate))
        if len(sec) < self.lorate:
            sec = numpy.append(sec, numpy.zeros(self.lorate - len(sec)))
        if len(sec) > self.lorate:
            sec = sec[0:self.lorate]
        return sec

    # slice the rectified smoothed 100 hz tone,
    # so that low amplitude is < 0 and high amplitude is > 0.
    # looks at a local window of a few seconds because
    # the levels change over time.
    def slice(self, smoothed):
        global slicewin, slicesmooth, slicethresh

        if slicewin < 0.001:
            # slicewin=0 disables fancy slicer. this actually works
            # pretty well, despite no adaptation and not setting
            # a specific min..max level.
            return smoothed - numpy.mean(smoothed)

        # look for min/max over this many samples.
        # XXX not clear what the right value is.
        win = int(slicewin * self.lorate)

        z = smoothed
        if (len(z) % win) != 0:
            # trim z so it's a multiple of win long.
            z = z[0:-(len(z)%win)]

        zsplit = numpy.split(z, len(z)/win) # split into win-size pieces
        maxes = numpy.amax(zsplit, axis=1) # max of each piece
        mins = numpy.amin(zsplit, axis=1)
        
        ii = numpy.arange(0, len(z), 1)
        ii = ii / win
        maxv = maxes[ii]
        minv = mins[ii]

        # smooth to avoid discontinuities at piece boundaries.
        # XXX what window here?
        if slicesmooth > 0.001:
            maxv = smooth(maxv, int(slicesmooth * slicewin * self.lorate))
            minv = smooth(minv, int(slicesmooth * slicewin * self.lorate))
        
        if len(maxv) < len(smoothed):
            maxv = numpy.append(maxv, maxv[0:(len(smoothed)-len(maxv))])
            minv = numpy.append(minv, minv[0:(len(smoothed)-len(minv))])
        elif len(maxv) > len(smoothed):
            maxv = maxv[0:len(smoothed)]
            minv = minv[0:len(smoothed)]

        # there's much more low-amplitude than high-amplitude,
        # so don't move the signal down so much.
        # XXX the factor here is a bit arbitrary.
        midv = minv + (maxv - minv) * slicethresh
        sliced = numpy.subtract(smoothed, midv)
        return sliced
        
    # tm0 is UNIX time in seconds of first sample for sound card input.
    # or time in seconds since start of .wav file.
    def process1(self, s, tm0):
        global smoothwin, topsecs, votewin

        s = abs(s)

        s = smooth(s, int(self.lorate * smoothwin))

        # center on zero so that correlation works better,
        # both for sync and later to decode bits.
        # basically sets the slicing level to zero.
        s = self.slice(s)

        # turn sync[] into amplitudes at self.lorate.
        syncsamples = numpy.array([])
        bitmap = { }
        bitmap[0] = self.mksyncbit(0.170, 1.0)
        bitmap[1] = self.mksyncbit(0.470, 1.0)
        bitmap[2] = self.mksyncbit(0.770, 1.0)
        bitmap[-1] = self.mksyncbit(0.470, 0.0)
        bitmap[-2] = numpy.repeat([-1], self.lorate) # no tone
        for ss in sync:
            sec = bitmap[ss]
            syncsamples = numpy.append(syncsamples, sec)

        # concatenate votewin copies of sync,
        # hoping to get better alignment and
        # be more likely to choose the correct minute boundary.
        # s[] contains multiple minutes due to voting.
        nsync = syncsamples
        for i in range(1, int(round(len(s)/float(60*self.lorate)))-2):
            nsync = numpy.append(nsync, syncsamples)

        # for each second, how much does it look like the
        # start of a minute?
        i = 0
        secs = [ ] # strength of correlation of sync sequence for this second.
        secsi = [ ] # sample offset within second of strongest.
        #ccc = numpy.correlate(s, nsync)
        ccc = numpy.correlate(s[0:60*self.lorate+len(nsync)], nsync)
        while i < 60*self.lorate and i + 61*self.lorate <= len(s):
            cc = ccc[i:i+self.lorate]
            mi = numpy.argmax(cc)
            secs.append(cc[mi])
            secsi.append(numpy.argmax(cc))
            i += self.lorate

        # sort the seconds, highest sync correlation first.
        ranked = sorted(range(0, len(secs)), key = lambda i : -secs[i])

        # accumulate [ Decode, votes ] pairs,
        # to choose the best second offset.
        allmins = [ ]

        for sec in ranked[0:topsecs]:
            off = sec*self.lorate + secsi[sec]
            ss = s[off:off+(votewin*60*self.lorate)]
            [ d, votes ] = self.process2(ss)
            if d != None:
                d.cardtime = tm0 + off/float(self.lorate)
                allmins.append([ d, votes ])

        if len(allmins) > 0:
            # sort by votes
            allmins = sorted(allmins, key = lambda dv : -dv[1])
            [ d, votes ] = allmins[0]
            if self.cb != None:
                self.cb(d)
            else:
                ts = time.gmtime(int(d.cardtime))
                fr = d.cardtime - int(d.cardtime)
                frs = "%.3f" % (fr)
                frs = frs[1:] # drop leading zero
                print "%02d:%02d:%02d%s %s %d" % (ts.tm_hour, ts.tm_min, ts.tm_sec, frs, d.s(), votes)
            sys.stdout.flush()

    # s is strength of tone, self.lorate samples per second,
    # rectified and smoothed.
    # caller thinks a minute begins at s[0].
    # return [ Decode, votes ]
    def process2(self, s):
        global votewin

        da = [ ]
        i = 0
        while (i+1)*60*self.lorate <= len(s):
            ss = s[i*60*self.lorate:(i+1)*60*self.lorate]
            [ bits, strengths ] = self.demod(ss)
            d = self.decode(bits)
            d.strengths = strengths
            if False and d.year == 2017 and d.day_of_year == 41:
                print d.s()
                print strengths
                sys.exit(1)

            # we want to vote on individual bits, since if there
            # are multiple wrong bits then voting on the entire
            # time is unlikely to work e.g. if all 3 times have
            # one bit wrong. but we have to reference later
            # minutes back to a common base. so subtract
            # and re-encode to bits.

            m = d.tomin() - i
            d1 = frommin(m)
            # change time fields in place to preserve dst/dut/leap.
            d.year = d1.year
            d.day_of_year = d1.day_of_year
            d.hour = d1.hour
            d.minute = d1.minute
            d.bits = d.encode()

            da.append(d)

            i += 1

        # vote on each bit, weighted by demod()'s "strength".
        bits = [ ]
        total_votes = 0
        total_strengths = 0.0
        for i in range(0, 60):
            if sync[i] == -1:
                strengths = [ 0.0, 0.0, 0.0 ] # sum of strengths of 0s and 1s
                votes = [ 0, 0, 0 ] # number of votes for 0s and 1s
                for ii in range(len(da)):
                    #votes[da[ii].bits[i]] += 1
                    strengths[da[ii].bits[i]] += da[ii].strengths[i]
                    votes[da[ii].bits[i]] += 1
                if strengths[1] > strengths[0]:
                    bits.append(1)
                    total_strengths += strengths[1]
                    total_votes += votes[1]
                else:
                    bits.append(0)
                    total_strengths += strengths[0]
                    total_votes += votes[0]
            else:
                bits.append(0)

        d = self.decode(bits)

        # 2nd value here is total goodness of this decode,
        # which process1() uses to pick the second offset
        # at which this minute starts. it works better
        # to use number of agreeing bits here, not
        # total "strength".

        return [ d, total_votes / float(len(da)) ]

        #return [ d, total_strengths / float(len(da)) ]

    # demodulate one second into 60 0/1 bits.
    # returns [ bits, strengths ]
    # strengths is a lame absolute metric of
    # bit strength, used to weight the voting.
    # XXX should estimate probability of correctness,
    #     from overall amplitude distributions.
    def demod(self, s):
        # process1() smoothed the rectified samples with
        # a window this many seconds wide (e.g. 0.02 seconds).
        global smoothwin

        bits = [ ]
        strengths = [ ]
        for secno in range(0, 60):
            # this bit's samples
            a = s[secno*self.lorate:(secno+1)*self.lorate]

            # measure known high and low amplitudes for
            # comparison; 30..200 ms is always high,
            # and 500..999 is always low (for a 0/1 bit).
            # the only part that varies for 0 vs 1 is
            # 200..500 ms.

            # a[] is smoothed with a window of smoothwin seconds, so
            # avoid areas where high/low might mix. assume smoothwin=0.02.
            high = numpy.mean(a[int(0.050*self.lorate):int(0.180*self.lorate)])
            low = numpy.mean(a[int(0.520*self.lorate):int(0.980*self.lorate)])
            got = numpy.mean(a[int(0.220*self.lorate):int(0.480*self.lorate)])

            gap = float(high - low)
            if got > (low + high) / 2.0:
                bits.append(1)
                str = got - low
                str /= gap
                str = max(str, 0)
                str = min(str, 1)
                strengths.append(str)
            else:
                bits.append(0)
                str = high - got
                str /= gap
                str = max(str, 0)
                str = min(str, 1)
                strengths.append(str)

        return [ bits, strengths ]

    # bits[] has 60 0/1/2 (2 means marker).
    # decode bits into a Decode.
    def decode(self, bits):
        d = Decode()
        d.bits = bits

        # copy b/c we can turn 2's into 1's.
        bits = bits[:]
        for i in range(0, 60):
            if bits[i] > 1:
                bits[i] = 1
            assert bits[i] == 0 or bits[i] == 1

        d.dst1 = bits[2] == 1
        d.leap = bits[3] == 1

        # year, minutes, hours, day-of-year are BCD,
        # least significant bit first.
        d.year = 2000 + 10*d4(bits[51:55]) + d4(bits[4:8])
                
        d.minute = d4(bits[10:14]) + 10*d3(bits[15:18])

        d.hour = d4(bits[20:24]) + 10*d2(bits[25:27])

        d.day_of_year = d4(bits[30:34]) + 10*d4(bits[35:39]) + 100*d2(bits[40:42])

        d.dut = 0.1 * d3(bits[56:59])
        if bits[50] == 0:
            d.dut *= -1

        d.dst2 = bits[55] == 1
        
        return d

    def gofile(self, filename, verbose):
        self.openwav(filename, verbose)
        count = 0 # count samples, to generate "time" for process.
        while True:
            buf = self.readwav()
            if buf.size < 1:
                break
            self.process(buf, False, count / float(self.inrate))
            count += len(buf)
        self.process(numpy.array([]), True, count / float(self.inrate))

    def openwav(self, filename, verbose):
        self.wav = wave.open(filename)
        self.wav_channels = self.wav.getnchannels()
        self.wav_width = self.wav.getsampwidth()
        self.inrate = self.wav.getframerate()
        if verbose:
            sys.stdout.write("file=%s chans=%d width=%d rate=%d\n" % (filename,
                                                                      self.wav_channels,
                                                                      self.wav_width,
                                                                      self.inrate))

    def readwav(self):
        z = self.wav.readframes(self.inrate)
        if self.wav_width == 1:
            zz = numpy.fromstring(z, numpy.int8)
        elif self.wav_width == 2:
            zz = numpy.fromstring(z, numpy.int16)
        else:
            sys.stderr.write("oops wave_width %d" % (self.wav_width))
            sys.exit(1)
        if self.wav_channels == 1:
            return zz
        elif self.wav_channels == 2:
            return zz[0::2] # left
        else:
            sys.stderr.write("oops wav_channels %d" % (self.wav_channels))
            sys.exit(1)

    def opencard(self, desc):
        self.audio = weakaudio.new(desc, None)
        self.inrate = self.audio.rate

    def gocard(self):
        while True:
            [ buf, buf_time ] = self.audio.read()
            if len(buf) > 0:
                # buf_time is UNIX seconds of the last sample of buf[].
                # convert to first sample.
                tm = buf_time - len(buf) / float(self.inrate)
                self.process(buf, False, tm)
            # sleep so that samples accumulate, which makes
            # resample() higher quality.
            time.sleep(0.2)

def onebench(filename, year0, yearday0, hour0, minute0, minutes, verbose):
    d0 = Decode()
    d0.year = year0
    d0.day_of_year = yearday0
    d0.hour = hour0
    d0.minute = minute0

    da = [ ]
    r = WWV()
    r.cb = lambda d : da.append(d)
    r.gofile(filename, False)

    # we're not sure exactly where the first correct
    # timestamp begins. so, for every second offset
    # within a minute, tally up the number of wins
    # and losses. we'll use the best.
    secgood = [ ]
    for off in range(0, 60):
        ngood = 0
        for m in range(0, minutes):
            found = False
            for d in da:
                ss = int(round(d.cardtime))
                ss = ss % 60
                if ss >= off-1 and ss <= off+1:
                    if d.tomin() == d0.tomin() + m:
                        found = True
            if found:
                ngood += 1
        secgood.append(ngood)

    if verbose:
        print "%s score %d of %d" % (filename, numpy.max(secgood), minutes)
    return numpy.max(secgood)

def optimize():
    vars = [
        [ "votewin", [ 1, 3, 5, 10 ] ],
        [ "filterwidth", [ 15, 20, 30 ] ],
        [ "filterorder", [ 2, 3, 4 ] ],
        [ "smoothwin", [ 0.015, 0.02, 0.025, 0.03, 0.05, 0.06, 0.07, 0.08, 0.1 ] ],
        [ "topsecs", [ 1, 2, 4, 8, 16, 25 ] ],
        [ "slicewin", [ 0, 2, 4, 8 ] ],
        #[ "slicesmooth", [ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ] ],
        #[ "slicethresh", [ 0.2, 0.225, 0.25, 0.27, 0.3 ] ],
        ]

    sys.stdout.write("# ")
    for v in vars:
        sys.stdout.write("%s=%s " % (v[0], eval(v[0])))
    sys.stdout.write("\n")

    # warm up any caches, JIT, &c.
    r = WWV()
    r.cb = lambda d : 1
    r.gofile("wwvx1.wav", False)

    for v in vars:
        for val in v[1]:
            old = None
            if "." in v[0]:
                xglob = ""
            else:
                xglob = "global %s ; " % (v[0])
            exec "%sold = %s" % (xglob, v[0])
            exec "%s%s = %s" % (xglob, v[0], val)

            #sys.stdout.write("# ")
            #for vx in vars:
            #    sys.stdout.write("%s=%s " % (vx[0], eval(vx[0])))
            #sys.stdout.write("\n")

            sc = 0
            sc += onebench("wwvx2.wav", 2017, 41, 20, 55, 18, False)
            sc += onebench("wwvx3.wav", 2017, 41, 21, 17, 60, False)
            sc += onebench("wwvx4.wav", 2017, 41, 22, 25, 92, False)
            sc += onebench("wwvx7.wav", 2017, 43, 21, 18, 50, False)
            sc += onebench("wwvx8.wav", 2017, 43, 22, 15, 72, False)
            sc += onebench("wwvx9.wav", 2017, 43, 23, 33, 51, False)
            sc += onebench("wwvx10.wav", 2017, 44, 9, 19, 54, False)

            exec "%s%s = old" % (xglob, v[0])
            sys.stdout.write("%s=%s : " % (v[0], val))
            sys.stdout.write("%d\n" % (sc))
            sys.stdout.flush()

def main():
    if False:
        optimize()
        sys.exit(0)

    if False:
        total = 0
        total += onebench("wwvx2.wav", 2017, 41, 20, 55, 18, True)
        total += onebench("wwvx3.wav", 2017, 41, 21, 17, 60, True)
        total += onebench("wwvx4.wav", 2017, 41, 22, 25, 92, True)
        total += onebench("wwvx7.wav", 2017, 43, 21, 18, 50, True)
        total += onebench("wwvx8.wav", 2017, 43, 22, 15, 72, True)
        total += onebench("wwvx9.wav", 2017, 43, 23, 33, 51, True)
        total += onebench("wwvx10.wav", 2017, 44, 9, 19, 54, True)
        print "%d total" % (total)
        sys.exit(0)

    parser = weakargs.stdparse('Decode WWV.')
    parser.add_argument("-file")
    args = weakargs.parse_args(parser)

    if (args.card == None) == (args.file == None):
        parser.error("one of -card and -file are required")

    if args.file != None:
        r = WWV()
        r.gofile(args.file, True)
        sys.exit(0)

    if args.card != None:
        r = WWV()
        r.opencard(args.card)
        r.gocard()
        sys.exit(0)

    parser.error("one of -card, -file, or -levels is required")
    sys.exit(1)

if __name__ == '__main__':
    if False:
        pfile = "cprof.out"
        sys.stderr.write("jt65: cProfile -> %s\n" % (pfile))
        import cProfile
        import pstats
        cProfile.run('main()', pfile)
        p = pstats.Stats(pfile)
        p.strip_dirs().sort_stats('time')
        # p.print_stats(10)
        p.print_callers()
    else:
        main()
