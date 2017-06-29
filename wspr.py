#!/usr/local/bin/python

#
# decode WSPR
#
# info from wsjt-x manual, wsjt-x source, and
# http://physics.princeton.edu/pulsar/k1jt/WSPR_3.0_User.pdf
# http://www.g4jnt.com/Coding/WSPR_Coding_Process.pdf
#
# uses Phil Karn's Fano convolutional decoder.
#
# Robert Morris, AB1HL
#

import numpy
import wave
import scipy
import scipy.signal
import sys
import os
import math
import time
import copy
import calendar
import subprocess
import threading
import re
import random
from scipy.signal import lfilter
import ctypes
import weakaudio
import weakutil

#
# WSPR tuning parameters.
#
budget = 9 # max seconds of CPU time, per file or two-minute interval (50).
agcwinseconds = 1.0 # AGC window size, seconds (2, 0.5, 0.8).
step_frac = 2.5 # fraction of FFT bin for frequency search (4).
ngoff = 2 # look at this many of guess_offset()'s results (6, 4).
goff_down = 64 # guess_offset down-conversion factor (32, 128).
fano_limit = 40000 # how hard fano will work (10000).
driftmax = 1.5 # look at drifts from -driftmax to +driftmax (2).
ndrift = 3 # number of drifts to try (including drift=0)
coarse_budget = 0.6 # fraction of budget to spend calling guess_offset() (0.25).

# the WSPR sync pattern. each of the 162 2-bit symbols includes one bit of
# sync in the low bit.
pattern = [
  1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1,
  -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1,
  1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1,
  1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1,
  -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1,
  1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1,
]

# Phil Karn's Fano convolutional decoder.
from ctypes import c_ubyte, c_int, byref, cdll
libfano = cdll.LoadLibrary("libfano/libfano.so") # both OSX and FreeBSD

# returns an array of 0/1 bits,
# 2x as many as in_bits.
def fano_encode(in_bits):
    # fano_encode(unsigned char in_bits[], int n_in, unsigned char out_bits[])

    in_array_type = c_ubyte * len(in_bits)
    in_array = in_array_type()
    for i in range(0, len(in_bits)):
        in_array[i] = in_bits[i]

    out_array_type = c_ubyte * (2*len(in_bits))
    out_array = out_array_type()

    n_in = c_int()
    n_in.value = len(in_bits)

    libfano.fano_encode(in_array, n_in, out_array)

    a = []
    for i in range(0, 2*len(in_bits)):
        a.append(out_array[i])

    return a

# in0[i] is log of probability that the i'th symbol is 0.
# returns an array of 0/1 bits, half as many as in0[].
# returns None on error.
def nfano_decode(in0, in1):
    global fano_limit

    in_array_type = c_int * len(in0)
    xin0 = in_array_type()
    xin1 = in_array_type()
    for i in range(0, len(in0)):
        xin0[i] = in0[i]
        xin1[i] = in1[i]

    out_array_type = c_ubyte * (len(in0) / 2)
    out_array = out_array_type()

    n_out = c_int()
    n_out.value = len(in0) / 2

    metric_out_type = c_int * 1
    metric_out = metric_out_type()

    limit = c_int()
    limit = fano_limit

    ok = libfano.nfano_decode(xin0, xin1, n_out, out_array, limit, metric_out)
    if ok != 1:
        return [ None, None ]

    a = []
    for i in range(0, len(in0) / 2):
        a.append(out_array[i])

    metric = metric_out[0]

    return [ a, metric ]

def ntest_fano():
    in_bits = [ 1, 0 ] * 36 # 76 bits, like JT9
    padded = in_bits + ([0] * 31)

    out_bits = fano_encode(padded)

    in0 = [ ]
    in1 = [ ]
    for b in out_bits:
        # center on 128, for signal strength
        if b == 1:
            in1.append(1)
            in0.append(-10)
        else:
            in0.append(1)
            in1.append(-10)

    [ dec, metric ] = nfano_decode(in0, in1)

    assert in_bits == dec[0:len(in_bits)]

if False:
    ntest_fano()
    sys.exit(0)

# Normal function integrated from -Inf to x. Range: 0-1.
# x in units of std dev.
# mean is zero.
def normal(x):
    y = 0.5 + 0.5*math.erf(x / 1.414213)
    return y

# given a distribution and a value, how likely is that value?
def prob(x, mean, std):
    hack = std / 4.0 # bucket size
    if abs(x - mean) > 2 * std:
        # probability of the whole tail above 2*std
        y = 1.0 - normal(2.0)
    else:
        y = normal((x - mean + hack/2) / std) - normal((x - mean - hack/2) / std)
    return y

# how much of the distribution is < x?
def problt(x, mean, std):
    y = normal((x - mean) / std)
    return y

# how much of the distribution is > x?
def probgt(x, mean, std):
    y = 1.0 - normal((x - mean) / std)
    return y

def bit_reverse(x, width):
    y = 0
    for i in range(0, width):
        z = (x >> i) & 1
        y <<= 1
        y |= z
    return y

# turn an array of bits into a number.
# most significant bit first.
def bits2num(bits):
    assert len(bits) < 32
    n = 0
    for i in range(0, len(bits)):
        n *= 2
        n += bits[i]
    return n

# gadget that returns FFT buckets of a fixed set of
# original samples, with required (inter-bucket)
# frequency, drift, and offset.
class Xform:
    def __init__(self, samples, jrate, jblock):
        self.jrate = jrate
        self.jblock = jblock
        self.samples = samples
        self.memo = { }

    # internal.
    def fft2(self, index, quarter):
        key = str(index) + "-" + str(quarter)
        if key in self.memo:
            return self.memo[key]
        
        # caller wants a frequency a bit higher than bin,
        # so shift *down* by the indicated number of quarter bins.
        block = self.samples[index:index+self.jblock]
        if quarter != 0:
            bin_hz = self.jrate / float(self.jblock)
            freq_off = quarter * (bin_hz / 4.0)
            block = weakutil.freq_shift(block, -freq_off, 1.0/self.jrate)
        a = numpy.fft.rfft(block)
        a = abs(a)

        self.memo[key] = a

        return a

    # internal.
    # offset is index into samples[].
    # return four buckets at hz.
    def fft1(self, index, hza):
        bin_hz = self.jrate / float(self.jblock)
        hz = hza[0] + (index / float(len(self.samples))) * (hza[1] - hza[0])

        bin = int(hz / bin_hz)
        binfrac = (hz - (bin * bin_hz)) / bin_hz

        # which of four quarter-bin increments?
        if binfrac < 0.25:
            quarter = 0
        elif binfrac < 0.5:
            quarter = 1
        elif binfrac < 0.75:
            quarter = 2
        else:
            quarter = 3

        a = self.fft2(index, quarter)
        m = a[bin:bin+4]
        return m

    # hza is [ hz0, hzN ] -- at start and end.
    # offset is 0..self.jblock.
    # return buckets[0..162ish][4] -- i.e. a mini-FFT per symbol.
    def get(self, hza, offset):
        # round offset to 1/8th of self.jblock.
        granule = int(self.jblock / 8)
        offset = int(offset / granule) * granule

        bin_hz = self.jrate / float(self.jblock)
        out = [ ]
        for i in range(offset, len(self.samples), self.jblock):
            if i + self.jblock > len(self.samples):
                break
            m = self.fft1(i, hza)
            out.append(m)
        return out

    def len(self):
        return len(self.samples)

class WSPR:
  debug = False

  offset = 0

  def __init__(self):
      self.msgs_lock = threading.Lock()
      self.msgs = [ ]
      self.verbose = False

      self.downhz = 1300 # shift this down to zero hz.
      self.lowhz = 1400 - self.downhz # WSPR signals start here
      self.jrate = 750 # sample rate for processing (FFT &c)
      self.jblock = 512 # samples per symbol

      # set self.start_time to the UNIX time of the start
      # of the last even UTC minute.
      now = int(time.time())
      gm = time.gmtime(now)
      self.start_time = now - gm.tm_sec
      if (gm.tm_min % 2) == 1:
          self.start_time -= 60

  def close(self):
      pass

  # return the minute number for t, a UNIX time in seconds.
  # truncates down, so best to pass a time mid-way through a minute.
  # returns only even minutes.
  def minute(self, t):
      dt = t - self.start_time
      mins = 2 * int(dt / 120.0)
      return mins

  # seconds since minute(), 0..119
  def second(self, t):
      dt = t - self.start_time
      m = 120 * int(dt / 120.0)
      return dt - m

  # printable UTC timestamp, e.g. "07/07/15 16:31:00"
  # dd/mm/yy hh:mm:ss
  # t is unix time.
  def ts(self, t):
      gm = time.gmtime(t)
      return "%02d/%02d/%02d %02d:%02d:%02d" % (gm.tm_mday,
                                                gm.tm_mon,
                                                gm.tm_year - 2000,
                                                gm.tm_hour,
                                                gm.tm_min,
                                                gm.tm_sec)

  def openwav(self, filename):
    self.wav = wave.open(filename)
    self.wav_channels = self.wav.getnchannels()
    self.wav_width = self.wav.getsampwidth()
    self.cardrate = self.wav.getframerate()

  def readwav(self, chan):
    z = self.wav.readframes(8192)
    if self.wav_width == 1:
      zz = numpy.fromstring(z, numpy.int8)
    elif self.wav_width == 2:
      if (len(z) % 2) == 1:
        return numpy.array([])
      zz = numpy.fromstring(z, numpy.int16)
    else:
      sys.stderr.write("oops wave_width %d" % (self.wav_width))
      sys.exit(1)
    if self.wav_channels == 1:
      return zz
    elif self.wav_channels == 2:
      return zz[chan::2] # chan 0/1 => left/right
    else:
      sys.stderr.write("oops wav_channels %d" % (self.wav_channels))
      sys.exit(1)

  def gowav(self, filename, chan):
    self.openwav(filename)
    bufbuf = [ ]
    while True:
      buf = self.readwav(chan)
      if buf.size < 1:
        break
      bufbuf.append(buf)
    samples = numpy.concatenate(bufbuf)
    self.process(samples, 0)

  def opencard(self, desc):
      self.cardrate = 12000
      self.audio = weakaudio.new(desc, self.cardrate)

  def gocard(self):
      bufbuf = [ ]
      nsamples = 0
      while True:
          [ buf, buf_time ] = self.audio.read()

          bufbuf.append(buf)
          nsamples += len(buf)
          samples_time = buf_time

          if len(buf) > 0:
              mx = numpy.max(numpy.abs(buf))
              if mx > 30000:
                  sys.stderr.write("!")

          if len(buf) == 0:
              time.sleep(0.2)

          # a WSPR frame starts on second 1, and takes 110.5 seconds, so
          # should end with the 112th second.
          # wait until we have enough samples through 113th second of minute.
          # 162 symbols, 0.682 sec/symbol, 110.5 seconds total.
          sec = self.second(samples_time)
          if sec >= 113 and nsamples >= 113*self.cardrate:
              # we have >= 113 seconds of samples, and second of minute is >= 113.

              samples = numpy.concatenate(bufbuf)

              # sample # of one second before start of two-minute interval.
              i0 = len(samples) - self.cardrate * self.second(samples_time)
              i0 -= self.cardrate # process() expects samples starting at 0:59
              i0 = int(i0)
              i0 = max(i0, 0)
              t = samples_time - (len(samples)-i0) * (1.0/self.cardrate)

              if False:
                  print("%s got %d samples, writing to x.wav" % (self.ts(time.time()), len(samples[i0:])))
                  writewav1(samples[i0:], "x.wav", self.cardrate)

              self.process(samples[i0:], t)

              bufbuf = [ ]
              nsamples = 0

  # received a message, add it to the list.
  # offset in seconds.
  # drift in hz/minute.
  def got_msg(self, minute, hz, txt, snr, offset, drift):
      if self.verbose:
          print("%6.1f %.1f %.1f %d %s" % (hz, offset, drift, snr, txt))
      now = time.time()
      item = [ minute, hz, txt, now, snr, offset, drift ]
      self.msgs_lock.acquire()
      self.msgs.append(item)
      self.msgs_lock.release()

  # someone wants a list of all messages received.
  # each msg is [ minute, hz, msg, decode_time, snr, offset, drift ]
  def get_msgs(self):
      self.msgs_lock.acquire()
      a = copy.copy(self.msgs)
      self.msgs_lock.release()
      return a

  def process(self, samples, samples_time):
    global budget, agcwinseconds, step_frac, ngoff, goff_down, fano_limit, driftmax, ndrift, coarse_budget

    # samples_time is UNIX time that samples[0] was
    # sampled by the sound card.
    samples_minute = self.minute(samples_time + 60)

    t0 = time.time()

    # trim trailing zeroes that wsjt-x adds
    i = len(samples)
    while i > 1000 and numpy.max(samples[i-1:]) == 0.0:
        if numpy.max(samples[i-1000:]) == 0.0:
            i -= 1000
        elif numpy.max(samples[i-100:]) == 0.0:
            i -= 100
        elif numpy.max(samples[i-10:]) == 0.0:
            i -= 10
        else:
            i -= 1
    samples = samples[0:i]

    # bandpass filter around 1400..1600.
    # down-convert by 1200 Hz (i.e. 1400-1600 -> 200->400),
    # and reduce sampling rate to 1500.
    assert self.cardrate == 12000 and self.jrate == 750
    filter = weakutil.butter_bandpass(1380, 1620, self.cardrate, 3)
    samples = lfilter(filter[0], filter[1], samples)
    # down-convert from 1400 to 100.
    samples = weakutil.freq_shift(samples, -self.downhz, 1.0/self.cardrate)
    # down-sample.
    samples = samples[0::16]

    agcwinlen = int(agcwinseconds * self.jrate)

    #
    # pad at start+end b/c transmission might have started early or late.
    # I've observed dt's from -2.8 to +3.4.
    # there's already two seconds of slop at start b/c xmission starts
    # at xx:01 but file seems to start at xx:59.
    # and we're going to trim a second either side after AGC.
    # so we want to add 2 seconds at start, and up to 5 at end.
    #
    startslop = 1 * self.jrate + agcwinlen  # add this much at start (1)
    endslop = 4 * self.jrate + agcwinlen    # add this much at end (4)
    sm = numpy.mean(samples) # pad with plausible signal levels
    sd = numpy.std(samples)
    samples = numpy.append(numpy.random.normal(sm, sd, startslop), samples)
    samples = numpy.append(samples, numpy.random.normal(sm, sd, endslop))

    if agcwinlen > 0.001:
        #
        # apply our own AGC, now that the band-pass filter has possibly
        # eliminated strong nearby JT65 that might have pumped receiver AGC.
        # perhaps this helps guess_offset().
        #
        # agcwin = numpy.ones(agcwinlen)
        # agcwin = numpy.hamming(agcwinlen)
        agcwin = scipy.signal.tukey(agcwinlen)
        agcwin = agcwin / numpy.sum(agcwin)
        mavg = numpy.convolve(abs(samples), agcwin)[agcwinlen/2:]
        mavg = mavg[0:len(samples)]
        samples = numpy.divide(samples, mavg)
        samples = numpy.multiply(samples, 1000.0)

        # drop first and last seconds, since probably wrecked by AGC.
        # they were slop anyway.
        samples = samples[agcwinlen:-agcwinlen]

    # we added 1 second to the start and a bunch to the end.
    # wsjt-x already added 2 seconds to the start.
    # trim so that we have 3 seconds at the start
    # and 4 seconds at the end.
    msglen = 162 * self.jblock # samples in a complete transmission
    samples = samples[0:msglen + 7 * self.jrate]

    bin_hz = self.jrate / float(self.jblock)

    # WSPR signals officially lie between 1400 and 1600 Hz.
    # we've down-converted to 100 - 300 Hz.
    # search a bit wider than that.
    min_hz = self.lowhz-20
    max_hz = self.lowhz+200+20

    # generate a few copies of samples corrected for various amounts of drift.
    # drift_samples[i] = [ [ drift_hz, samples ] ]
    drift_samples = [ ]
    if ndrift == 1:
        driftstart = 0.0
        driftend = 0.1
        driftinc = 1
    else:
        driftstart = -driftmax
        driftend = driftmax+0.001
        driftinc = 2.0*driftmax / (ndrift - 1)
    for drift in numpy.arange(driftstart, driftend, driftinc):
        if drift == 0:
            drift_samples.append( [ 0, samples ] )
        else:
            z = weakutil.freq_shift_ramp(samples, [drift,-drift], 1.0/self.jrate)
            drift_samples.append( [ drift, z ] )

    # sum FFTs over the whole two minutes to find likely frequencies.
    # coarse_rank[i] is the sum of the four tones starting at bin i,
    # so that the ranks refer to a signal whose base tone is in bin i.
    coarse_rank = [ ]
    for di in range(0, len(drift_samples)):
        coarse = numpy.zeros(self.jblock / 2 + 1)
        coarseblocks = 0
        for i in range(2*self.jrate, len(samples)-2*self.jrate, self.jblock/2):
            block = drift_samples[di][1][i:i+self.jblock]
            a = numpy.fft.rfft(block)
            a = abs(a)
            coarse = numpy.add(coarse, a)
            coarseblocks = coarseblocks + 1
        coarse = coarse / coarseblocks # sum -> average, for noise calculation
        xrank = [ [ i * bin_hz,
                    (coarse[i+0]+coarse[i+1]+coarse[i+2]+coarse[i+3]) / (coarse[i-2]+coarse[i-1]+coarse[i+4]+coarse[i+5]),
                    di ] for i in range(2, len(coarse)-6) ]
        coarse_rank += xrank
    
    # sort coarse bins, biggest signal first.
    # coarse_rank[i] = [ hz, strength, drift_index ]
    coarse_rank = [ e for e in coarse_rank if (e[0] >= min_hz and e[0] < max_hz) ]
    coarse_rank = sorted(coarse_rank, key = lambda e : -e[1])

    # calculate noise for snr, mimicing wsjtx wsprd.c.
    # first average in freq domain over 7-bin window.
    # then noise from 30th percentile.
    nn = numpy.convolve(coarse, [ 1, 1, 1, 1, 1, 1, 1 ])
    nn = nn / 7.0
    nn = nn[6:]
    nns = sorted(nn[int(min_hz/bin_hz):int(max_hz/bin_hz)])
    noise = nns[int(0.3*len(nns))]

    # avoid checking a given step_hz more than once.
    # already is indexed by int(hz / step_hz)
    step_hz = bin_hz / step_frac
    already = { }

    # for each WSJT FFT bin and offset and drift, the strength of
    # the sync correlation.
    # fine_rank[i] = [ drift_index, hz, offset, strength ]
    fine_rank = [ ]

    for ce in coarse_rank:
        if time.time() - t0 >= budget * coarse_budget:
            break

        # center of bin
        hz0 = ce[0]

        # hz0 is the lowest tone of a suspected signal.
        # but, due to FFT granularity, actual signal
        # may be half a tone lower or higher.
        start_hz = (hz0 - bin_hz/2.0) + (step_hz / 2.0)
        end_hz = hz0 + bin_hz/2.0 - step_hz/100.0

        #print "%.1f %.1f..%.1f" % (hz0, start_hz, end_hz)
        for hz in numpy.arange(start_hz, end_hz, step_hz):
            di = ce[2] # drift_samples[di] = [ drift_hz, shifted samples ]
            hzkey = str(int(hz / step_hz)) + " " + str(di)
            # hzkey = int(hz / step_hz)
            if hzkey in already:
                break
            already[hzkey] = True
            offsets = self.guess_offset(drift_samples[di][1], hz)
            # offsets[i] is [ offset, strength ]
            offsets = offsets[0:ngoff]
            triples = [ [ di, hz, offset, strength ] for [ offset, strength ] in offsets ]
            fine_rank += triples

    #print "%d in fine_rank, spent %.1f seconds" % (len(fine_rank), time.time() - t0)

    # call Fano on the bins with the higest sync correlation first,
    # since there's not enough time to look at all bins.
    fine_rank = sorted(fine_rank, key=lambda r : -r[3])

    # store each message just once, to suppress duplicates.
    # indexed by message text; value is [ samples_minute, hz, msg, snr, offset, drift ]
    msgs = { }

    # set up to cache FFTs at various offsets and drifts.
    xf = Xform(samples, self.jrate, self.jblock)

    for rr in fine_rank:
        # rr = [ drift_index, hz, offset, strength ]
        if time.time() - t0 >= budget:
            break
        hz = rr[1]
        offset = rr[2]
        drift = drift_samples[rr[0]][0]
        if offset < 0:
            continue

        if True:
            hza = [ hz - drift, hz + drift ]
        else:
            xf = Xform(drift_samples[rr[0]][1], self.jrate, self.jblock)
            hza = [ hz, hz ]
        ss = xf.get(hza, offset)
    
        # ss has one element per symbol time.
        # ss[i] is a 4-element FFT.
    
        # first symbol is in ss[0]
        # return is [ hza, msg, snr ]
        x = self.process1(samples_minute, ss[0:162], hza, noise)

        if x != None:
            # info is [ minute, hz, msg, snr, offset, drift ]
            info = [ samples_minute, numpy.mean(hza), x[1], x[2], offset, drift ]
            #print "found at %.1f %s %d %s" % (hz, hza, offset, x[1])
            if not (x[1] in msgs):
                msgs[x[1]] = info
            elif x[2] > msgs[x[1]][3]:
                # we have a higher SNR.
                msgs[x[1]] = info

        sys.stdout.flush()

    for txt in msgs:
        info = msgs[txt]
        hz = info[1] + self.downhz
        txt = info[2]
        snr = info[3]
        offset = (info[4] / float(self.jrate)) - 2.0 # convert to seconds
        drift = info[5] # hz / minute
        self.got_msg(info[0], hz, txt, snr, offset, drift)

  def hz0(self, hza, sym):
      hz = hza[0] + (hza[1] - hza[0]) * (sym / float(len(pattern)))
      return hz

  # since there have been so many bugs in guess_offset().
  def test_guess_offset(self):
      mo = 0
      bin_hz = self.jrate / float(self.jblock)
      n = 0
      sumabs = 0.0
      sum = 0.0
      for iters in range(0, 250):
          hz = 80 + random.random() * 240
          nstart = int(random.random() * 3000)
          nend = 1000 + int(random.random() * 3000)
          symbols = [ ]
          for p in pattern:
              sym = 2 * random.randint(0, 1)
              if p > 0:
                  sym += 1
              symbols.append(sym)
          samples = numpy.random.normal(0, 0.5, nstart)
          samples = numpy.append(samples, weakutil.fsk(symbols, [ hz, hz ], bin_hz, self.jrate, self.jblock))
          samples = numpy.append(samples, numpy.random.normal(0, 0.5, nend))
          samples = samples * 1000
          xa = self.guess_offset(samples, hz, nstart)
          x0start = xa[0][0]
          x0abs = abs(nstart - x0start)
          #print("%.1f %d: %d %d" % (hz, nstart, x0start, x0abs))
          if abs(xa[1][0] - nstart) < x0abs:
              print("%.1f %d: %d %d -- %d" % (hz, nstart, x0start, x0abs, xa[1][0]))
          mo = max(mo, x0abs)
          sumabs += x0abs
          sum += x0start - nstart
          n += 1
      # jul 18 2016 -- max diff was 53.
      # jun 16 2017 -- max diff was 77 (but with different hz range).
      # jun 16 2017 -- max diff 52, avg abs diff 17, avg diff -1
      # jun 16 2017 -- max diff 33, avg abs diff 16, avg diff 0 (using tone, not bandpass filter)
      print("max diff %d, avg abs diff %d, avg diff %d" % (mo, sumabs/n, sum/n))

  # returns an array of [ offset, strength ], sorted
  # by strength, most-plausible first.
  def guess_offset(self, samples, hz):
      global goff_down
      bin_hz = self.jrate / float(self.jblock)

      ntaps = self.jblock

      # average y down to a much lower rate to make the
      # correlate() go faster. 64 works well.
      downfactor = goff_down

      # filter each of the four tones
      tones = [ ]
      bigtones = [ ]
      for tone in range(0, 4):
          thz = hz + tone*bin_hz
          taps = weakutil.costone(self.jrate, thz, ntaps)
          # yx = lfilter(taps, 1.0, samples)
          # yx = numpy.convolve(samples, taps, mode='valid')
          # yx = scipy.signal.convolve(samples, taps, mode='valid')
          yx = scipy.signal.fftconvolve(samples, taps, mode='valid')
          # hack to match size of lfilter() output
          yx = numpy.append(numpy.zeros(ntaps-1), yx)
          yx = abs(yx)

          # scipy.signal.resample(yx, len(yx) / downfactor) works, but too slow.
          # re = weakutil.Resampler(downfactor*64, 64)
          # yx = re.resample(yx)
          yx = weakutil.moving_average(yx, downfactor)
          bigtones.append(yx)
          yx = yx[0::downfactor]

          tones.append(yx)

      # we just care about sync vs no sync,
      # so combine tones 0 and 2, and 1 and 3
      #tones[0] = numpy.add(tones[0], tones[2])
      #tones[1] = numpy.add(tones[1], tones[3])
      tones[0] = numpy.maximum(tones[0], tones[2])
      tones[1] = numpy.maximum(tones[1], tones[3])

      # now tone1 - tone0.
      # yields +/- that should match pattern.
      tt = numpy.subtract(tones[1], tones[0])

      z = numpy.repeat(pattern, int(self.jblock / downfactor))
      cc = numpy.correlate(tt, z)

      indices = list(range(0, len(cc)))
      indices = sorted(indices, key=lambda i : -cc[i])
      offsets = numpy.multiply(indices, downfactor)
      offsets = numpy.subtract(offsets, ntaps / 2)

      both = [ [ offsets[i], cc[indices[i]] ] for i in range(0, len(offsets)) ]

      return both

  # returns None or [ hz, start, nerrs, msg, twelve ]
  def process1(self, samples_minute, m, hza, noise):
    if len(m) < 162:
        print "process1: too short %d < 162" % (len(m))
        return None

    # for each symbol time, figure out the levels of the
    # 0 and 1 signals. here's the place we'll get rid of sync.
    # levels[i][j] is the i'th symbol level for bit j=0 or 1.
    levels = [ ]
    for pi1 in range(0, len(pattern)):
        if True:
            # factor out the sync.
            # this works the best.
            if pattern[pi1] > 0:
                sync = 1
            else:
                sync = 0
            sig0 = m[pi1][sync]
            sig1 = m[pi1][2+sync]
        if False:
            # ignore sync, look at sum of bins. works badly.
            sig0 = m[pi1][0] + m[pi1][1]
            sig1 = m[pi1][2] + m[pi1][3]
        if False:
            # ignore sync. works badly.
            sig0 = max(m[pi1][0], m[pi1][1])
            sig1 = max(m[pi1][2], m[pi1][3])
        levels.append([ sig0, sig1 ])

    # estimate distributions of bin strengths for 
    # winning and losing FSK bins, ignoring sync bins.
    # this is not perfect, since we don't really know
    # winning vs losing, and the distributions don't
    # seem to be normal.
    winners = [ ]
    losers = [ ]
    for pi1 in range(0, len(pattern)):
        sig0 = levels[pi1][0]
        sig1 = levels[pi1][1]
        winners.append(max(sig0, sig1))
        losers.append(min(sig0, sig1))
    winmean = numpy.mean(winners)
    winstd = numpy.std(winners)
    losemean = numpy.mean(losers)
    losestd = numpy.std(losers)

    # power rather than voltage.
    rawsnr = (winmean*winmean) / (noise*noise)
    # the "-1" turns (s+n)/n into s/n
    rawsnr -= 1
    if rawsnr < 0.1:
        rawsnr = 0.1
    rawsnr /= (2500.0 / 1.5) # 1.5 hz noise b/w -> 2500 hz b/w
    snr = 10 * math.log10(rawsnr)

    if False and snr < -30:
        # decodes for "signals" this weak are usually incorrect.
        return None

    # for each time slot, decide probability it's a 0,
    # and probability it's a 1.
    # we know the effect
    # of sync, so this is really 2-FSK (despite the 4 tones).
    # we'll pass Fano the two probabilities.
    softsyms = [ ]
    for pi in range(0, len(pattern)):
        v0 = levels[pi][0]
        v1 = levels[pi][1]

        if True:
          # we have two separate sources of evidence -- v0 and v1.
          # figure out what each implies about 0 vs 1.
          # then combine with Bayes' rule.

          # if a 0 were sent, how likely is v0? (it's the "0" FSK bin)
          p00 = problt(v0, winmean, winstd)
          # if a 1 were sent, how likely is v0?
          p01 = probgt(v0, losemean, losestd)

          # if a 0 were sent, how likely is v1? (it's the "1" FSK bin)
          p10 = probgt(v1, losemean, losestd)
          # if a 1 were sent, how likely is v1?
          p11 = problt(v1, winmean, winstd)

          # Bayes' rule, for P(0) given v0 and v1
          a = 0.5 * p00 * p10
          b = 0.5*p00*p10 + 0.5*p01*p11

          if b == 0:
              p0 = 0.5
              p1 = 0.5
          else:
              p0 = a / b
              p1 = 1 - p0

        if False:
            # from wspr-analyze.dat analysis of ratio of signal
            # levels vs probability of error.
            # map from 10*(stronger/weaker) to probability of error,
            # generated by analyze1.py from wspr-analyze.dat.
            m = [
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.487, 0.442, 0.358, 0.315, 0.258,
0.219, 0.201, 0.169, 0.127, 0.127, 0.098, 0.098, 0.091, 0.085, 0.060,
0.074, 0.066, 0.067, 0.047, 0.048, 0.039, 0.038, 0.046, 0.034, 0.031,
0.029, 0.033, 0.038, 0.029, 0.032, 0.029, 0.030, 0.028, 0.030, 0.013,
0.038, 0.026, 0.030, 0.023, 0.012, 0.022, 0.020, 0.011, 0.029, 0.010,
0.015, 0.029, 0.018, 0.021, 0.015, 0.014, 0.005, 0.016, 0.015, 0.011,
0.025, 0.006, 0.018, 0.014, 0.019, 0.010, 0.014, 0.017, 0.009, 0.014,
0.011, 0.004,
                ]
            if v0 > v1:
                strength = int(10.0 * (v0 / v1))
                if strength >= len(m):
                    p0 = 0.99
                    p1 = 0.01
                else:
                    p1 = m[strength]
                    p0 = 1.0 - p1
            else:
                strength = int(10.0 * (v1 / v0))
                if strength >= len(m):
                    p0 = 0.01
                    p1 = 0.99
                else:
                    p0 = m[strength]
                    p1 = 1.0 - p0

        if False:
            # from wspr-analyze.dat analysis of ratio of signal
            # levels vs probability of error.
            # map from strength to probability of error,
            # generated by analyze1.py from wspr-analyze.dat.
            m = [
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.479, 0.454, 0.363, 0.358, 0.339,
0.253, 0.227, 0.180, 0.212, 0.191, 0.148, 0.128, 0.126, 0.098, 0.112,
0.098, 0.089, 0.109, 0.089, 0.094, 0.060, 0.072, 0.079, 0.062, 0.071,
0.032, 0.075, 0.053, 0.040, 0.045, 0.043, 0.073, 0.053, 0.071, 0.054,
0.045, 0.058, 0.058, 0.062, 0.050, 0.032, 0.049, 0.032, 0.024, 0.041,
0.028, 0.021, 0.020, 0.028, 0.015, 0.014, 0.036, 0.045, 0.044, 0.018,
0.000, 0.031, 0.039, 0.015, 0.051, 0.040, 0.018, 0.037,
                ]
            if v0 > v1:
                if v1 > noise:
                    strength = (v0 - noise) / (v1 - noise)
                else:
                    strength = 10
                strength = int(10.0 * strength)
                if strength >= len(m):
                    p0 = 0.99
                    p1 = 0.01
                else:
                    p1 = m[strength]
                    p0 = 1.0 - p1
            else:
                if v0 > noise:
                    strength = (v1 - noise) / (v0 - noise)
                else:
                    strength = 10
                strength = int(10.0 * strength)
                if strength >= len(m):
                    p0 = 0.01
                    p1 = 0.99
                else:
                    p0 = m[strength]
                    p1 = 1.0 - p0

        if False:
            # map from s/n to p(wrong), calculated from training set
            # of signals (see below, wspr-analyze.dat).
            m = [ 
0.880, 0.865, 0.825, 0.737, 0.599, 0.452, 0.331, 0.250, 0.204, 0.171,
0.140, 0.141, 0.118, 0.119, 0.100, 0.102, 0.096, 0.114, 0.093, 0.102,
0.070, 0.106, 0.065, 0.096, 0.058, 0.066, 0.060, 0.032, 0.046, 0.049,
0.046, 0.034, 0.016, 0.056, 0.061, 0.014, 0.033, 0.032, 0.045, 0.023,
0.021, 0.009, 0.006, 0.011, 0.028, 0.019, 0.024, 0.031, 0.026, 0.017,
0.006, 0.031, 0.011, 0.012, 0.000,
                ]
            p0 = 1.0 - m[min(int(2.0 * v0 / noise), len(m)-1)]
            p1 = 1.0 - m[min(int(2.0 * v1 / noise), len(m)-1)]

        if False:
            # for each snr*20, cumulative probability that if a
            # signal was sent, we receive snr <= this.
            # from ./analyze2.py < wspr-analyze.dat
            yes = [
0.000076, 0.000515, 0.001272, 0.002225, 0.003875, 0.005722, 0.008144,
0.010793, 0.013623, 0.016863, 0.020087, 0.023780, 0.028079, 0.032499,
0.037358, 0.042202, 0.046849, 0.051799, 0.057566, 0.063424, 0.069631,
0.075443, 0.081180, 0.087538, 0.094576, 0.101343, 0.108260, 0.115314,
0.122550, 0.129664, 0.136521, 0.143923, 0.151825, 0.159847, 0.167779,
0.175635, 0.183719, 0.191650, 0.199870, 0.207363, 0.215643, 0.223605,
0.231991, 0.239695, 0.247915, 0.256482, 0.264672, 0.272634, 0.280596,
0.288285, 0.296384, 0.304391, 0.312520, 0.320679, 0.328777, 0.336119,
0.343611, 0.350529, 0.358219, 0.366226, 0.373613, 0.380985, 0.388084,
0.394775, 0.401314, 0.408867, 0.415679, 0.422521, 0.429045, 0.436205,
0.443168, 0.449753, 0.456216, 0.462543, 0.468901, 0.475380, 0.481480,
0.487898, 0.493983, 0.500068, 0.506002, 0.511996, 0.517324, 0.522683,
0.528677, 0.534248, 0.539546, 0.544995, 0.550399, 0.555682, 0.561025,
0.565824, 0.570561, 0.575769, 0.580370, 0.585017, 0.589831, 0.594145,
0.598595, 0.603636, 0.607905, 0.612355, 0.616941, 0.620938, 0.624904,
0.628869, 0.633062, 0.636847, 0.640707, 0.644551, 0.648835, 0.652286,
0.655995, 0.659522, 0.662928, 0.666470, 0.669739, 0.672979, 0.676491,
0.680305, 0.683938, 0.687980, 0.691461, 0.694504, 0.697531, 0.700937,
0.704161, 0.707446, 0.710640, 0.714015, 0.717134, 0.720010, 0.722825,
0.725641, 0.728653, 0.731605, 0.734496, 0.737190, 0.739764, 0.742382,
0.745334, 0.748286, 0.750950, 0.753311, 0.755627, 0.758170, 0.760774,
0.763181, 0.765663, 0.767888, 0.770492, 0.772672, 0.774882, 0.777243,
0.779453, 0.781602, 0.783828, 0.786022, 0.788414, 0.790760, 0.792834,
0.794408, 0.796512, 0.798828, 0.800705, 0.802552, 0.804444, 0.806382,
0.808062, 0.809757, 0.811771, 0.813390, 0.815434, 0.817114, 0.818643,
0.820293, 0.821928, 0.823699, 0.825152, 0.826998, 0.828724, 0.830208,
0.831812, 0.833174, 0.834552, 0.835914, 0.837458, 0.838699, 0.840259,
0.841606, 0.843362, 0.844800, 0.845935, 0.847206, 0.848357, 0.849689,
0.850915, 0.852323, 0.853337, 0.854790, 0.856016, 0.857182, 0.858362,
0.859543, 0.860845, 0.861844, 0.862979, 0.863993, 0.865129, 0.866219,
0.867430, 0.868519, 0.869776, 0.870760, 0.872046, 0.872939, 0.874044,
0.875467, 0.876557, 0.877647, 0.878798, 0.879570, 0.880569, 0.881613,
0.882703, 0.883732, 0.884595, 0.885579, 0.886517, 0.887411, 0.888349,
0.889076, 0.890044, 0.891013, 0.891952, 0.892890, 0.893753, 0.894343,
0.895130, 0.895842, 0.896720, 0.897567, 0.898203, 0.898945, 0.899747,
0.900671, 0.901609, 0.902517, 0.903153, 0.903864, 0.904636, 0.905484,
0.906241, 0.907074, 0.907649, 0.908406, 0.909299, 0.910237, 0.911130,
0.911978, 0.912447, 0.913204, 0.913779, 0.914400, 0.915202, 0.915989,
                ]
            # for each snr*20, cumulative probability that if this
            # signal wasn't sent, we receive snr <= this.
            no = [
0.000431, 0.003388, 0.008686, 0.016568, 0.026518, 0.039425, 0.054802,
0.071241, 0.089776, 0.110135, 0.131900, 0.155719, 0.180227, 0.205539,
0.232287, 0.258029, 0.285854, 0.312257, 0.338861, 0.366039, 0.391753,
0.417682, 0.443253, 0.468450, 0.492642, 0.516001, 0.539274, 0.561787,
0.583380, 0.604370, 0.623781, 0.642259, 0.661096, 0.677679, 0.693572,
0.709552, 0.724153, 0.738195, 0.750528, 0.761970, 0.773183, 0.783377,
0.793844, 0.802874, 0.811905, 0.820146, 0.827741, 0.835638, 0.842012,
0.848775, 0.854647, 0.860433, 0.866104, 0.870899, 0.876097, 0.880662,
0.884783, 0.888458, 0.892550, 0.895766, 0.899126, 0.902787, 0.905716,
0.908458, 0.911286, 0.914072, 0.916469, 0.919025, 0.921480, 0.923849,
0.925916, 0.927898, 0.929764, 0.931559, 0.933282, 0.934890, 0.936613,
0.938307, 0.939786, 0.941350, 0.942901, 0.944222, 0.945500, 0.946361,
0.947811, 0.949046, 0.950209, 0.951200, 0.952276, 0.953339, 0.954258,
0.955435, 0.956426, 0.957301, 0.958034, 0.958795, 0.959469, 0.960144,
0.960934, 0.961652, 0.962341, 0.963016, 0.963532, 0.964265, 0.964911,
0.965471, 0.966318, 0.966878, 0.967509, 0.968299, 0.968816, 0.969376,
0.970008, 0.970438, 0.970812, 0.971458, 0.971759, 0.972219, 0.972506,
0.973023, 0.973439, 0.973798, 0.974272, 0.974659, 0.975133, 0.975794,
0.976181, 0.976569, 0.976957, 0.977402, 0.977689, 0.978119, 0.978608,
0.978895, 0.979311, 0.979541, 0.979828, 0.980173, 0.980417, 0.980747,
0.980991, 0.981307, 0.981537, 0.981867, 0.982154, 0.982412, 0.982628,
0.982958, 0.983173, 0.983389, 0.983690, 0.983848, 0.984135, 0.984480,
0.984767, 0.984954, 0.985097, 0.985356, 0.985542, 0.985743, 0.985944,
0.986145, 0.986389, 0.986576, 0.986820, 0.987064, 0.987337, 0.987653,
0.987825, 0.988083, 0.988299, 0.988586, 0.988686, 0.988902, 0.989060,
0.989232, 0.989404, 0.989620, 0.989792, 0.989921, 0.990151, 0.990294,
0.990424, 0.990524, 0.990653,
                ]
            v0x = int(round(20.0 * v0 / noise))
            v1x = int(round(20.0 * v1 / noise))

            # if a 0 were sent, how likely is v0? (it's the "0" FSK bin)
            p00 = yes[min(v0x, len(yes)-1)]
            # if a 1 were sent, how likely is v0?
            p01 = 1.0 - no[min(v0x, len(no)-1)]

            # if a 0 were sent, how likely is v1? (it's the "1" FSK bin)
            p10 = 1.0 - no[min(v1x, len(no)-1)]
            # if a 1 were sent, how likely is v1?
            p11 = yes[min(v1x, len(yes)-1)]

            # Bayes' rule, for P(0) given v0 and v1
            a = 0.5 * p00 * p10
            b = 0.5*p00*p10 + 0.5*p01*p11

            if b == 0:
                p0 = 0.5
                p1 = 0.5
            else:
                p0 = a / b
                p1 = 1 - p0

        assert p0 >= 0 and p1 >= 0

        # mimic Karn's metrics.c.
        if p0 > 0:
            logp0 = math.log(2*p0, 2) - 0.5
            logp0 = math.floor(logp0 * 4 + 0.5)
            logp0 = int(logp0)
        else:
            logp0 = -100
        if p1 > 0:
            logp1 = math.log(2*p1, 2) - 0.5
            logp1 = math.floor(logp1 * 4 + 0.5)
            logp1 = int(logp1)
        else:
            logp1 = -100

        softsyms.append( [ logp0, logp1, v0, v1 ] )

    sys.exit(1)

    # un-interleave softsyms[], by bit-reversal of index.
    p = 0
    ss1 = [None]*162
    for i in range(0, 256):
        j = bit_reverse(i, 8)
        if j < 162:
            ss1[p] = softsyms[j]
            p += 1
    softsyms = ss1

    sym0 = [ ]
    sym1 = [ ]
    for e in softsyms:
        sym0.append(e[0])
        sym1.append(e[1])

    [ dec, metric ] = nfano_decode(sym0, sym1)

    if dec == None:
        # Fano could not decode
        return None

    if numpy.array_equal(dec[0:80], [0]*80):
        # all bits are zero
        return None


    dec = dec[0:-31] # drop the 31 bits of padding, yielding 50 bits

    msg = self.unpack(dec)
    if msg == None:
        return None

    if False:
        # analyze what weak and strong symbols look like.
        # i.e. prepare a map from stronger/weaker to probability
        # that it's really stronger.
        re_enc = fano_encode(dec + ([0] * 31))
        
        # re_enc[i] is the correct (originally transmitted) symbol
        # softsyms[i][2] is the received FSK 0 signal level
        # softsyms[i][3] is the received FSK 1 signal level

        f = open("wspr-analyze.dat", "a")
        for i in range(0, len(re_enc)):
            v0 = softsyms[i][2]
            v1 = softsyms[i][3]

            # strength = max(v0, v1) / min(v0, v1), ok)

            #if min(v0, v1) > noise:
            #    strength = (max(v0, v1) - noise) / (min(v0, v1) - noise)
            #else:
            #    strength = 10

            #if (re_enc[i] == 0) == (v0 > v1):
            #    ok = 1
            #else:
            #    ok = 0
            #f.write("%f %s\n" % (strength, ok))

            f.write("%f %s\n" % (v0 / noise, int(re_enc[i] == 0)))
            f.write("%f %s\n" % (v1 / noise, int(re_enc[i] == 1)))
        f.close()

    return [ hza, msg, snr ]

  # convert packed character to Python string.
  # 0..9 a..z space
  def charn(self, c):
    if c >= 0 and c <= 9:
      return chr(ord('0') + c)
    if c >= 10 and c < 36:
      return chr(ord('A') + c - 10)
    if c == 36:
      return ' '
    sys.stderr.write("wspr charn(%d) bad\n" % (c))
    return '?'

  # bits[] has 50 bits.
  # 28 for callsign.
  # 15 for grid locator.
  # 7 for power level.
  # http://physics.princeton.edu/pulsar/k1jt/WSPR_3.0_User.pdf
  # http://www.g4jnt.com/Coding/WSPR_Coding_Process.pdf
  # details from K1JT/K9AN's unpk_() in wsprd_utils.c.
  def unpack(self, bits):
      assert len(bits) == 50

      # 28-bit call sign
      n = bits2num(bits[0:28])
      num = "0123456789"
      alnum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
      al = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
      call = ""
      call = al[(n % 27)] + call
      n /= 27
      call = al[(n % 27)] + call
      n /= 27
      call = al[(n % 27)] + call
      n /= 27
      call = num[(n % 10)] + call
      n /= 10
      call = alnum[(n % 36)] + call
      n /= 36
      call = alnum[(n % 37)] + call
      if n != (n % 37):
          # XXX might be Type 3
          sys.stderr.write("wspr unpack oops1\n")

      # combined grid and power level
      m = bits2num(bits[28:])

      # for a type 1 message,
      # valid power levels are 0, 3, 7, 10, 13, ..., 60.
      power = (m % 128) - 64
      n2 = m
      m /= 128

      if power >= 0 and power <= 60 and (power % 10) in [ 0, 3, 7 ]:
          # Type 1: CALL GRID POWER

          # maidenhead grid locator
          loc4 = m % 10
          m /= 10
          loc2 = m % 18
          m /= 18
          m = 179 - m
          loc3 = m % 10
          m /= 10
          loc1 = m
        
          # 18 letters, A through R.
          if loc1 < 0 or loc1 >= 18:
              sys.stderr.write("wspr unpack oops2\n")
              return None

          grid = ""
          grid += chr(ord('A')+loc1)
          grid += chr(ord('A')+loc2)
          grid += chr(ord('0')+loc3)
          grid += chr(ord('0')+loc4)

          # XXX nhash is too complex for me.
          # ihash = self.nhash(call)
          # self.callhash[ihash] = call

          return "%s %s %d" % (call, grid, power)

      if power > 0:
          # Type 2: PFX/CALL/PFX POWER
          nu = power % 10
          nadd = nu
          if nu > 3:
              nadd = nu - 3
          if nu > 7:
              nadd = nu - 7
          n3 = m + 32768 * (nadd - 1)
          call = self.unpackpfx(n3, call)
          if call == None:
              return None
          power = power - nadd
          if (power % 10) in [ 0, 3, 7 ]:
              # ok
              return "%s %d" % (call, power)
          else:
              sys.stderr.write("wspr unpack oops3\n")
              return None

      # Type 3: hash 6-char-locator power
      # hash is 15 bits.
      ntype = power
      power = -(ntype+1)
      grid6 = call[5:6] + call[0:5]

      # XXX missing something about 0/3/7 and whether grid6 looks like a grid.

      # XXX missing hash and hash table check.
      # ihash = (n2 - ntype - 64) / 128
      # if ihash in self.callhash:
      #     call = self.callhash[ihash]
      # else:
      #     call = "<...>"
      call = "<...>"

      return "%s %s %d" % (call, grid6, power)

  # copied from wsjt-x.
  def unpackpfx(self, n3, call):
      if n3 < 60000:
          # prefix of 1 to 3 characters.
          pfx = [ "?", "?", "?" ]
          for i in [ 2, 1, 0 ]:
              nc = n3 % 37
              if nc >= 0 and nc <= 9:
                  pfx[i] = chr(ord('0') + nc)
              elif nc >= 10 and nc <= 35:
                  pfx[i] = chr(ord('A') + nc)
              else:
                  pfx[i] = " "
              n3 /= 37
          return "%s%s%s/%s" % (pfx[0], pfx[1], pfx[2], call)
      else:
          # suffix of 1 or 2 characters.
          nc = n3 - 60000
          if nc >= 0 and nc <= 9:
              return "%s/%s" % (call, chr(ord('0')+nc))
          if nc >= 0 and nc <= 35:
              return "%s/%s" % (call, chr(ord('A')+nc))
          if nc >= 36 and nc <= 125:
              p0 = chr(ord('0')+(nc-26)/10)
              p1 = chr(ord('0')+(nc-26)%10)
              return "%s/%s%s" % (call, p0, p1)
          sys.stderr.write("unpackpfx oops, call %s\n" % (call))
          return None


def usage():
  sys.stderr.write("Usage: wspr.py -in CARD:CHAN\n")
  sys.stderr.write("       wspr.py -file fff [-chan xxx]\n")
  sys.stderr.write("       wspr.py -bench wsprfiles/xxx.txt\n")
  sys.stderr.write("       wspr.py -opt wsprfiles/xxx.txt\n")
  # list sound cards
  weakaudio.usage()
  sys.exit(1)

def benchmark(wsjtfile, verbose):
    dir = os.path.dirname(wsjtfile)
    minutes = { } # keyed by hhmm
    wsjtf = open(wsjtfile, "r")
    for line in wsjtf:
        line = re.sub(r'\xA0', ' ', line) # 0xA0 -> space
        line = re.sub(r'[\r\n]', '', line)
        m = re.match(r'^([0-9]{4}) +.*$', line)
        if m == None:
            print("oops: " + line)
            continue
        hhmm = m.group(1)
        if not hhmm in minutes:
            minutes[hhmm] = ""
        minutes[hhmm] += line + "\n"
    wsjtf.close()

    info = [ ]
    for hhmm in sorted(minutes.keys()):
        ff = [ x for x in os.listdir(dir) if re.match('......_' + hhmm + '.wav', x) != None ]
        if len(ff) == 1:
            filename = ff[0]
            info.append([ True, filename, minutes[hhmm] ])
        elif len(ff) == 0:
            sys.stderr.write("could not find .wav file in %s for %s\n" % (dir, hhmm))
        else:
            sys.stderr.write("multiple files in %s for %s: %s\n" % (dir, hhmm, ff))

    return benchmark1(dir, info, verbose)

def benchmark1(dir, bfiles, verbose):
    global chan
    chan = 0
    score = 0 # how many we decoded
    wanted = 0 # how many wsjt-x decoded
    for bf in bfiles:
        if not bf[0]: # only the short list
            continue
        if verbose:
            print(bf[1])
        filename = dir + "/" + bf[1]
        r = WSPR()
        r.verbose = False
        r.gowav(filename, chan)
        all = r.get_msgs()
        got = { } # did wsjt-x see this? indexed by msg.
        any_no = False

        wsa = bf[2].split("\n")
        for wsx in wsa:
            wsx = wsx.strip()
            if wsx != "":
                wanted += 1
                wsx = re.sub(r'  *', ' ', wsx)
                found = None
                for x in all:
                    # x is [ minute, hz, msg, decode_time, snr, offset, drift ]
                    mymsg = x[2]
                    mymsg = mymsg.strip()
                    mymsg = re.sub(r'  *', ' ', mymsg)
                    if mymsg in wsx:
                        found = x
                        got[x[2]] = True

                wa = wsx.split(' ')
                wmsg = ' '.join(wa[5:8])
                whz = float(wa[3])
                if whz >= 10 and whz < 11:
                    whz = (whz - 10.1387) * 1000000.0
                elif whz >= 14 and whz < 15:
                    whz = (whz - 14.0956) * 1000000.0
                elif whz < 1.0:
                    whz = whz * 1000000.0

                if found != None:
                    score += 1
                    if verbose:
                        print("yes %4.0f %s (%.1f %.1f) %s" % (float(whz), wa[2], found[1], found[5], wmsg))
                else:
                    any_no = True
                    if verbose:
                        print("no  %4.0f %s %s" % (float(whz), wa[2], wmsg))
                sys.stdout.flush()
        if True and verbose:
            for x in all:
                if not (x[2] in got):
                    print("EXTRA: %6.1f %s" % (x[1], x[2]))
        if False and any_no:
            # help generate small.txt
            for wsx in wsa:
                print "NONONO %s" % (wsx)
    if verbose:
        print("score %d of %d" % (score, wanted))
    return [ score, wanted ]

vars = [
    [ "driftmax", [ 0.75, 1.0, 1.25, 1.5, 1.75, 2, 3 ] ],
    [ "coarse_budget", [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ] ],
    [ "ngoff", [ 1, 2, 3, 4, 6 ] ],
    [ "ndrift", [ 1, 2, 3, 4, 5 ] ],
    [ "fano_limit", [ 5000, 10000, 20000, 30000, 40000, 60000 ] ],
    [ "step_frac", [ 1, 1.5, 2, 2.5, 3, 4 ] ],
    # [ "goff_down", [ 32, 64, 128, ] ],
    # [ "agcwinseconds", [ 0.0, 0.3, 0.6, 0.8, 1.0, 1.5 ] ],
    [ "budget", [ 9, 20, 50 ] ],
    ]

def printvars():
    s = ""
    for v in vars:
        s += "%s=%s " % (v[0], eval(v[0]))
    return s

def optimize(wsjtfile):
    # warm up any caches, JIT, &c.
    r = WSPR()
    r.verbose = False
    r.gowav("wsprfiles/160708_1114.wav", 0)

    for v in vars:
        for val in v[1]:
            old = None
            if "." in v[0]:
                xglob = ""
            else:
                xglob = "global %s ; " % (v[0])
            exec("%sold = %s" % (xglob, v[0]))
            exec("%s%s = %s" % (xglob, v[0], val))

            sc = benchmark(wsjtfile, False)
            exec("%s%s = old" % (xglob, v[0]))
            sys.stdout.write("%s=%s : " % (v[0], val))
            sys.stdout.write("%d\n" % (sc[0]))
            sys.stdout.flush()

if False:
    r = WSPR()
    # k1jt's data symbols example
    bits = [
1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 
    ]
    x = r.process2(bits, [1.0]*len(bits))
    # expecting " K1ABC FN42 37"
    print(x[1])
    sys.exit(0)

filename = None
card = None
bench = None
opt = None

def main():
  global filename, card, bench, opt
  i = 1
  while i < len(sys.argv):
    if sys.argv[i] == "-in":
      card = sys.argv[i+1]
      i += 2
    elif sys.argv[i] == "-file":
      filename = sys.argv[i+1]
      i += 2
    elif sys.argv[i] == "-bench":
      bench = sys.argv[i+1]
      i += 2
    elif sys.argv[i] == "-opt":
      opt = sys.argv[i+1]
      i += 2
    else:
      usage()
  
  if False:
    xr = WSPR()
    xr.test_guess_offset()
    sys.exit(0)
  
  if bench != None:
    sys.stdout.write("# %s %s\n" % (bench, printvars()))
    benchmark(bench, True)
    sys.exit(0)

  if opt != None:
    sys.stdout.write("# %s %s\n" % (opt, printvars()))
    optimize(opt)
    sys.exit(0)
  
  if filename != None and card == None:
    r = WSPR()
    r.verbose = True
    r.gowav(filename, 0)
  elif filename == None and card != None:
    r = WSPR()
    r.verbose = True
    r.opencard(card)
    r.gocard()
  else:
    usage()

if __name__ == '__main__':
  if False:
    pfile = "cprof.out"
    sys.stderr.write("wspr: cProfile -> %s\n" % (pfile))
    import cProfile
    import pstats
    cProfile.run('main()', pfile)
    p = pstats.Stats(pfile)
    p.strip_dirs().sort_stats('time')
    # p.print_stats(10)
    p.print_callers()
  else:
    main()
