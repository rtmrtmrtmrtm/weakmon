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
import thread
import re
import random
from scipy.signal import lfilter
import ctypes
import weakaudio
import weakutil

#
# WSPR tuning parameters.
#
budget = 50 # max seconds of CPU time, per file or two-minute interval (50).
agcwinseconds = 0.8 # AGC window size, seconds (2, 0.5, 0.8).
step_frac = 4 # fraction of FFT bin for frequency search (4).
ngoff = 3 # look at this many of guess_offset()'s results (6, 4).
goff_taps = 451 # guess_offset filter taps (501, 701, 751, 813).
goff_down = 64 # guess_offset down-conversion factor (32, 128).
goff_hz = 2 # guess_offset filter +/- hz (2).
fano_limit = 20000 # how hard fano will work (10000).
driftmax = 2.0 # look at drifts from -driftmax to +driftmax (2).
driftinc = 0.5 # drifts at these intervals (0.666).
coarse_bins = 1 # search granularity (2, 1).
coarse_budget = 0.25 # fraction of budget to spend calling guess_offset() (0.25).

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
      self.msgs_lock = thread.allocate_lock()
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
                  print "%s got %d samples, writing to x.wav" % (self.ts(time.time()), len(samples[i0:]))
                  writewav1(samples[i0:], "x.wav", self.cardrate)

              self.process(samples[i0:], t)

              bufbuf = [ ]
              nsamples = 0

  # received a message, add it to the list.
  # offset in seconds.
  # drift in hz/minute.
  def got_msg(self, minute, hz, txt, snr, offset, drift):
      if self.verbose:
          print "%6.1f %.1f %.1f %d %s" % (hz, offset, drift, snr, txt)
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
    global budget, agcwinseconds, step_frac, ngoff, goff_taps, goff_down, goff_hz, fano_limit, driftmax, driftinc, coarse_bins, coarse_budget

    # samples_time is UNIX time that samples[0] was
    # sampled by the sound card.
    samples_minute = self.minute(samples_time + 60)

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
    startslop = 1 * self.jrate + agcwinlen  # add this much at start
    endslop = 4 * self.jrate + agcwinlen    # add this much at end
    sm = numpy.mean(samples) # pad with plausible signal levels
    sd = numpy.std(samples)
    samples = numpy.append(numpy.random.normal(sm, sd, startslop), samples)
    samples = numpy.append(samples, numpy.random.normal(sm, sd, endslop))

    if agcwinlen > 0:
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

    # coarse-granularity FFT over the whole two minutes to
    # find likely frequencies.
    coarseblock = self.jblock / coarse_bins
    coarse_hz = self.jrate / float(coarseblock)
    coarse = numpy.zeros(coarseblock / 2 + 1)
    coarseblocks = 0
    for i in range(2*self.jrate, len(samples)-2*self.jrate, coarseblock/2):
        block = samples[i:i+coarseblock]
        a = numpy.fft.rfft(block)
        a = abs(a)
        coarse = numpy.add(coarse, a)
        coarseblocks = coarseblocks + 1
    coarse = coarse / coarseblocks # sum -> average
    
    # sort coarse bins, biggest signal first.
    coarse_rank = range(int(min_hz/coarse_hz), int(max_hz/coarse_hz))
    coarse_rank = sorted(coarse_rank, key=lambda i : -coarse[i])

    # calculate noise for snr, mimicing wsjtx wsprd.c.
    # first average in freq domain over 7-bin window.
    # then noise from 30th percentile.
    nn = numpy.convolve(coarse, [ 1, 1, 1, 1, 1, 1, 1 ])
    nn = nn / 7.0
    nn = nn[6:]
    nns = sorted(nn[int(min_hz/coarse_hz):int(max_hz/coarse_hz)])
    noise = nns[int(0.3*len(nns))]
    noise /= coarse_bins # so that it refers to one WSPR-sized bin.

    # set up to cache FFTs at various offsets.
    xf = Xform(samples, self.jrate, self.jblock)

    # avoid checking a given step_hz more than once.
    # already is indexed by int(hz / step_hz)
    step_hz = bin_hz / step_frac
    already = { }

    exhaustive = False

    t0 = time.time()

    # for each WSJT FFT bin and offset, the strength of
    # the sync correlation. perhaps includes just the most
    # likely-looking coarse bins.
    # fine_rank[i] = [ hz, offset, strength ]
    fine_rank = [ ]

    for ci in coarse_rank:
        if time.time() - t0 >= budget * coarse_budget:
            break

        hz0 = ci * coarse_hz # center of coarse bin

        # assume we get a good coarse bin that either starts
        # at the lower edge of the lower wspr bin, or (because
        # of alignment) starts 1/2 coarse bin higher.
        # we want to avoid needless searching here since we
        # have a limited CPU budget (<= 2 minutes).
        if coarse_bins == 1:
            start_hz = hz0 - 1*bin_hz
            end_hz = hz0 + 0.5*bin_hz
        if coarse_bins == 2:
            start_hz = hz0 - 2*bin_hz
            end_hz = hz0 + 0*bin_hz
        if coarse_bins == 4:
            start_hz = hz0 - 4*bin_hz
            end_hz = hz0 - 1*bin_hz

        # print "%.1f %.1f..%.1f" % (hz0, start_hz, end_hz)
        for hz in numpy.arange(start_hz, end_hz, step_hz):
            hzkey = int(hz / step_hz)
            if hzkey in already:
                break
            already[hzkey] = True
            offsets = self.guess_offset(samples, hz)
            # offsets[i] is [ offset, strength ]
            offsets = offsets[0:ngoff]
            triples = [ [ hz, offset, strength ] for [ offset, strength ] in offsets ]
            fine_rank += triples

    # print "%d in fine_rank, spent %.1f seconds" % (len(fine_rank), time.time() - t0)

    # call Fano on the bins with the higest sync correlation first,
    # since there's not enough time to look at all bins.
    fine_rank = sorted(fine_rank, key=lambda r : -r[2])

    # store each message just once, to suppress duplicates.
    # indexed by message text; value is [ samples_minute, hz, msg, snr, offset, drift ]
    msgs = { }
    
    for rr in fine_rank:
        if time.time() - t0 >= budget:
            break
        hz = rr[0]
        offset = rr[1]
        if offset < 0:
            continue
        drifts = numpy.arange(-driftmax, driftmax+0.1, driftinc)
        for drift in drifts:
            hza = [ hz - drift, hz + drift ]
    
            ss = xf.get(hza, offset)
    
            # ss has one element per symbol time.
            # ss[i] is a 4-element FFT.
    
            # first symbol is in ss[0]
            # return is [ hza, msg, snr ]
            x = self.process1(samples_minute, ss[0:162], hza, noise)

            if x != None:
                # info is [ minute, hz, msg, snr, offset, drift ]
                info = [ samples_minute, numpy.mean(hza), x[1], x[2], offset, drift ]
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
        offset = (info[4] / float(self.jrate)) - 3.0 # convert to seconds
        drift = info[5] # hz / minute
        self.got_msg(info[0], hz, txt, snr, offset, drift)

  def hz0(self, hza, sym):
      hz = hza[0] + (hza[1] - hza[0]) * (sym / float(len(pattern)))
      return hz

  # since there have been so many bugs in guess_offset().
  def test_guess_offset(self):
      mo = 0
      bin_hz = self.jrate / float(self.jblock)
      for hz in numpy.arange(180, 420, 27.1):
          for nstart in range(0, 3000, 571):
              for nend in range(500, 3000, 737):
                  samples = numpy.random.normal(0, 0.5, nstart)
                  # samples = numpy.zeros(nstart)
                  for p in pattern:
                      bit = random.randint(0, 1)
                      if p > 0:
                          phz = hz + bit*2*bin_hz + bin_hz
                      else:
                          phz = hz + bit*2*bin_hz
                      tttt = numpy.arange(0, 0.6826, 1.0/1500.0)
                      sss = numpy.sin(2 * numpy.pi * phz * tttt)
                      samples = numpy.append(samples, sss)
                  samples = numpy.append(samples, numpy.random.normal(0, 0.5, nend))
                  # samples = numpy.append(samples, numpy.zeros(nend))
                  samples = samples * 1000
                  x = self.guess_offset(samples, hz)
                  print "%d %d %d" % (abs(x[0] - nstart), nstart, x[0])
                  sys.stdout.flush()
                  #assert x[0] >= nstart - 32
                  #assert x[0] <= nstart + 32
                  mo = max(mo, abs(x[0]  - nstart))
      # jul 18 2016 -- max diff was 53.
      print "max diff %d" % (mo)

  # returns an array of [ offset, strength ], sorted
  # by strength, most-plausible first.
  def guess_offset(self, samples, hz):
      global goff_taps, goff_down, goff_hz
      bin_hz = self.jrate / float(self.jblock)

      # FIR filter so we can predict delay through the filter.
      ntaps = goff_taps # 301 to 1001 all work, though 1501 is bad
      fdelay = ntaps / 2

      # average y down to a much lower rate to make the
      # correlate() go faster. 32 works well.
      downfactor = goff_down

      # filter each of the four tones
      tones = [ ]
      for tone in range(0, 4):
          thz = hz + tone*bin_hz
          # +/- 2 works well here.
          taps = weakutil.bandpass_firwin(ntaps, thz-goff_hz, thz+goff_hz, self.jrate)
          # yx = lfilter(taps, 1.0, samples)
          # yx = numpy.convolve(samples, taps, mode='valid')
          # yx = scipy.signal.convolve(samples, taps, mode='valid')
          yx = scipy.signal.fftconvolve(samples, taps, mode='valid')
          # hack to match size of lfilter() output
          yx = numpy.append(numpy.zeros(ntaps-1), yx)
          yx = abs(yx)

          # scipy.resample() works but is much too slow.
          #yx = scipy.signal.resample(yx, len(yx) / downfactor)
          yx = weakutil.moving_average(yx, downfactor)
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

      z = numpy.array([])
      for p in pattern:
        x = float(p)
        z = numpy.append(z, x * numpy.ones(self.jblock / downfactor))

      cc = numpy.correlate(tt, z)

      indices = range(0, len(cc))
      indices = sorted(indices, key=lambda i : -cc[i])
      offsets = numpy.multiply(indices, downfactor)
      offsets = numpy.subtract(offsets, fdelay)
      both = [ [ offsets[i], cc[indices[i]] ] for i in range(0, len(offsets)) ]

      return both

  # returns None or [ hz, start, nerrs, msg, twelve ]
  def process1(self, samples_minute, m, hza, noise):
    if len(m) < 162:
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

    if snr < -30:
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

        softsyms.append( [ logp0, logp1 ] )

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
    print x[1]
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
