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
from scipy.signal import butter, lfilter, firwin
import ctypes
import weakaudio

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

# make a butterworth IIR bandpass filter
def butter_bandpass(lowcut, highcut, samplerate, order=5):
  # http://wiki.scipy.org/Cookbook/ButterworthBandpass
  nyq = 0.5 * samplerate
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='bandpass')
  return b, a

# FIR bandpass filter
# http://stackoverflow.com/questions/16301569/bandpass-filter-in-python
def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def butter_lowpass(cut, samplerate, order=5):
  nyq = 0.5 * samplerate
  cut = cut / nyq
  b, a = scipy.signal.butter(order, cut, btype='lowpass')
  return b, a

#
# frequency shift via hilbert transform (like SSB).
# Lev Givon, https://gist.github.com/lebedov/4428122
#
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""
    return int(numpy.ceil(numpy.log2(numpy.abs(x))))

# f_shift in Hz.
# dt is 1 / sample rate (e.g. 1.0/11025).
# frequencies near the low and high ends of the spectrum are
# reflected, so large shifts will be a problem.
def freq_shift(x, f_shift, dt):
    """Shift the specified signal by the specified frequency."""

    # Pad the signal with zeros to prevent the FFT invoked by the transform from
    # slowing down the computation:
    N_orig = len(x)
    N_padded = 2**nextpow2(N_orig)
    x0 = numpy.append(x, numpy.zeros(N_padded-N_orig, x.dtype))

    t = numpy.arange(0, N_padded)
    lo = numpy.exp(2j*numpy.pi*f_shift*dt*t)
    h = scipy.signal.hilbert(x0)*lo

    ret = h[:N_orig].real
    return ret

def moving_average(a, n):
    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# https://gist.github.com/endolith/255291
# thank you, endolith.
def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f is a vector and x is an index for that vector.
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

fff_cached_window_set = False
fff_cached_window = None

# https://gist.github.com/endolith/255291
def freq_from_fft(sig, rate, minf, maxf):
    global fff_cached_window, fff_cached_window_set
    if fff_cached_window_set == False:
      # this uses a bunch of CPU time.
      fff_cached_window = scipy.signal.blackmanharris(len(sig))
      fff_cached_window_set = True
    assert len(sig) == len(fff_cached_window)

    windowed = sig * fff_cached_window
    f = numpy.fft.rfft(windowed)
    fa = abs(f)

    # find max between minf and maxf
    mini = int(minf * len(windowed) / float(rate))
    maxi = int(maxf * len(windowed) / float(rate))
    i = numpy.argmax(fa[mini:maxi]) + mini # peak bin

    if fa[i] <= 0.0:
        return None

    true_i = parabolic(numpy.log(fa), i)[0] # interpolate

    return rate * true_i / float(len(windowed)) # convert to frequency

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
            block = freq_shift(block, -freq_off, 1.0/self.jrate)
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
      self.audio = weakaudio.open(desc, self.cardrate)

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
    filter = butter_bandpass(1380, 1620, self.cardrate, 3)
    samples = lfilter(filter[0], filter[1], samples)
    # down-convert from 1400 to 100.
    samples = freq_shift(samples, -self.downhz, 1.0/self.cardrate)
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
          taps = bandpass_firwin(ntaps, thz-goff_hz, thz+goff_hz, self.jrate)
          # yx = lfilter(taps, 1.0, samples)
          # yx = numpy.convolve(samples, taps, mode='valid')
          # yx = scipy.signal.convolve(samples, taps, mode='valid')
          yx = scipy.signal.fftconvolve(samples, taps, mode='valid')
          # hack to match size of lfilter() output
          yx = numpy.append(numpy.zeros(ntaps-1), yx)
          yx = abs(yx)

          # scipy.resample() works but is much too slow.
          #yx = scipy.signal.resample(yx, len(yx) / downfactor)
          yx = moving_average(yx, downfactor)
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
  sys.stderr.write("       wspr.py -bench\n")
  sys.stderr.write("       wspr.py -big\n")
  # list sound cards
  weakaudio.usage()
  sys.exit(1)

# write a stereo file
def writewav2(left, right, filename, rate):
  ww = wave.open(filename, 'w')
  ww.setnchannels(2)
  ww.setsampwidth(2)
  ww.setframerate(rate)

  # interleave.
  a = numpy.zeros(len(left) + len(right))
  a[0::2] = left
  a[1::2] = right

  # convert to 16-bit ints
  a = numpy.array(a, dtype=numpy.int16)

  # convert to python raw byte string
  a = a.tostring()

  ww.writeframes(a)

  ww.close()

# write a mono file
def writewav1(left, filename, rate):
  ww = wave.open(filename, 'w')
  ww.setnchannels(1)
  ww.setsampwidth(2)
  ww.setframerate(rate)

  # convert to 16-bit ints
  a = numpy.array(left, dtype=numpy.int16)

  # convert to python raw byte string
  a = a.tostring()

  ww.writeframes(a)

  ww.close()

# what wsjt-x says is in each benchmark wav file.
# files in wsprfiles/*
bfiles = [
[ False, "160708_1058.wav", """
1058  -16  -1.9   10.140188    0   K4PRA         EM74     37   1506
1058  -21  -1.8   10.140209    0   W4MO          EL87     37   1988
""" ],
[ False, "160708_1104.wav", """
1104  -26   1.9   10.140222    0   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1106.wav", """
1106  -17  -2.0   10.140113   -4   WB4HIR        EM95     33   1162
1106  -23  -1.8   10.140209    1   W4MO          EL87     37   1988
""" ],
[ False, "160708_1108.wav", """
1108  -17  -2.0   10.140107    0   N9PBD         EM58     37   1585
""" ],
[ False, "160708_1110.wav", """
1110  -14  -2.0   10.140188   -1   K4PRA         EM74     37   1506
""" ],
[ False, "160708_1112.wav", """
1112  -11  -1.9   10.140187    0   W4MO          EL86     37   2084
1112  -24  -1.7   10.140266    0   W3CSW         FM19     37    604
""" ],
[ True, "160708_1114.wav", """
1114  -26  -1.9   10.140210    0   W4MO          EL87     37   1988
1114  -27  -2.0   10.140252    0   WA3DSP        FN20     23    401
""" ],
[ False, "160708_1116.wav", """
1116  -25  -1.9   10.140121    0   WC8J          EN80     23   1025
1116  -18  -1.8   10.140218    0   KD6RF         EM22     37   2383
""" ],
[ False, "160708_1118.wav", """
1118  -24  -2.0   10.140113   -4   WB4HIR        EM95     33   1162
1118  -13  -2.0   10.140188   -1   K4PRA         EM74     37   1506
1118  -24  -1.9   10.140200    0   K5OK          EM12     37   2538
""" ],
[ False, "160708_1120.wav", """
1120  -27  -1.9   10.140161   -2   AJ8S          EM89     27   1062
1120  -26  -2.6   10.140217   -1   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1122.wav", """
1122    0  -1.0   10.140165    1   K3SC          FM19     37    604
""" ],
[ False, "160708_1124.wav", """
1124  -18  -2.0   10.140107    0   N9PBD         EM58     37   1585
1124  -17  -1.9   10.140187   -1   W4MO          EL86     37   2084
1124  -20   2.0   10.140222    0   KE9FQ         EN61     33   1328
1124   -4  -1.8   10.140266    0   W3CSW         FM19     37    604
""" ],
[ False, "160708_1126.wav", """
1126  -26  -2.0   10.140113   -4   WB4HIR        EM95     33   1162
1126  -16  -1.7   10.140191    0   K3CXW         FM19     30    604
""" ],
[ False, "160708_1128.wav", """
1128  -27  -2.0   10.140200    1   K5OK          EM12     37   2538
1128  -19  -2.0   10.140266   -1   K9AN          EN50     33   1516
""" ],
[ False, "160708_1130.wav", """
1130  -28  -1.9   10.140121    0   WC8J          EN80     23   1025
1130  -25  -2.1   10.140161    0   AJ8S          EM89     27   1062
1130  -21  -2.0   10.140188   -1   K4PRA         EM74     37   1506
1130  -17  -2.6   10.140217    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1134.wav", """
1134  -21  -2.0   10.140114   -4   WB4HIR        EM95     33   1162
1134  -15  -1.9   10.140192    0   K3CXW         FM19     30    604
1134  -28  -1.8   10.140210    0   W4MO          EL87     37   1988
""" ],
[ False, "160708_1136.wav", """
1136  -15  -2.0   10.140187   -1   W4MO          EL86     37   2084
1136   -2  -1.9   10.140266   -1   W3CSW         FM19     37    604
""" ],
[ False, "160708_1138.wav", """
1138   -1  -1.0   10.140164    1   K3SC          FM19     37    604
1138  -15  -2.0   10.140188   -1   K4PRA         EM74     37   1506
""" ],
[ False, "160708_1140.wav", """
1140  -11  -2.6   10.140217   -1   WB4CSD        FM08     27    810
1140  -23  -2.0   10.140266   -1   K9AN          EN50     33   1516
""" ],
[ False, "160708_1142.wav", """
1142  -13  -2.0   10.140094    0   W9MDO         EN60     37   1352
1142  -27   1.3   10.140166    1   <...>         EL89TP   30   1757
1142  -24  -1.7   10.140246    3   WD4AHB        EL89     30   1799
""" ],
[ False, "160708_1144.wav", """
1144  -16   2.0   10.140222    0   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1146.wav", """
1146  -19  -1.9   10.140187    0   W4MO          EL86     37   2084
1146  -24  -1.9   10.140210    0   W4MO          EL87     37   1988
""" ],
[ False, "160708_1148.wav", """
1148  -10  -2.0   10.140107    0   N9PBD         EM58     37   1585
1148  -18  -2.0   10.140188   -1   K4PRA         EM74     37   1506
1148  -27  -1.7   10.140216   -1   KD6RF         EM22     37   2383
1148  -22  -2.0   10.140266    0   K9AN          EN50     33   1516
""" ],
[ False, "160708_1150.wav", """
1150  -20  -1.0   10.140164    1   K3SC          FM19     37    604
1150  -14  -2.7   10.140217   -1   WB4CSD        FM08     27    810
1150  -11  -1.9   10.140266    0   W3CSW         FM19     37    604
""" ],
[ True, "160708_1152.wav", """
1152  -29  -2.0   10.140241   -1   AJ8S          EM89     27   1062
""" ],
[ False, "160708_1154.wav", """
1154  -18  -1.8   10.140192    0   K3CXW         FM19     30    604
""" ],
[ False, "160708_1156.wav", """
1156  -15  -1.9   10.140187    0   W4MO          EL86     37   2084
""" ],
[ True, "160708_1158.wav", """
1158  -19  -2.1   10.140114   -4   WB4HIR        EM95     33   1162
1158  -27  -2.0   10.140121    0   WC8J          EN80     23   1025
1158  -25  -2.1   10.140188   -1   K4PRA         EM74     37   1506
""" ],
[ False, "160708_1200.wav", """
1200  -22  -2.1   10.140161    0   AJ8S          EM89     27   1062
1200  -11  -2.6   10.140216    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1202.wav", """
1202  -26  -2.1   10.140241    0   AJ8S          EM89     27   1062
""" ],
[ 1204, "160708_1204.wav", """
1204  -22   1.9   10.140222    0   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1206.wav", """
1206  -21  -1.8   10.140192    0   K3CXW         FM19     30    604
""" ],
[ False, "160708_1208.wav", """
1208  -16  -2.0   10.140094    0   W9MDO         EN60     37   1352
1208  -12  -2.0   10.140107    0   N9PBD         EM58     37   1585
1208   -9  -2.0   10.140251    0   AE0MT         EN34     40   1787
""" ],
[ False, "160708_1210.wav", """
1210  -17  -2.1   10.140188   -1   K4PRA         EM74     37   1506
1210  -25  -1.8   10.140209    1   W4MO          EL87     37   1988
1210  -15  -2.7   10.140216    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1212.wav", """
1212  -19  -2.0   10.140112   -4   WB4HIR        EM95     33   1162
1212  -17  -0.9   10.140162    0   K3SC          FM19     37    604
1212  -13  -2.0   10.140187    0   W4MO          EL86     37   2084
""" ],
[ False, "160708_1214.wav", """
1214  -10  -0.6   10.140178   -1   KF9KV         EN52     37   1476
1214  -22  -1.8   10.140266    0   W3CSW         FM19     37    604
""" ],
[ True, "160708_1216.wav", """
1216  -28  -1.9   10.140121    0   WC8J          EN80     23   1025
""" ],
[ False, "160708_1218.wav", """
1218  -21  -1.7   10.140209    1   W4MO          EL87     37   1988
""" ],
[ False, "160708_1220.wav", """
1220   -9  -0.6   10.140178   -1   KF9KV         EN52     37   1476
1220  -21  -2.6   10.140216    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1222.wav", """
1222  -17  -2.1   10.140106    0   N9PBD         EM58     37   1585
1222  -20  -2.1   10.140187   -2   K4PRA         EM74     37   1506
1222  -26  -1.9   10.140244    1   WD4AHB        EL89     30   1799
1222  -15  -2.1   10.140266    0   K9AN          EN50     33   1516
""" ],
[ True, "160708_1224.wav", """
1224  -23  -2.1   10.140112   -4   WB4HIR        EM95     33   1162
1224  -13  -1.9   10.140187    0   W4MO          EL86     37   2084
1224  -27  -2.0   10.140196    0   W8DGN         EM79     23   1222
1224  -19   1.9   10.140222    0   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1226.wav", """
1226  -14  -1.9   10.140266    0   W3CSW         FM19     37    604
""" ],
[ False, "160708_1228.wav", """
1228  -25  -2.1   10.140199    0   K5OK          EM12     37   2538
1228  -15  -2.1   10.140266    0   K9AN          EN50     33   1516
""" ],
[ False, "160708_1230.wav", """
1230  -26  -2.0   10.140120    0   WC8J          EN80     23   1025
1230  -26  -1.7   10.140209    0   W4MO          EL87     37   1988
1230  -22  -2.7   10.140216   -1   WB4CSD        FM08     27    810
""" ],
[ True, "160708_1232.wav", """
1232  -18  -2.0   10.140112   -2   WB4HIR        EM95     33   1162
1232  -16  -2.0   10.140187    0   W4MO          EL86     37   2084
1232  -24  -2.1   10.140187   -1   K4PRA         EM74     37   1506
1232  -24  -1.9   10.140218   -1   KD6RF         EM22     37   2383
""" ],
[ False, "160708_1236.wav", """
1236  -26   3.4   10.140209    0   KO8C          EN81     30    999
""" ],
[ False, "160708_1240.wav", """
1240  -21  -1.7   10.140138    0   N0UE          EN34     10   1787
1240  -17  -2.0   10.140186    0   W4MO          EL86     37   2084
1240  -24  -1.8   10.140209    1   W4MO          EL87     37   1988
1240  -25  -2.6   10.140216    0   WB4CSD        FM08     27    810
1240  -18  -2.1   10.140265    0   K9AN          EN50     33   1516
""" ],
[ False, "160708_1242.wav", """
1242  -23  -2.1   10.140113   -4   WB4HIR        EM95     33   1162
""" ],
[ False, "160708_1244.wav", """
1244  -27  -2.0   10.140120    0   WC8J          EN80     23   1025
1244  -15  -2.1   10.140187   -1   K4PRA         EM74     37   1506
1244  -16   0.9   10.140221    0   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1248.wav", """
1248  -28  -2.1   10.140199    0   K5OK          EM12     37   2538
1248  -24  -1.9   10.140216    0   KD6RF         EM22     37   2383
""" ],
[ False, "160708_1250.wav", """
1250  -14  -2.7   10.140215    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1252.wav", """
1252  -24  -1.9   10.140265    0   W3CSW         FM19     37    604
""" ],
[ False, "160708_1254.wav", """
1254  -24  -2.1   10.140113   -3   WB4HIR        EM95     33   1162
""" ],
[ False, "160708_1300.wav", """
1300  -18  -2.0   10.140186   -1   W4MO          EL86     37   2084
1300  -20  -2.6   10.140215    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1302.wav", """
1302  -19  -2.0   10.140265    0   K9AN          EN50     33   1516
""" ],
[ True, "160708_1304.wav", """
1304  -23  -2.1   10.140093    0   KM4WPM        EM74     47   1506
1304  -28   3.4   10.140209    0   KO8C          EN81     30    999
1304  -11   1.9   10.140221    0   KE9FQ         EN61     33   1328
""" ],
[ True, "160708_1306.wav", """
1306  -16  -2.0   10.140092    0   KM4WPM        EM74     47   1506
1306  -29   2.7   10.140267    0   W3HH          EL89     30   1799
""" ],
[ False, "160708_1308.wav", """
1308  -18  -2.1   10.140106   -1   N9PBD         EM58     37   1585
1308  -17  -2.0   10.140186    0   W4MO          EL86     37   2084
1308  -25  -1.8   10.140208    1   W4MO          EL87     37   1988
""" ],
[ True, "160708_1310.wav", """
1310  -21  -2.4   10.140187    0   K4PRA         EM74     37   1506
1310  -20  -3.7   10.140215    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1316.wav", """
1316  -25  -1.8   14.097113    0   KD6RF         EM22     37   2383
""" ],
[ False, "160708_1318.wav", """
1318  -24  -2.1   14.097153    0   K4COD         EM73     33   1580
""" ],
[ False, "160708_1320.wav", """
1320  -18  -2.0   14.097043    0   K5XL          EM12     33   2538
1320  -21  -1.9   14.097077    0   WA9EIC        EN60     37   1352
""" ],
[ False, "160708_1322.wav", """
1322  -27   3.0   14.097121   -1   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1324.wav", """
1324  -24  -2.3   14.097032    0   N0UE          EN34     10   1787
1324   -6  -2.1   14.097163    0   K9AN          EN50     33   1516
""" ],
[ False, "160708_1330.wav", """
1330  -20  -2.1   14.097148    0   AE0MT         EN34     40   1787
1330  -14  -2.0   14.097153    0   K4COD         EM73     33   1580
1330   -4  -2.1   14.097163    0   K9AN          EN50     33   1516
""" ],
[ False, "160708_1332.wav", """
1332   -8  -2.0   14.097043    0   K5XL          EM12     33   2538
1332  -16  -1.8   14.097116    0   KD6RF         EM22     37   2383
1332  -15  -2.2   14.097143   -1   KA3JIJ        EM84     17   1370
""" ],
[ False, "160708_1340.wav", """
1340  -19  -2.1   14.097037    0   W7NIX         EM65     33   1585
1340  -27  -1.9   14.097077    0   WA9EIC        EN60     37   1352
1340  -27  -2.1   14.097136   -1   KK4YEL        EL98     23   1796
""" ],
[ False, "160708_1342.wav", """
1342  -17  -2.2   14.097094    0   W8DGN         EM79     23   1222
1342  -24  -2.1   14.097142    0   KA3JIJ        EM84     17   1370
""" ],
[ False, "160708_1344.wav", """
1344  -16  -2.1   14.097042    1   K5XL          EM12     33   2538
1344  -14  -2.0   14.097152    0   K4COD         EM73     33   1580
""" ],
[ False, "160708_1348.wav", """
1348  -11  -2.1   10.140105    0   N9PBD         EM58     37   1585
1348  -10  -2.1   10.140186   -1   K4PRA         EM74     37   1506
1348  -21  -1.4   10.140188    0   KE4MXW        EL98     37   1796
1348  -19  -1.9   10.140226    0   KD6RF         EM22     37   2383
1348   -6  -2.1   10.140249    0   AE0MT         EN34     40   1787
1348  -16  -2.1   10.140265    0   K9AN          EN50     33   1516
""" ],
[ True, "160708_1350.wav", """
1350  -23  -2.1   10.140094    0   W4ENN         EM64     20   1650
1350  -26   3.4   10.140204    0   K4EH          EM73     37   1580
1350   -5  -2.7   10.140214    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1352.wav", """
1352  -24   1.2   10.140165    1   <...>         EL89TP   30   1757
1352  -19  -2.1   10.140194    0   W8DGN         EM79     23   1222
""" ],
[ False, "160708_1354.wav", """
1354   -9  -2.2   10.140113   -4   WB4HIR        EM95     33   1162
1354  -22   2.6   10.140267    0   W3HH          EL89     30   1799
""" ],
[ False, "160708_1356.wav", """
1356  -19  -1.4   10.140188    0   KE4MXW        EL98     37   1796
""" ],
[ False, "160708_1358.wav", """
1358  -21  -2.0   10.140119    0   WC8J          EN80     23   1025
1358  -12  -2.1   10.140186   -1   K4PRA         EM74     37   1506
1358  -23  -2.1   10.140198    0   K5OK          EM12     37   2538
1358    1  -2.0   10.140265    0   W3CSW         FM19     37    604
""" ],
[ False, "160708_1400.wav", """
1400  -20   1.2   10.140165    1   WD4LHT        EL89     30   1799
1400   -4  -2.7   10.140215    0   WB4CSD        FM08     27    810
1400  -17  -1.9   10.140227    1   KD6RF         EM22     37   2383
""" ],
[ False, "160708_1404.wav", """
1404  -16  -1.4   10.140188    0   KE4MXW        EL98     37   1796
1404  -10   2.8   10.140220    0   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1406.wav", """
1406  -23  -2.1   10.140193    0   W8DGN         EM79     23   1222
1406  -26   2.6   10.140267    0   W3HH          EL89     30   1799
""" ],
[ False, "160708_1408.wav", """
1408    1  -2.1   10.140092    0   W9MDO         EN60     37   1352
1408   -7  -2.1   10.140105    0   N9PBD         EM58     37   1585
1408  -26  -2.1   10.140114    0   W9MDO         EN60     37   1352
1408  -18  -2.1   10.140192    0   K2LYV         EL88     37   1892
1408  -18  -2.2   10.140265    0   K9AN          EN50     33   1516
""" ],
[ True, "160708_1410.wav", """
1410  -14  -2.1   10.140116   -2   WB4HIR        EM95     33   1162
1410  -25  -2.1   10.140198    0   K5OK          EM12     37   2538
1410   -1  -2.7   10.140215    0   WB4CSD        FM08     27    810
1410    7  -2.0   10.140264    0   W3CSW         FM19     37    604
""" ],
[ False, "160708_1412.wav", """
1412  -21   1.2   10.140165    1   <WD4LHT>      EL89TP   30   1757
""" ],
[ False, "160708_1416.wav", """
1416  -27  -2.1   10.140119    0   WC8J          EN80     23   1025
1416   -2  -2.1   10.140191    1   K4RCG/4                37
1416  -19  -2.0   10.140231    0   KD6RF         EM22     37   2383
1416  -22   2.5   10.140267    0   W3HH          EL89     30   1799
""" ],
[ True, "160708_1420.wav", """
1420   -7  -2.2   10.140105    0   N9PBD         EM58     37   1585
1420  -19  -2.1   10.140116   -2   WB4HIR        EM95     33   1162
1420  -22   1.2   10.140164    1   WD4LHT        EL89     30   1799
1420  -27  -2.0   10.140198    0   K5OK          EM12     37   2538
1420    2  -2.8   10.140215    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1424.wav", """
1424  -18   2.8   10.140220    0   KE9FQ         EN61     33   1328
""" ],
[ False, "160708_1428.wav", """
1428   -3  -2.6   10.140105    0   N9PBD         EM58     37   1585
1428  -20  -2.6   10.140192    0   K2LYV         EL88     37   1892
1428  -22   2.0   10.140267    0   W3HH          EL89     30   1799
""" ],
[ False, "160708_1430.wav", """
1430  -26  -2.1   10.140119    0   WC8J          EN80     23   1025
1430  -20   1.2   10.140164    0   <WD4LHT>      EL89TP   30   1757
1430   -4  -2.7   10.140215    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1432.wav", """
1432  -11  -2.1   10.140116   -2   WB4HIR        EM95     33   1162
1432  -24  -2.0   10.140234    0   KD6RF         EM22     37   2383
""" ],
[ False, "160708_1436.wav", """
1436  -19   2.4   10.140267    0   W3HH          EL89     30   1799
""" ],
[ False, "160708_1438.wav", """
1438  -27  -2.1   10.140193    0   W8DGN         EM79     23   1222
""" ],
[ False, "160708_1440.wav", """
1440  -18  -2.1   10.140116   -3   WB4HIR        EM95     33   1162
1440   -3  -2.7   10.140215    0   WB4CSD        FM08     27    810
""" ],
[ False, "160708_1444.wav", """
1444  -25  -2.2   10.140119    0   WC8J          EN80     23   1025
1444  -18   2.8   10.140220    0   KE9FQ         EN61     33   1328
1444  -21   2.7   10.140266    0   W3HH          EL89     30   1799
""" ],
[ True, "160708_1448.wav", """
1448   -8  -2.2   10.140092    0   W9MDO         EN60     37   1352
1448  -14  -2.2   10.140116   -1   WB4HIR        EM95     33   1162
1448  -30  -1.8   10.140193    0   WA8KNE        EM90     37   1602
1448  -23  -2.1   10.140237    0   KD6RF         EM22     37   2383
""" ],
]

def benchmark(verbose):
    global chan
    chan = 0
    score = 0 # how many we decoded
    wanted = 0 # how many wsjt-x decoded
    for bf in bfiles:
        if not bf[0]: # only the short list
            continue
        if verbose:
            print bf[1]
        filename = "wsprfiles/" + bf[1]
        r = WSPR()
        r.verbose = False
        r.gowav(filename, chan)
        all = r.get_msgs()
        got = { } # did wsjt-x see this? indexed by msg.

        wsa = bf[2].split("\n")
        for wsx in wsa:
            wsx = wsx.strip()
            if wsx != "":
                wanted += 1
                wsx = re.sub(r'  *', ' ', wsx)
                found = False
                for x in all:
                    mymsg = x[2]
                    mymsg = mymsg.strip()
                    mymsg = re.sub(r'  *', ' ', mymsg)
                    if mymsg in wsx:
                        found = True
                        got[x[2]] = True

                wa = wsx.split(' ')
                wmsg = ' '.join(wa[5:8])
                whz = float(wa[3])
                if whz >= 10 and whz < 11:
                    whz = (whz - 10.1387) * 1000000.0
                if whz >= 14 and whz < 15:
                    whz = (whz - 14.0956) * 1000000.0

                if found:
                    score += 1
                    if verbose:
                        print "yes %4.0f %s" % (float(whz), wmsg)
                else:
                    if verbose:
                        print "no  %4.0f %s" % (float(whz), wmsg)
                sys.stdout.flush()
        if True and verbose:
            for x in all:
                if not (x[2] in got):
                    print "MISSING: %6.1f %s" % (x[1], x[2])
    if verbose:
        print "score %d of %d" % (score, wanted)
    return [ score, wanted ]

def bigmark():
    global budget, agcwinseconds, step_frac, ngoff, goff_taps, goff_down, goff_hz, fano_limit, driftmax, driftinc, coarse_bins, coarse_budget
    while True:
        budget = 50
        step_frac = 4
        driftmax = 2.0
        ngoff = random.choice([ 2, 4, 6, 8, 10, 12 ])
        fano_limit = random.choice([ 5000, 10000, 20000, 40000])
        agcwinseconds = random.choice([ 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 ])
        goff_taps = random.choice([625, 751, 813, 875])
        goff_down = random.choice([64, 128, 256])
        goff_hz = random.choice([1, 1.5, 2])
        # driftmax = random.choice([0.75, 1.0, 1.25, 1.5, 2.0])
        driftinc = random.choice([0.1666, 0.333, 0.5, 0.666])
        coarse_bins = random.choice([1,2,4])
        coarse_budget = random.choice([ 0.1, 0.2, 0.3, 0.4, 0.5 ])

        sc = benchmark(False)
        print "%d : %d %.1f %d %d %d %d %.1f %d %.1f %.1f %d %.1f" % (sc[0],
                                                          budget,
                                                          agcwinseconds,
                                                          step_frac,
                                                          ngoff,
                                                          goff_taps,
                                                          goff_down,
                                                          goff_hz,
                                                          fano_limit,
                                                          driftmax,
                                                          driftinc,
                                                          coarse_bins,
                                                          coarse_budget
        )

def smallmark():
    global budget, agcwinseconds, step_frac, ngoff, goff_taps, goff_down, goff_hz, fano_limit, driftmax, driftinc, coarse_bins, coarse_budget

    o_budget = budget
    o_agcwinseconds = agcwinseconds
    o_step_frac = step_frac
    o_ngoff = ngoff
    o_goff_taps = goff_taps
    o_goff_down = goff_down
    o_goff_hz = goff_hz
    o_fano_limit = fano_limit
    o_driftmax = driftmax
    o_driftinc = driftinc
    o_coarse_bins = coarse_bins
    o_coarse_budget = coarse_budget

    for ngoff in [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ]:
        sc = benchmark(False)
        print "%d : ngoff=%d" % (sc[0], ngoff)
    ngoff = o_ngoff

    for goff_taps in [ 401, 451, 501 ]:
        sc = benchmark(False)
        print "%d : goff_taps=%d" % (sc[0], goff_taps)
    goff_taps = o_goff_taps

    for fano_limit in [ 5000, 10000, 15000, 20000, 30000, 40000 ]:
        sc = benchmark(False)
        print "%d : fano_limit=%d" % (sc[0], fano_limit)
    fano_limit = o_fano_limit

    for goff_down in [ 32, 64, 128, 256, 512 ]:
        sc = benchmark(False)
        print "%d : goff_down=%d" % (sc[0], goff_down)
    goff_down = o_goff_down

    for budget in [ 10, 20, 30, 40, 50, 100 ]:
        sc = benchmark(False)
        print "%d : budget=%d" % (sc[0], budget)
    budget = o_budget

    for coarse_budget in [ 0.1, 0.2, 0.3, 0.4, 0.5 ]:
        sc = benchmark(False)
        print "%d : coarse_budget=%.1f" % (sc[0], coarse_budget)
    coarse_budget = o_coarse_budget

    for agcwinseconds in [ 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 ]:
        sc = benchmark(False)
        print "%d : agcwinseconds=%.1f" % (sc[0], agcwinseconds)
    agcwinseconds = o_agcwinseconds

    for driftinc in [ 0.333, 0.5, 0.666 ]:
        sc = benchmark(False)
        print "%d : driftinc=%.1f" % (sc[0], driftinc)
    driftinc = o_driftinc

    for goff_hz in [ 0.5, 1, 1.5, 2, 2.5 ]:
        sc = benchmark(False)
        print "%d : goff_hz=%.1f" % (sc[0], goff_hz)
    goff_hz = o_goff_hz

    for step_frac in [ 2, 4, 8 ]:
        sc = benchmark(False)
        print "%d : step_frac=%d" % (sc[0], step_frac)
    step_frac = o_step_frac

    for driftmax in [ 0.75, 1, 1.25, 1.5, 1.75, 2 ]:
        sc = benchmark(False)
        print "%d : driftmax=%.1f" % (sc[0], driftmax)
    driftmax = o_driftmax

    for coarse_bins in [ 1, 2, 4 ]:
        sc = benchmark(False)
        print "%d : coarse_bins=%d" % (sc[0], coarse_bins)
    coarse_bins = o_coarse_bins

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
big = False
small = False

def main():
  global filename, card, bench, big, small
  i = 1
  while i < len(sys.argv):
    if sys.argv[i] == "-in":
      card = sys.argv[i+1]
      i += 2
    elif sys.argv[i] == "-file":
      filename = sys.argv[i+1]
      i += 2
    elif sys.argv[i] == "-bench":
      bench = True
      i += 1
    elif sys.argv[i] == "-big":
      big = True
      i += 1
    elif sys.argv[i] == "-small":
      small = True
      i += 1
    else:
      usage()
  
  if False:
    xr = WSPR()
    xr.test_guess_offset()
  
  if bench == True:
    benchmark(True)
    sys.exit(0)

  if big == True:
    bigmark()
    sys.exit(0)
  
  if small == True:
    smallmark()
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
