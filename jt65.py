#!/usr/local/bin/python

#
# decode JT65
#
# inspired by the QEX May/June 2016 article by K9AN and K1JT
# about soft-decision JT65 decoding.
#
# much information and code from the WSJT-X source distribution.
#
# uses Phil Karn's Reed-Solomon software.
#

import numpy
import wave
import weakaudio
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
from ctypes import c_int, byref, cdll

#
# performance tuning parameters.
#
budget = 6     # CPU seconds (6 for benchmarks, 9 for real).
noffs = 4      # look for sync every jblock/noffs (4)
off_scores = 4 # consider off_scores*noffs starts per freq bin (3, 4)
pass1_frac = 0.9 # fraction budget to spend before subtracting (0.5, 0.9)
hetero_thresh = 7 # zero out bin that wins too many times (9, 5, 7)
soft_iters = 75 # try r-s soft decode this many times (35, 125, 75)

# Phil Karn's Reed-Solomon decoder.
# copied from wsjt-x, along with wrapkarn.c.
librs = cdll.LoadLibrary("librs/librs.so")

# the JT65 sync pattern
pattern = [
  1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,
  -1,1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1,1,-1,1,-1,1,-1,1,1,
  -1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,1,1,
  -1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,
  1,1,1,1,1,1
  ]

# start of special 28-bit callsigns, e.g. CQ.
NBASE = 37*36*10*27*27*27

# start of special grid locators for sig strength &c.
NGBASE = 180*180

# does this decoded message contain text that's generated
# mistakenly from noise by the reed-solomon decoder?
def broken_msg(msg):
    bads = [ "OL6MWK", "1S9LND", "9M3QHC", "TIKK+", "J87FOE", "000AAA",
             "TG7HQQ", "475IVR", "L16RAH", "XO2QLH" ]
    for bad in bads:
        if bad in msg:
            return True
    return False

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
    t = numpy.arange(0, N_padded)
    lo = numpy.exp(2j*numpy.pi*f_shift*dt*t)
    x0 = numpy.append(x, numpy.zeros(N_padded-N_orig, x.dtype))
    h = scipy.signal.hilbert(x0)*lo
    ret = h[:N_orig].real
    return ret

#
# gray code
# https://rosettacode.org/wiki/Gray_code
#

def int2bin(n, nb):
  'From positive integer to list of binary bits, msb at index 0'
  if n:
    bits = []
    while n:
      n,remainder = divmod(n, 2)
      bits.insert(0, remainder)
    out = bits
  else:
    out = [0]
  while len(out) < nb:
    out = [0] + out
  return out
 
 
def bin2int(bits):
  'From binary bits, msb at index 0 to integer'
  i = 0
  for bit in bits:
    i = i * 2 + bit
  return i

def bin2gray(x, nb):
  bits = int2bin(x, nb)
  gbits = bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]
  return bin2int(gbits)
 
def gray2bin(x, nb):
  bits = int2bin(x, nb)
  b = [bits[0]]
  for nextb in bits[1:]: b.append(b[-1] ^ nextb)
  return bin2int(b)

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
    mini = int(minf * len(windowed) / rate)
    maxi = int(maxf * len(windowed) / rate)
    i = numpy.argmax(fa[mini:maxi]) + mini # peak bin

    if fa[i] <= 0.0:
        return None

    true_i = parabolic(numpy.log(fa), i)[0] # interpolate

    return rate * true_i / float(len(windowed)) # convert to frequency

class JT65:
  debug = False

  offset = 0

  def __init__(self):
      self.done = False
      self.msgs_lock = thread.allocate_lock()
      self.msgs = [ ]
      self.verbose = False
      self.hints = [ ] # each element is [ hz, call ] to look for

      self.jrate = 11025/2 # sample rate for processing (FFT &c)
      self.jblock = 4096/2 # samples per symbol

      # set self.start_time to the UNIX time of the start
      # of a UTC minute.
      now = int(time.time())
      gm = time.gmtime(now)
      self.start_time = now - gm.tm_sec

  # return the minute number for t, a UNIX time in seconds.
  # truncates down, so best to pass a time mid-way through a minute.
  def minute(self, t):
      dt = t - self.start_time
      dt /= 60.0
      return int(dt)

  def second(self, t):
      dt = t - self.start_time
      dt /= 60.0
      m = int(dt)
      return 60.0 * (dt - m)

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
    z = self.wav.readframes(1024)
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
      self.cardrate = 11025 # XXX
      self.audio = weakaudio.open(desc, self.cardrate)

  def gocard(self):
      samples_time = time.time()
      bufbuf = [ ]
      nsamples = 0
      while self.done == False:
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

          # wait until we have enough samples through 49th second of minute.
          # we want to start on the minute (i.e. a second before nominal
          # start time), and end a second after nominal end time.
          # thus through 46.75 + 2 = 48.75.

          sec = self.second(samples_time)
          if sec >= 49 and nsamples >= 49*self.cardrate:
              # we have >= 49 seconds of samples, and second of minute is >= 49.

              samples = numpy.concatenate(bufbuf)

              # sample # of start of minute.
              i0 = len(samples) - self.cardrate * self.second(samples_time)
              i0 = int(i0)
              t = samples_time - (len(samples)-i0) * (1.0/self.cardrate)
              self.process(samples[i0:], t)

              bufbuf = [ ]
              nsamples = 0

  def close(self):
      # ask gocard() thread to stop.
      self.done = True 

  # received a message, add it to the list.
  def got_msg(self, minute, hz, msg, nerrs, snr):
      mm = [ minute, hz, msg, time.time(), nerrs, snr ]

      self.msgs_lock.acquire()

      # already in msgs with worse nerrs?
      found = False
      for i in range(max(0, len(self.msgs)-20), len(self.msgs)):
          xm = self.msgs[i]
          if xm[0] == minute and abs(xm[1] - hz) < 10 and xm[2] == msg:
              # we already have this msg
              found = True
              if nerrs < xm[4]:
                  self.msgs[i] = mm
                  
      if found == False:
          self.msgs.append(mm)

      self.msgs_lock.release()

  # someone wants a list of all messages received.
  # each msg is [ minute, hz, msg, decode_time, nerrs, snr ]
  def get_msgs(self):
      self.msgs_lock.acquire()
      a = copy.copy(self.msgs)
      self.msgs_lock.release()
      return a

  def process(self, samples, samples_time):
    global budget, noffs, off_scores, pass1_frac

    # for budget.
    t0 = time.time()

    # samples_time is UNIX time that samples[0] was
    # sampled by the sound card.
    samples_minute = self.minute(samples_time + 30)

    if self.cardrate != self.jrate:
      # reduce rate from self.cardrate to self.jrate.
      assert self.jrate >= 2 * 2300
      filter = butter_lowpass(2300.0, self.cardrate, order=10)
      samples = scipy.signal.lfilter(filter[0],
                                     filter[1],
                                     samples)
      if self.cardrate/2 == self.jrate:
        samples = samples[0::2]
      else:
        # brutally slow sometimes, e.g. 11025->5512
        want = (len(samples) / float(self.cardrate)) * self.jrate
        want = int(want)
        samples = scipy.signal.resample(samples, want)

    # pad at start+end b/c transmission might have started early
    # or late.
    sm = numpy.mean(abs(samples))
    samples = numpy.concatenate(((numpy.random.random(self.jrate*3) - 0.5) * sm * 2,
                                 samples,
                                 (numpy.random.random(self.jrate*4) - 0.5) * sm * 2))

    bin_hz = self.jrate / float(self.jblock)
    minbin = 5
    maxbin = int(2500 / bin_hz) # avoid JT9

    # assign a score to each frequency bin,
    # according to how similar it seems to a sync tone pattern.

    # offs = [ 0, self.jblock / 2 ]
    # offs = [ 0, 512, 1024, 3*512 ]
    # offs = [ 0, 256, 2*256, 3*256, 4*256, 5*256, 6*256, 7*256 ]
    offs = [ (x*self.jblock)/noffs for x in range(0, noffs) ]
    m = []
    noises = numpy.zeros(self.jblock/2 + 1) # for SNR
    nnoises = 0
    for oi in range(0, len(offs)):
      m.append([])
      si = offs[oi]
      while si + self.jblock <= len(samples):
          block = samples[si:si+self.jblock]
          # block = block * scipy.signal.blackmanharris(len(block))
          a = numpy.fft.rfft(block)
          a = abs(a)
          m[oi].append(a)
          noises = numpy.add(noises, a)
          nnoises += 1
          si += self.jblock
    
    noises /= nnoises

    # calculate noise for snr, mimicing wsjtx wsprd.c.
    # first average in freq domain over 7-bin window.
    # then noise from 30th percentile.
    nn = numpy.convolve(noises, [ 1, 1, 1, 1, 1, 1, 1 ])
    nn = nn / 7.0
    nn = nn[6:]
    nns = sorted(nn[minbin:maxbin])
    noise = nns[int(0.3*len(nns))]

    # scores[i] = [ bin, correlation, valid, start ]
    scores = [ ]

    # for each frequency bin, strength of correlation with sync pattern.
    # searches (w/ correlation) for best match to sync pattern.
    # tries different offsets in time (from offs[]).
    for j in range(minbin, maxbin):
      for oi in range(0, len(offs)):
        v = [ ]
        for mx in m[oi]:
          v.append(mx[j])

        cc = numpy.correlate(v, pattern)

        indices = range(0, len(cc))
        indices = sorted(indices, key=lambda i : -cc[i])
        indices = indices[0:off_scores] # XXX should be a parameter
        for ii in indices:
          scores.append([ j, cc[ii], True, offs[oi] + ii*self.jblock ])

    # scores[i] = [ bin, correlation, valid, start ]
              
    if False:
        # don't do this any more b/c we want to explore the best
        # few offsets per bin.

        # sort by bin (frequency).
        scores = sorted(scores, key=lambda sc : sc[0])

        # suppress frequencies that aren't local peak scores,
        # reflected in scores[i][2] (true for ok, false for ignore).
        for i in range(0, len(scores)):
          ok = True
          sc = scores[i][1]
          for j in range(i-1,i+2):
              if i != j and j >= 0 and j < len(scores) and sc < scores[j][1]:
                  ok = False
          scores[i][2] = ok

        # filter out valid=False entries.
        scores = [ x for x in scores if x[2] ]

    # highest scores first.
    scores = sorted(scores, key=lambda sc : -sc[1])

    ssamples = numpy.copy(samples) # subtracted
    already = { } # suppress duplicate msgs

    # first without subtraction
    # we get decodes all the way down to e.g. i=49.
    # but only try through 40 this time, in order to
    # ensure there's time to start decoding on subtracted
    # signals before our alloted 10 seconds expires.
    for hint in self.hints:
      self.process1(samples_minute, samples, hint[0], hint[1], noise, None, already)
    i = 0
    while i < len(scores) and (time.time() - t0) < budget * pass1_frac:
        hz = scores[i][0] * (self.jrate / float(self.jblock))
        x = self.process1(samples_minute, samples, hz, None, noise, scores[i][3], already)
        if x != None:
            scores[i][2] = False
            ssamples = self.subtract_v3(ssamples, x[0], x[1], x[4])
        i += 1

    # filter out entries that we just decoded.
    scores = [ x for x in scores if x[2] ]

    # now try again, on subtracted signal.
    # we do a complete new pass since a strong signal might have
    # been unreadable due to another signal at a somewhat higher
    # frequency.
    for hint in self.hints:
      self.process1(samples_minute, ssamples, hint[0], hint[1], noise, None, already)
    i = 0
    while i < len(scores) and (time.time() - t0) < budget:
        hz = scores[i][0] * (self.jrate / float(self.jblock))
        x = self.process1(samples_minute, ssamples, hz, None, noise, scores[i][3], already)
        if x != None:
            ssamples = self.subtract_v3(ssamples, x[0], x[1], x[4])
        i += 1

  # subtract a decoded signal (hz/start/twelve) from the samples,
  # to that we can then decode weaker signals underneath it.
  # i.e. interference cancellation.
  # generates the right tone for each symbol, finds the best
  # offset w/ correlation, finds the amplitude, subtracts in the time domain.
  # no FFT.
  def subtract_v3(self, osamples, hza, start, twelve):
      sender = JT65Send()

      # the 126 symbols, each 0..66
      symbols = sender.symbols(twelve)

      samples = numpy.copy(osamples)

      if start < 0:
          samples = numpy.append([0.0]*(-start), samples)
      else:
          samples = samples[start:]

      for i in range(0, 126):
          # generate 4096 samples of the symbol's tone.
          t = numpy.arange(0, (float(self.jblock)/self.jrate), 1.0/self.jrate)
          sync_hz = self.sync_hz(hza, i)
          hz = sync_hz + symbols[i] * (self.jrate / float(self.jblock))
          tone = numpy.cos(t * 2.0 * numpy.pi * hz)

          # nominal start of symbol in samples[]
          i0 = i * self.jblock
          i1 = i0 + self.jblock
          
          # search +/- slop
          i0 = max(0, i0 - 50)
          i1 = min(len(samples), i1 + 50)

          cc = numpy.correlate(samples[i0:i1], tone)
          mm = numpy.argmax(cc) # thus samples[i0+mm]

          # what is the amplitude?
          # if actual signal had a peak of 1.0, then
          # correlation would be sum(tone*tone).
          cx = cc[mm]
          c1 = numpy.sum(tone * tone)
          amp = cx / c1

          samples[i0+mm:i0+mm+self.jblock] -= tone * amp

      if start < 0:
          nsamples = samples[(-start):]
      else:
          nsamples = numpy.append(osamples[0:start], samples)

      return nsamples

  #
  # guess the sample number at which the first sync symbol starts.
  #
  # this is basically never used any more; it seems more effective
  # to just look for sync correlation at a bunch of sub-symbol offsets.
  #
  def guess_offset(self, samples, hz):
      if True:
          # FIR filter so we can predict delay through the filter.
          ntaps = 1001 # XXX 1001 works well
          fdelay = ntaps / 2
          taps = bandpass_firwin(ntaps, hz-4, hz+4, self.jrate)

          # filtered = lfilter(taps, 1.0, samples)

          # filtered = numpy.convolve(samples, taps, mode='valid')
          # filtered = scipy.signal.convolve(samples, taps, mode='valid')
          filtered = scipy.signal.fftconvolve(samples, taps, mode='valid')
          # hack to match size of lfilter() output
          filtered = numpy.append(numpy.zeros(ntaps-1), filtered)
      else:
          # butterworth IIR
          fdelay = 920 # XXX I don't know how to predict filter delay.
          filter = butter_bandpass(hz - 4, hz + 4, self.jrate, 3)
          filtered = lfilter(filter[0], filter[1], samples)

      y = filtered

      # correlation rapidly changes from negative to positive
      # even when in a sync tone.
      y = abs(y)

      # average y down to a much lower rate to make the
      # correlate() go faster. scipy.resample() works but
      # is much too slow.
      downfactor = 32
      #y = scipy.signal.resample(y, len(y) / downfactor)
      ya = moving_average(y, downfactor)
      y = ya[0::downfactor]

      z = numpy.array([])
      for p in pattern:
        if p > 0:
          z = numpy.append(z, numpy.ones(self.jblock / downfactor))
        else:
          z = numpy.append(z, -1*numpy.ones(self.jblock / downfactor))

      cc = numpy.correlate(y, z)
      mm = numpy.argmax(cc)

      # sort, putting index of highest correlation first.
      iv = sorted(range(0, len(cc)), key = lambda i : -cc[i])

      offs = [ ]
      for mm in iv[0:20]:
          if len(offs) == 0 or (len(offs) < 3 and abs(mm - offs[0]) >= 5 and abs(mm-offs[-1]) >= 5):
              offs.append(mm)

      for i in range(0, len(offs)):
          mm = offs[i]
          mm *= downfactor
          mm -= fdelay
          offs[i] = mm

      offs = offs[0:1] # XXX too expensive to look at multiple offsets
      return offs

  # the sync tone is believed to be hz to within one fft bin.
  # return hz with higher resolution.
  # returns a two-element array of hz at start, hz at end.
  def guess_freq(self, samples, hz):
      bin_hz = self.jrate / float(self.jblock)
      bin = int(round(hz / bin_hz))
      freqs = [ ]
      for i in range(0, len(pattern)):
          if pattern[i] == 1:
              sx = samples[i*self.jblock:(i+1)*self.jblock]
              ff = freq_from_fft(sx, self.jrate,
                                 bin_hz * (bin - 1),
                                 bin_hz * (bin + 2))
              if ff != None and not numpy.isnan(ff):
                  freqs.append(ff)

      if len(freqs) < 1:
          return None

      # nhz = numpy.median(freqs)
      # nhz = numpy.mean(freqs)
      # return nhz

      # frequencies at 1/4 and 3/4 way through samples.
      n = len(freqs)
      m1 = numpy.median(freqs[0:n/2])
      m2 = numpy.median(freqs[n/2:]) 
      
      # frequencies at start and end.
      m0 = m1 - (m2 - m1) / 2.0
      m3 = m2 + (m2 - m1) / 2.0

      hza = [ m0, m3 ]

      return hza

  # given hza[hz0,hzn] from guess_freq(),
  # and a symbol number (0..126),
  # return the sync bin.
  # the point is to correct for frequency drift.
  def sync_bin(self, hza, sym):
      hz = self.sync_hz(hza, sym)
      bin_hz = self.jrate / float(self.jblock) # FFT bin size, in Hz
      bin = int(round(hz / bin_hz))
      return bin

  def sync_hz(self, hza, sym):
      hz = hza[0] + (hza[1] - hza[0]) * (sym / 126.0)
      return hz

  # xhz is the sync tone frequency.
  # returns None or [ hz, start, nerrs, msg, twelve ]
  # if hint!=None, it's some text to demand in the decode msg,
  # to guide soft decode.
  def process1(self, samples_minute, samples, xhz, hint, noise, start, already):
    if len(samples) < 126*self.jblock:
        return None

    bin_hz = self.jrate / float(self.jblock) # FFT bin size, in Hz

    if start != None:
        ret = self.process1a(samples_minute, samples, xhz, hint, start, noise, already)
        if ret != None:
            return ret
    else:
        starts = self.guess_offset(samples, xhz)
        for start in starts:
            ret = self.process1a(samples_minute, samples, xhz, hint, start, noise, already)
            if ret != None:
                return ret
    return None

  def process1a(self, samples_minute, samples, xhz, hint, start, noise, already):
    global hetero_thresh

    bin_hz = self.jrate / float(self.jblock) # FFT bin size, in Hz

    if start < 0:
        samples = numpy.append([0.0]*(-start), samples)
    else:
        samples = samples[start:]
    if len(samples) < 126*self.jblock:
        return None

    hza = self.guess_freq(samples, xhz)
    if hza == None:
        return None
    if self.sync_bin(hza, 0) < 5 or self.sync_bin(hza, 0) + 2+64 > 2048:
        return None
    #freq_off = hz - (sync_bin * bin_hz)
    # shift samples down in frequency by freq_off
    #samples = freq_shift(samples, -freq_off, 1.0/self.jrate)

    m = [ ]
    for i in range(0, 126):
      # block = block * scipy.signal.blackmanharris(len(block))
      sync_bin = self.sync_bin(hza, i)
      sync_hz = self.sync_hz(hza, i)
      freq_off = sync_hz - (sync_bin * bin_hz)
      block = samples[i*self.jblock:(i+1)*self.jblock]
      block = freq_shift(block, -freq_off, 1.0/self.jrate)
      a = numpy.fft.rfft(block)
      a = abs(a)
      m.append(a)

    # look for bins that win too often, perhaps b/c they are
    # syncs from higher-frequency JT65 transmissions.
    wins = [ 0 ] * 66
    for pi in range(0,126):
      if pattern[pi] == -1:
        bestj = None
        bestv = None
        sync_bin = self.sync_bin(hza, pi)
        for j in range(sync_bin+2, sync_bin+2+64):
          if bestj == None or m[pi][j] > bestv:
            bestj = j
            bestv = m[pi][j]
        wins[bestj-sync_bin] += 1

    # zero out bins that win too often. a given symbol
    # (bin) should only appear two or three times in
    # a transmission.
    # XXX make more conservative since we're subtracting
    # decoded transmissions.
    for j in range(2, 66):
        if wins[j] >= hetero_thresh:
            # zero bin j
            for pi in range(0,126):
                sync_bin = self.sync_bin(hza, pi)
                m[pi][sync_bin+j] = 0

    # for each non-sync time slot, decide which tone is strongest,
    # which yields the channel symbol.
    sa = [ ]
    strength = [ ] # symbol signal / 2nd-best signal
    sigs = [ ] # for SNR
    for pi in range(0,126):
      if pattern[pi] == -1:
        sync_bin = self.sync_bin(hza, pi)
        a = sorted(range(0,64), key=lambda bin: -m[pi][sync_bin+2+bin])
        sa.append(a[0])

        b0 = sync_bin+2+a[0] # bucket w/ strongest signal
        b1 = sync_bin+2+a[1] # bucket w/ 2nd-strongest signal

        s0 = m[pi][b0] # level of strongest symbol
        sigs.append(s0)

        s1 = numpy.mean(m[pi][sync_bin+2:sync_bin+2+64]) # mean of bins in same time slot

        if s1 != 0.0:
            strength.append(s0 / s1)
        else:
            strength.append(0.0)

    [ nerrs, msg, twelve ] = self.process2(sa, strength, hint)

    if nerrs < 0 or broken_msg(msg):
        return None

    # SNR
    sig = numpy.mean(sigs)
    # power rather than voltage.
    rawsnr = (sig*sig) / (noise*noise)
    # the "-1" turns (s+n)/n into s/n
    rawsnr -= 1
    if rawsnr < 0.1:
        rawsnr = 0.1
    rawsnr /= (2500.0 / 2.7) # 2.7 hz noise b/w -> 2500 hz b/w
    snr = 10 * math.log10(rawsnr)


    if self.verbose and not (msg in already):
      print "%6.1f %5d: %2d %3.0f %s" % ((hza[0]+hza[1])/2.0, start, nerrs, snr, msg)
    already[msg] = True

    self.got_msg(samples_minute, numpy.mean(hza), msg, nerrs, snr)
    return [ hza, start, nerrs, msg, twelve ]

  # sa[] is 63 channel symbols, each 0..63.
  # it needs to be un-gray-coded, un-interleaved,
  # un-reed-solomoned, &c.
  # strength[] indicates how sure we are about each symbol
  #   (ratio of winning FFT bin to second-best bin).
  # if hint!=None, it's some text to require in any
  # soft-decoded message.
  def process2(self, sa, strength, hint):
    global soft_iters

    # un-gray-code
    for i in range(0, len(sa)):
      sa[i] = gray2bin(sa[i], 6)

    # un-interleave
    un = [ 0 ] * 63
    un_strength = [ 0 ] * 63
    for c in range(0, 7):
      for r in range(0, 9):
        un[(r*7)+c] = sa[(c*9)+r]
        un_strength[(r*7)+c] = strength[(c*9)+r]

    sa = un
    strength = un_strength

    [nerrs,twelve] = self.rs_decode(sa, [])
    if nerrs >= 0:
        # successful decode.
        sym0 = twelve[0]
        if numpy.array_equal(twelve, [sym0]*12):
            # a JT69 signal...
            return [-1, "???", None]
        msg = self.unpack(twelve)
        if not broken_msg(msg):
            return [nerrs, msg, twelve]

    if True:
        # attempt soft decode 

        # at this point we know there must be at least 25
        # errors, since otherwise Reed-Solomon would have
        # decoded.

        # for each symbol time, how likely to be wrong.
        weights = numpy.divide(1.0, numpy.add(strength, 1.0))
        total_weight = numpy.sum(weights)

        # weakest first
        worst = sorted(range(0, 63), key = lambda i: strength[i])

        # try various numbers of erasures.
        first = None
        for iter in range(0, soft_iters):
            nera = (iter % 35) + 5
            eras = [ ]
            for j in range(0, 63):
                if len(eras) >= nera:
                    break
                si = worst[j]
                if random.random() < nera*(weights[si] / total_weight):
                    # rs_decode() has this weird convention for erasures.
                    eras.append(63-1-si)
            [nerrs,twelve] = self.rs_decode(sa, eras)
            if nerrs >= 0:
                msg = self.unpack(twelve)
                if broken_msg(msg):
                    continue
                if hint == None or hint in msg:
                    if first != None:
                        print "**** soft hint %s (ignored %s, hint %s, hint)" % (msg, first)
                    return [nerrs, msg, twelve ]
                first = msg

    # Reed Solomon could not decode.
    return [-1, "???", None ]

  # convert packed character to Python string.
  # 0..9 a..z space
  def charn(self, c):
    if c >= 0 and c <= 9:
      return chr(ord('0') + c)
    if c >= 10 and c < 36:
      return chr(ord('A') + c - 10)
    if c == 36:
      return ' '
    # sys.stderr.write("jt65 charn(%d) bad\n" % (c))
    return '?'

  # x is an integer, e.g. nc1 or nc2, containing all the
  # call sign bits from a packed message.
  # 28 bits.
  def unpackcall(self, x):
    a = [ 0, 0, 0, 0, 0, 0 ]
    a[5] = self.charn((x % 27) + 10) # + 10 b/c only alpha+space
    x = int(x / 27)
    a[4] = self.charn((x % 27) + 10)
    x = int(x / 27)
    a[3] = self.charn((x % 27) + 10)
    x = int(x / 27)
    a[2] = self.charn(x%10) # digit only
    x = int(x / 10)
    a[1] = self.charn(x % 36) # letter or digit
    x = int(x / 36)
    a[0] = self.charn(x)
    return ''.join(a)

  # extract maidenhead locator
  def unpackgrid(self, ng):
    if ng == NGBASE+1:
        return "    "
    if ng >= NGBASE+1 and ng < NGBASE+31:
      return " -%02d" % (ng - (NGBASE+1)) # sig str, -01 to -30 DB
    if ng >= NGBASE+31 and ng < NGBASE+62:
      return "R-%02d" % (ng - (NGBASE+31))
    if ng == NGBASE+62:
      return "RO  "
    if ng == NGBASE+63:
      return "RRR "
    if ng == NGBASE+64:
      return "73  "
      
      
    lat = (ng % 180) - 90
    ng = int(ng / 180)
    long = (ng * 2) - 180

    g = "%c%c%c%c" % (ord('A') + int((179-long)/20),
                      ord('A') + int((lat+90)/10),
                      ord('0') + int(((179-long)%20)/2),
                      ord('0') + (lat+90)%10)

    #print "lat %d, long %d, %s" % (lat, long, g)
    return g

  def unpack(self, a):
    # a[] has 12 0..63 symbols, or 72 bits.
    # turn them into the original human-readable message.
    # unpack([61, 37, 30, 28, 9, 27, 61, 58, 26, 3, 49, 16]) -> "G3LTF DL9KR JO40"
    nc1 = 0 # 28 bits of first call
    nc1 |= a[4] >> 2 # 4 bits
    nc1 |= a[3] << 4 # 6 bits
    nc1 |= a[2] << 10 # 6 bits
    nc1 |= a[1] << 16 # 6 bits
    nc1 |= a[0] << 22 # 6 bits

    nc2 = 0 # 28 bits of second call
    nc2 |= (a[4] & 3) << 26 # 2 bits
    nc2 |= a[5] << 20 # 6 bits
    nc2 |= a[6] << 14 # 6 bits
    nc2 |= a[7] << 8 # 6 bits
    nc2 |= a[8] << 2 # 6 bits
    nc2 |= a[9] >> 4 # 2 bits

    ng = 0 # 16 bits of grid
    ng |= (a[9] & 15) << 12 # 4 bits
    ng |= a[10] << 6 # 6 bits
    ng |= a[11]

    if ng >= 32768:
      txt = self.unpacktext(nc1, nc2, ng)
      return txt

    if nc1 == NBASE+1:
      c2 = self.unpackcall(nc2)
      grid = self.unpackgrid(ng)
      return "CQ %s %s" % (c2, grid)

    if nc1 >= 267649090 and nc1 <= 267698374:
        # CQ with suffix (e.g. /QRP)
        n = nc1 - 267649090
        sf = self.charn(n % 37)
        n /= 37
        sf = self.charn(n % 37) + sf
        n /= 37
        sf = self.charn(n % 37) + sf
        n /= 37
        c2 = self.unpackcall(nc2)
        grid = self.unpackgrid(ng)
        return "CQ %s/%s %s" % (c2, sf, grid)

    c1 = self.unpackcall(nc1)
    if c1 == "CQ9DX ":
        c1 = "CQ DX "
    c2 = self.unpackcall(nc2)
    grid = self.unpackgrid(ng)
    return "%s %s %s" % (c1, c2, grid)

  def unpacktext(self, nc1, nc2, nc3):
    c = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ +-./?"

    nc3 &= 32767
    if (nc1 & 1) != 0:
      nc3 += 32768
    nc1 >>= 1
    if (nc2 & 1) != 0:
      nc3 += 65536
    nc2 >>= 1
      
    msg = [""] * 22

    for i in range(4, -1, -1):
      j = nc1 % 42
      msg[i] = c[j]
      nc1 = nc1 / 42

    for i in range(9, 4, -1):
      j = nc2 % 42
      msg[i] = c[j]
      nc2 = nc2 / 42

    for i in range(12, 9, -1):
      j = nc3 % 42
      msg[i] = c[j]
      nc3 = nc3 / 42

    return ''.join(msg)

  # call the Reed-Solomon decoder.
  # symbols is 63 integers, the channel symbols after
  # un-gray-coding and un-interleaving.
  # era is an array of integers indicating which
  # symbols are erasures.
  # returns 12 original symbols of the packed message,
  # or none.
  def rs_decode(self, symbols, era):
    int63 = c_int * 63
    int12 = c_int * 12

    recd0 = int63()
    for i in range(0, 63):
      recd0[i] = symbols[i]

    era0 = int63()
    numera0 = c_int()
    numera0.value = len(era)
    for i in range(0, len(era)):
        era0[i] = era[i]

    decoded = int12()
    nerr = c_int()
    nerr.value = 0
  
    librs.rs_decode_(recd0, era0, byref(numera0), decoded, byref(nerr))

    if nerr.value < 0:
      # could not decode
      return [-1, None]
    
    a = [ ]
    for i in range(0, 12):
      a.append(decoded[i])

    return [ nerr.value, a ]

class JT65Send:
    def __init__(self):
        pass

    # convert a character into a number; order is
    # 0..9 A..Z space
    def nchar(self, ch):
        if ch >= '0' and ch <= '9':
            return ord(ch) - ord('0')
        if ch >= 'A' and ch <= 'Z':
            return ord(ch) - ord('A') + 10
        if ch == ' ':
            return 36
        print "NT65Send.nchar(%s) oops" % (ch)
        return 0

    # returns a 28-bit number.
    # we need call to be:
    #   lds lds d ls ls ls
    # l-etter, d-igit, s-pace
    # 28-bit number's high bits correspond to first call sign character.
    def packcall(self, call):
        call = call.strip()
        call = call.upper()

        if call == "CQ":
            return NBASE + 1
        if call == "QRZ":
            return NBASE + 2
        if call == "DE":
            return 267796945

        if len(call) > 2 and len(call) < 6 and not call[2].isdigit():
            call = " " + call
        while len(call) < 6:
            call = call + " "

        if re.search(r'^[A-Z0-9 ][A-Z0-9 ][0-9][A-Z ][A-Z ][A-Z ]$', call) == None:
            return -1

        x = 0
        x += self.nchar(call[0])

        x *= 36
        x += self.nchar(call[1])

        x *= 10
        x += self.nchar(call[2])

        x *= 27
        x += self.nchar(call[3]) - 10

        x *= 27
        x += self.nchar(call[4]) - 10

        x *= 27
        x += self.nchar(call[5]) - 10
        
        return x

    # returns 16-bit number.
    # g is maidenhead grid, or signal strength, or 73.
    def packgrid(self, g):
        g = g.strip()
        g = g.upper()

        if g[0] == '-':
            return NGBASE + 1 + int(g[1:])
        if g[0:2] == 'R-':
            return NGBASE + 31 + int(g[2:])
        if g == "RO":
            return NGBASE + 62
        if g == "RRR":
            return NGBASE + 63
        if g == "73":
            return NGBASE+64

        if re.match(r'^[A-R][A-R][0-9][0-9]$', g) == None:
            return -1

        long = (ord(g[0]) - ord('A')) * 20
        long += (ord(g[2]) - ord('0')) * 2
        long = 179 - long

        lat = (ord(g[1]) - ord('A')) * 10
        lat += (ord(g[3]) - ord('0')) * 1
        lat -= 90

        x = (long + 180) / 2
        x *= 180
        x += lat + 90

        return x

    # turn three numbers into 12 6-bit symbols.
    def pack3(self, nc1, nc2, g):
        a = [0] * 12
        a[0] = (nc1 >> 22) & 0x3f
        a[1] = (nc1 >> 16) & 0x3f
        a[2] = (nc1 >> 10) & 0x3f
        a[3] = (nc1 >> 4) & 0x3f
        a[4] = ((nc1 & 0xf) << 2) | ((nc2 >> 26) & 0x3)
        a[5] = (nc2 >> 20) & 0x3f
        a[6] = (nc2 >> 14) & 0x3f
        a[7] = (nc2 >> 8) & 0x3f
        a[8] = (nc2 >> 2) & 0x3f
        a[9] = ((nc2 & 0x3) << 4) | ((g >> 12) & 0xf)
        a[10] = (g >> 6) & 0x3f
        a[11] = (g >> 0) & 0x3f
        return a
        
    def pack(self, msg):
        msg = msg.strip()
        msg = re.sub(r'  *', ' ', msg)
        msg = re.sub(r'^CQ DX ', 'CQ9DX ', msg)

        # try CALL CALL GRID
        a = msg.split(' ')
        if len(a) == 3:
            nc1 = self.packcall(a[0])
            nc2 = self.packcall(a[1])
            g = self.packgrid(a[2])
            if nc1 >= 0 and nc2 >= 0 and g >= 0:
                return self.pack3(nc1, nc2, g)

        # never finished this -- no text &c.
        sys.stderr.write("JT65Send.pack(%s) -- cannot parse\n" % (msg))
        sys.exit(1)

        return [0] * 12

    def testpack(self):
        r = JT65()
        for g in [ "FN42", "-22", "R-01", "RO", "RRR", "73", "AA00", "RR99" ]:
            pg = self.packgrid(g)
            upg = r.unpackgrid(pg)
            if g != upg.strip():
                print "packgrid oops %s" % (g)
        for call in [ "AB1HL", "K1JT", "M0TRJ", "KK4BMV", "2E0CIN", "HF9D",
                      "6Y4K", "D4Z", "8P6DR", "ZS2I", "3D2RJ",
                      "WB3D", "S59GCD", "T77C", "4Z5AD", "A45XR", "OJ0V",
                      "6Y6N", "S57V", "3Z0R" ]:
            # XXX 3XY1T doesn't work
            pc = self.packcall(call)
            upc = r.unpackcall(pc)
            if call != upc.strip():
                print "packcall oops %s %d %s" % (call, pc, upc)
        for msg in [ "AB1HL K1JT FN42", "CQ DX CO3HMR EL82", "KD6HWI PY7VI R-12",
                     "KD5RBW TU 73", "CQ N5OSK EM25", "PD9BG KG7EZ RRR",
                     "W1JET KE0HQZ 73", "WB3D OM4SX -16", "WA3ETR IZ2QGB RR73",
                     "BG THX JOE 73"]:
            pm = self.pack(msg)
            upm = r.unpack(pm)
            upm = re.sub(r'  *', ' ', upm)
            if msg != upm.strip():
                print "pack oops %s %s %s" % (msg, pm, upm)
        for bf in bfiles:
            wsa = bf[1].split("\n")
            for wsx in wsa:
                wsx = wsx.strip()
                m = re.search(r'# (.*)', wsx)
                if m != None:
                    msg = m.group(1)
                    pm = self.pack(msg)
                    upm = r.unpack(pm)
                    upm = re.sub(r'  *', ' ', upm)
                    if msg != upm.strip():
                        print "pack oops %s %s %s" % (msg, pm, upm)

    # call the Reed-Solomon encoder.
    # twelve is 12 6-bit symbol numbers (after packing).
    # returns 63 symbols.
    def rs_encode(self, twelve):
        int63 = c_int * 63
        int12 = c_int * 12
        
        tw = int12()
        for i in range(0, 12):
            tw[i] = twelve[i]

        out = int63()
  
        librs.rs_encode_(byref(tw), byref(out))
    
        a = [ ]
        for i in range(0, 63):
            a.append(out[i])

        return a

    def sync_hz(self, hza, sym):
        hz = hza[0] + (hza[1] - hza[0]) * (sym / 126.0)
        return hz

    # ba should be 126 symbols, each 0..66.
    # hza is [start,end] frequency of symbol 0,
    #   as from guess_freq().
    # spacing is inter-symbol frequency spacing.
    # returns an array of audio samples at 11025.
    # if rate is 11025, symsamples should be 4096.
    def fsk(self, ba, hza, spacing, rate, symsamples):
        # the frequency needed at each sample.
        hzv = numpy.array([])
        for bi in range(0, len(ba)):
            base = self.sync_hz(hza, bi)
            fr = base + (ba[bi] * spacing)
            block = numpy.repeat(fr, symsamples)
            hzv = numpy.append(hzv, block)

        # cumulative angle.
        angles = numpy.cumsum(2.0 * math.pi / (float(rate) / hzv))

        a = numpy.sin(angles)

        return a

    # twelve[] is 12 6-bit symbols to send.
    # returns an array of 126 symbol numbers, each 0..66,
    # including sync tones.
    def symbols(self, twelve):
        # Reed-Solomon -> 63 symbols
        enc = self.rs_encode(twelve)
        
        # interleave 
        inter = [ 0 ] * 63
        for c in range(0, 7):
            for r in range(0, 9):
                inter[(c*9)+r] = enc[(r*7)+c]
        
        # gray-code
        gray = [ bin2gray(x, 6) for x in inter ]

        # sync pattern -> 126 "symbols", each 0..66
        synced = [ 0 ] * 126
        i = 0
        for j in range(0, 126):
            if pattern[j] == 1:
                synced[j] = 0
            else:
                synced[j] = gray[i] + 2
                i += 1

        return synced

    # twelve[] is 12 6-bit symbols to send.
    # tone is Hz of sync tone.
    # returns an array of audio samples.
    def send12(self, twelve, tone, rate, symsamples):
        synced = self.symbols(twelve)

        samples = self.fsk(synced, [tone, tone], 2.6918, rate, symsamples)

        return samples

    def testsend(self):
        random.seed(0) # XXX determinism
        
        # G3LTF DL9KR JO40
        x1 = self.send12([61, 37, 30, 28, 9, 27, 61, 58, 26, 3, 49, 16], 1000, 11025, 4096)
        x1 = numpy.concatenate(([0]*1,  x1, [0]*(8192-1) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x1 = x1 * rv

        # RA3Y VE3NLS 73
        x2 = self.send12([46, 6, 32, 22, 55, 20, 11, 32, 53, 23, 59, 16], 1050, 11025, 4096)
        x2 = numpy.concatenate(([0]*4096,  x2, [0]*(8192-4096) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x2 = x2 * rv

        # CQ DL7ACA JO40
        x3 = self.send12([62, 32, 32, 49, 37, 27, 59, 2, 30, 19, 49, 16], 1100, 11025, 4096)
        x3 = numpy.concatenate(([0]*5120,  x3, [0]*(8192-5120) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x3 = x3 * rv

        # VA3UG   F1HMR 73  
        x4 = self.send12([52, 54, 60, 12, 55, 54, 7, 19, 2, 23, 59, 16], 1150, 11025, 4096)
        x4 = numpy.concatenate(([0]*1,  x4, [0]*(8192-1) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x4 = x4 * rv

        x = 3*x1 + 2*x2 + 1.0*x3 + 0.5*x4

        x += numpy.random.rand(len(x)) * 1.0
        x *= 1000.0

        x = numpy.append(x, [0]*(12*11025))

        r = JT65()
        r.cardrate = 11025.0
        r.gotsamples(x)
        r.process(self.samples)

    def send(self, msg):
        self.testsend()

def usage():
  sys.stderr.write("Usage: jt65.py -in CARD:CHAN [-center xxx]\n")
  sys.stderr.write("       jt65.py -file fff [-center xxx] [-chan xxx]\n")
  sys.stderr.write("       jt65.py -bench\n")
  sys.stderr.write("       jt65.py -send msg\n")
  # list sound cards
  weakaudio.usage()
  sys.exit(1)

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

if False:
  r = JT65()
  print r.unpack([61, 37, 30, 28, 9, 27, 61, 58, 26, 3, 49, 16]) # G3LTF DL9KR JO40
  print r.unpack([61, 37, 30, 28, 5, 27, 61, 58, 26, 3, 49, 16]) # G3LTE DL9KR JO40
  print r.unpack([61, 37, 30, 28, 9, 27, 61, 58, 26, 3, 49, 17]) # G3LTF DL9KR JO41
  sys.exit(0)

if False:
  r = JT65()
  # G3LTF DL9KR JO40
  print r.process2([
    14, 16, 9, 18, 4, 60, 41, 18, 22, 63, 43, 5, 30, 13, 15, 9, 25, 35, 50, 21, 0,
    36, 17, 42, 33, 35, 39, 22, 25, 39, 46, 3, 47, 39, 55, 23, 61, 25, 58, 47, 16, 38,
    39, 17, 2, 36, 4, 56, 5, 16, 15, 55, 18, 41, 7, 26, 51, 17, 18, 49, 10, 13, 24
    ], None)
  sys.exit(0)

if False:
  s = JT65Send()

  # G3LTF DL9KR JO40
  x = s.send12([61, 37, 30, 28, 9, 27, 61, 58, 26, 3, 49, 16], 1000, 11025, 4096)

  # inject some bad symbols
  # note x[] has sync in it.
  # 1 2 5 6 7 14 16 18 19 20
  n = 28
  for pi in range(0, len(pattern)):
      if pattern[pi] < 0 and n > 0:
          #x[si*4096:(si+1)*4096] = numpy.random.random(4096)
          x = numpy.concatenate((x[0:pi*4096], numpy.random.random(4096), x[(pi+1)*4096:]))
          n -= 1
  
  r = JT65()
  r.cardrate = 11025.0
  r.verbose = True
  r.gotsamples(x)
  r.process(r.samples, 0)

  sys.exit(0)

# what wsjt-x says is in each benchmark wav file.
# files in jt65files/*
bfiles = [
    [ False, "jt65-1.wav", """
      0065 -20  0.3  718 # VE6WQ SQ2NIJ -14
      0065  -6  0.3  815 # KK4DSD W7VP -16
      0065 -10  0.5  975 # CQ DL7ACA JO40
      0065  -8  0.8 1089 # N2SU W0JMW R-14
      0065 -11  0.8 1259 # YV6BFE F6GUU R-08
      0065  -9  1.7 1471 # VA3UG F1HMR 73
      0065  -1  0.6 1718 # BG THX JOE 73""" ],
    [ False, "j5.wav", """
      0000  -1  0.4  790 # CQ N8DEA/QRP EN91
      0000  -7 -0.9 1259 # KM2S YV6BFE -11
      0000  -9  0.9 1497 # CQ YV5DRN FK60""" ],
    [ False, "j6.wav", """
      0000 -26  1.2  610 # CQ VK2DX QF56
      0000  -1  1.2 1020 # FK8HN W8TIC R-21
      0000 -16  1.4 1358 # LU2DO KA5JTM EL29""" ],
    [ True, "j7.wav", """
      0000 -21  1.7  611 # VK2DX KD2FSI FN20
      0000  -1  1.6  790 # CQ N8DEA/QRP EN91
      0000 -10  0.2 1258 # KM2S YV6BFE RR73
      0000  -9  2.1 1496 # VK6YTS YV5DRN -25""" ],
    [ True, "j8.wav", """
      0000 -24  2.1  611 # KD2FSI VK2DX -18
      0000  -1  2.1 1020 # FK8HN W8TIC 73
      0000 -26  1.7 1259 # YV6BFE RRTU73
      0000 -17  1.8 1358 # LU2DO KA5JTM R-13
      0000  -2  1.7 1711 # JQ1HDR KE4ZUN -19""" ],
    [ False, "160421_1919.wav", """
      1919  -1 -0.2  634 # S59GCD K9BCT EL96
      1919 -22 -0.3  956 # WA3ETR K3JZ RR73
      1919  -4 -0.3 1339 # SP3JHZ N4HYK EL87
      1919  -6 -0.9 1563 # PY2RJ ES6DO -09
      1919  -8 -1.2 1754 # IT9SUQ PD0RWL JO21""" ],
    [ False, "160421_1921.wav", """
      1921  -9  0.2  290 # K9BCT PA1JT JO21
      1921 -10 -0.8  366 # CQ PD9BG JO21
      1921 -13 -0.1  422 # SM2EKA 9A3BDE JN74
      1921 -14  0.5  634 # S59GCD N8CWU R-09
      1921  -3 -0.2  993 # CQ RJ3AA KO85
      1921  -1  0.0 1340 # SP3JHZ N4HYK EL87
      1921  -2 -0.3 1562 # PY2RJ ES6DO -09
      1921 -18 -0.9 1753 # IT9SUQ PD0RWL R-01""" ],
    [ False, "160421_1922.wav", """
      1922 -20  0.1  699 # CQ W7DAU CN84
      1922 -17 -0.3 1363 # CQ DX DK9JC JN39
      1922  -1 -0.3 1611 # EA1HNY K1JSN 73
      1922  -2 -0.7 1752 # PD0RWL IT9SUQ 73""" ],
    [ False, "160421_1923.wav", """
      1923 -12 -0.6  195 # CQ KB4QLL EL98
      1923  -5  0.2  290 # CQ PA1JT JO21
      1923 -12 -1.2  422 # SM2EKA K9GVM EN61
      1923 -13  0.5  634 # S59GCD N8CWU R-09
      1923  -4 -1.0  698 # W7DAU PD9BG JO21
      1923 -15 -0.7  874 # R2DHC EA1BPA IN73
      1923  -5 -0.3  993 # CQ RJ3AA KO85
      1923 -19  0.5 1191 # CQ F5NAA JN39
      1923  -1 -0.0 1340 # SP3JHZ N4HYK -10
      1923 -11 -0.4 1618 # CQ ES6DO KO27""" ],
    # ...
    [ False, "160421_1958.wav", """
1958 -11  2.7 1491 # CQ PD5HW JO22
1958  -1 -0.2  283 # W0OS OE1SZW JN88
1958 -10 -0.1  830 # CQ IZ8GUU JN70
1958  -1 -0.4  973 # RJ3AA W5THT R-13
1958 -16  0.2 1366 # CQ IT9SUQ JM77
1958  -1 -0.2 1811 # CQ KK4RDI EM90""" ],
    [ False, "160421_1959.wav", """
1959 -13  0.4  287 # CQ PA1JT JO21
1959 -10 -0.8  508 # CQ K9GVM EN61
1959  -5 -0.2  592 # CQ WA2DX FN42
1959 -14 -0.2  830 # IZ8GUU AI4DD EL87
1959 -13 -0.3  972 # W5THT RJ3AA RR73
1959 -14 -0.2 1182 # SQ9BZN EA3EJQ R-10
1959 -11 -0.1 1811 # KK4RDI NV0O EM28""" ],
    [ False, "160421_2020.wav", """
2020  -9 -0.2  303 # WA3ETR IZ2QGB RR73
2020  -7 -0.0  831 # WB8VGE IZ8GUU -17
2020 -13 -1.7  923 # PU5MDD ON6UF 73
2020  -7 -0.0 1024 # CQ IT9SUQ JM77
2020 -12 -0.7 1234 # AI4DD K9GVM R-01
2020  -1 -0.2 1332 # F5NAA KK4RDI R-20
2020  -4 -0.2 1621 # OM4SX WB3D EL87""" ],
    [ True, "160421_2021.wav", """
2021 -14  0.3  694 # CQ IW1FOO JN35
2021  -2  0.7 1031 # IT9SUQ KK4YEL EL98
2021  -2 -0.2 1291 # PY1JD R73 OBR
2021  -5 -0.5 1394 # CQ F5NAA JN39
2021 -20 -0.2 1626 # WB3D OM4SX -16
2021 -16 -0.1 1823 # CQ EA3EJQ JN11""" ],
    [ True, "160421_2040.wav", """
2040  -7 -0.2  350 # KE0HQZ RN2A R-19
2040  -7  0.7 1056 # CQ IT9SUQ JM77
2040  -1  0.1 1197 # CQ K9WZB DM24
2040 -21  0.1 1410 # EB5AG PY7VI 73
2040  -2  0.3 1589 # HF9D AM1MDC -15
2040  -7 -0.1 1659 # GI4SZW K4WQ EL98""" ],
    [ False, "160421_2041.wav", """
2041  -7 -0.2  350 # RN2A KE0HQZ -20
2041 -16 -0.0 1056 # IT9SUQ K3JZ CN87
2041  -5 -0.1 1198 # K9WZB W1HIJ DM14
2041  -8 -0.2 1408 # PY7VI EB5AG 73
2041  -8 -0.1 1589 # AM1MDC HF9D R-06
2041  -1 -0.2 1683 # PY7VI RJ3AA KO85""" ],
    [ False, "160421_2115.wav", """
2115 -14 -0.1 1491 # PD5HW OE3UKW -18
2115  -6 -0.3  308 # W1JET KE0HQZ 73
2115  -6 -1.6  434 # CQ OM3CUP JN88
2115  -9 -0.4  622 # YV5MBI F4BQS -02
2115  -1 -1.5 1230 # LZ1VDK KB5IKR EM70
2115 -11  0.9 1460 # 2E0CIN EA1EAS RR73""" ],
    [ False, "160421_2116.wav", """
2116  -4 -1.2  433 # OM3CUP IW2DIW JN45
2116 -11 -0.1 1246 # K9ZJ PY5VC GG46
2116  -6 -1.7 1436 # CQ ON6UF JO10
2116 -16  0.0 1723 # GK7KFQ PY7VI -10""" ],
    [ False, "160421_2203.wav", """
2203 -10  0.8  302 # KK4BMV F1LII -05
2203  -1 -0.1  645 # M0TRJ W9MDB EM49
2203  -8  0.2 1019 # KE0EFX IW2DIW R-14
2203 -15 -0.1 1303 # VE2SCA EA5KL R-10
2203  -1 -0.1 1611 # CQ K1JSN EM63
2203  -1  0.8 1828 # KK2WW WW5TT -08""" ],
    [ False, "160421_2204.wav", """
2204  -9  0.1  645 # CQ M0TRJ JO02
2204  -7 -0.1  851 # PY6JB GK4KAW IO70
2204  -3 -0.1 1018 # IW2DIW KE0EFX R-15""" ],
    [ False, "160421_2230.wav", """
2230  -1  0.5 1496 # CQ WW5TT EM25
2230 -15 -1.0  328 # KG7EZ PD9BG R-15
2230 -19  0.0  574 # KD2EIP KG7NXU RRR
2230  -3 -0.2  975 # W4DRK KA4RSZ EM73
2230 -14  1.6 1271 # CQ KA0RGT DM79""" ],
    [ False, "160421_2231.wav", """
2231  -5  0.3 1494 # WW5TT KA5JTM EL29
2231 -17 -0.2  329 # PD9BG KG7EZ RRR
2231 -11 -0.2  367 # CQ KE0HQZ EN12
2231  -4  0.2  975 # KA4RSZ W4DRK -09
2231  -3 -0.0 1271 # KA0RGT W9BS EL96""" ],
    [ False, "160421_2351.wav", """
2351  -9 -0.1  278 # KE0HQZ K6BRN DM03
2351  -6  0.3  635 # WB2LPC CO2VE -19
2351 -12 -0.1 1202 # CQ N5OSK EM25
2351 -16 -0.1 1554 # KD5RBW TU 73
2351  -1  1.5 1697 # K9VER WD8LJP -05""" ],
    [ False, "160421_2352.wav", """
2352 -10 -0.2  279 # K6BRN KE0HQZ -01
2352  -1 -0.1  478 # KD2CNC N4UQM R-01
2352  -6  0.1  893 # KD6HWI PY7VI R-12
2352  -1 -1.2 1696 # CQ DX CO3HMR EL82""" ],
[ False, "160422_0001.wav", """
0001 -22 -0.2  279 # KE0HQZ AF7HL R-01
0001 -13  0.2  409 # AA1XQ K9VER R-11
0001  -2  0.1  700 # CQ CO2VE EL83
0001 -16 -0.3 1202 # CQ N5OSK EM25
0001 -10 -0.1 1385 # KD2INN K6BRN 73
0001 -10  0.2 1631 # KD5RBW W6DPD R-07
0001  -1  1.5 1980 # VK2LAW WD8LJP 73
""" ],
[ False, "160422_0002.wav", """
0002 -11 -0.3  279 # RR73 25W VERT
0002  -1 -0.1  485 # KD2CNC K4PDS EM75
0002 -15 -0.2  701 # CQ K1RI FN41
0002 -22 -0.0 1202 # N5OSK VA3TX EN94
0002  -1 -1.2 1592 # WA3ETR CO3HMR RR73
0002 -21  0.3 1982 # WD8LJP VK2LAW 73
""" ],
[ False, "160422_0004.wav", """
0004  -7 -0.3  279 # CQ KE0HQZ EN12
0004 -13  0.0  319 # CQ DX PQ8VA GJ40
0004  -1 -0.1  484 # KD2CNC K4PDS R-01
0004  -5  0.1 1455 # W8NIL VE3KRP -05
0004  -1 -1.3 1592 # CQ DX CO3HMR EL82
""" ],
[ False, "160422_0005.wav", """
0005 -17 -0.7  277 # KE0HQZ N7NON DM43
0005 -18 -0.3  318 # PQ8VA W6OPQ CM97
0005  -3  0.2  701 # KC6WFS CO2VE -11
0005  -7 -0.1 1034 # N1ARE K6BRN R-12
0005 -10 -0.1 1202 # RR73 EQSLOTW
0005 -18 -0.4 1260 # TI4DJ KD2HRD FN32
0005  -1  1.4 1876 # CQ WD8LJP DM65
""" ],
[ False, "160422_0012.wav", """
0012 -12 -0.3  278 # K4PDS KE0HQZ -03
0012  -1  0.3 1088 # KD9CHO KA5JTM EL29
0012  -1 -1.3 1591 # WA2DIY CO3HMR R-07
0012 -18 -0.1 1800 # KC3FL N7DTP R-21
0012 -17  0.3 1974 # CQ VK2LAW QF56
0012 -13  0.0 2013 # KB3OZC TU 73
""" ],
[ False, "160422_0013.wav", """
0013  -2 -0.1  283 # KE0HQZ K4PDS R-01
0013 -17 -0.1  471 # CQ VE4DPR EN19
0013  -3 -0.1  700 # K1RI K6BRN DM03
0013 -14 -0.2  983 # AC0LP WI0E EN34
0013 -11 -0.0 1202 # CQ N5OSK EM25
0013  -5  1.5 1259 # CQ WD8LJP DM65
0013 -14  0.3 1799 # N7DTP KC3FL RR73
0013 -20 -0.2 2012 # CQ KB3OZC FN20
""" ],
[ False, "160422_0023.wav", """
0023 -12 -0.2  279 # KE0HQZ N5PT EM15
0023 -13 -0.0  469 # N5VX VE4DPR -05
0023  -1 -0.1  716 # TF2MSN W5FDB R-10
0023 -13 -0.1  950 # KD2CNC K6BRN R-18
0023 -13 -0.2  984 # CQ N5OSK EM25
0023 -14  0.0 1049 # CQ VE6RMB DO21
0023 -19 -0.0 1257 # TI4DJ VA3TX EN94
0023  -2  0.2 1426 # K9VER W7UT DM37
0023  -1 -0.1 1973 # VK2LAW N4MRM EM78
""" ],
[ False, "160422_0024.wav", """
0024  -1 -0.2  470 # VE4DPR N5VX R-03
0024 -10 -0.2  716 # W5FDB TF2MSN 73
0024  -8 -1.8 1146 # CQ YV5FRD FK60
0024  -2  0.2 1256 # CQ TI4DJ EK70
0024 -10  0.2 1425 # W7UT K9VER -08
0024 -16 -0.1 1664 # CQ WA2HIP FN54
""" ],
[ False, "160422_0030.wav", """
0030  -5 -0.1  834 # JR1AQN K4PDS EM75
0030 -15 -0.1  469 # VE4DPR AC2PB R-12
0030  -5  0.0 1049 # VE6RMB K3JZ CN87
0030  -6  0.2 1254 # VA3TX TI4DJ RRR
0030 -12  0.9 1868 # N4MRM WD5BFH 73
""" ],
[ False, "160422_0031.wav", """
0031 -19  1.4  471 # 5W TU 73
0031  -8 -0.2  987 # VA3RTX N5OSK -01
0031  -1 -0.1 1641 # KD2INN N4WXB R-10
0031  -1 -0.1 1664 # CQ N1DAY EM85
0031  -1  0.9 1870 # WD5BFH N4MRM 73
0031  -6  0.1 2152 # AA1XQ KA5JTM R-14
""" ],
[ False, "160422_0034.wav", """
0034  -4 -0.1  833 # JR1AQN K4PDS R-08
0034 -13  1.0 1050 # VE6RMB WB1ABQ R-05
0034 -16  0.3 1253 # CQ TI4DJ EK70
0034  -8  0.9 1372 # CQ K0JY DM68
0034 -15 -0.2 2152 # AA1XQ K5RHD DM65
""" ],
[ True, "160422_0035.wav", """
0035 -11 -0.3  986 # RR73 EQSLOTW
0035 -17 -0.0 1049 # WB1ABQ RRR 73
0035  -1 -0.2 1254 # TI4DJ N1DAY EM85
0035  -1  0.0 1854 # K6LER N4MRM -14
0035 -24 -0.2 2152 # K5RHD AA1XQ -04
""" ],
[ False, "160422_0036.wav", """
0036  -2 -0.2  833 # QRZ+EQSL 73
0036 -13  1.0 1049 # VE6RMB WB1ABQ 73
0036 -19  0.2 1252 # CQ TI4DJ EK70
0036 -10  0.3 1372 # VA3TX K0JY -07
0036  -6 -0.2 2151 # AA1XQ K5RHD R-12
""" ],
[ False, "160422_0116.wav", """
0116  -1 -0.2  291 # UN6TA K4PDS EM75
0116 -20  0.3  981 # K7CAH NP4JV RR73
0116  -1  0.4 1859 # KV4PC N4MRM RR73
""" ],
[ False, "160422_0117.wav", """
0117 -18  0.3  510 # CQ K6LER CM95
0117  -1 -0.3  750 # KD2INN K2MOB R-05
0117 -17  0.2  983 # NP4JV K7CAH 73
0117  -1 -0.0 1244 # TI4DJ K4KFN EM75
0117  -1  0.3 1858 # TU DONALD 73
""" ],
[ False, "160422_0121.wav", """
0121 -15  0.2  981 # KM4OVT K7CAH -18
0121 -10  0.8 1228 # CQ WB9VGJ DM34
0121  -3 -0.2 1558 # UN7FU PY7VI HI21
0121  -1  0.1 1857 # N4MRM K4KFN R-16
""" ],
[ False, "160422_0122.wav", """
0122  -1 -0.3  508 # K6LER N1DAY 73
0122  -1  0.4 1859 # K4KFN N4MRM RRR
""" ],
[ False, "160422_0128.wav", """
0128 -17  0.2  596 # CQ KB7RAF CM98
0128 -23 -0.2  786 # WA3ETR W1FIT -06
0128  -3 -0.2 1184 # CQ W9BBF EN54
0128  -1 -0.2 1570 # CQ PY7VI HI21
0128 -20  0.4 1858 # CQ N4MRM EM78
""" ],
[ False, "160422_0129.wav", """
0129 -16 -0.0  309 # CQ WB9VGJ DM34
0129  -1  0.0  596 # KB7RAF K4KFN EM75
0129 -10  0.3  747 # KD2INN N6RBW -09
0129  -8 -0.3 1188 # W9BBF K2MOB EM60
0129  -2 -0.3 1563 # UN7FU W5FDB EM40
""" ],
[ False, "160422_0134.wav", """
0134 -15 -0.2  842 # N7LVS NQ6F DM12
0134 -18  0.2  562 # CQ NP4JV DM41
0134  -8 -0.2 1199 # K2MOB W9BBF 73
0134  -1 -0.2 1564 # K4KFN PY7VI R-10
0134 -11 -0.1 1858 # NEDED WV TNX
0134 -13  0.3 2173 # UN7FU KG5TED EM30
""" ],
[ False, "160422_0135.wav", """
0135 -19 -0.0  305 # K4ARE WB9VGJ -12
0135  -5 -0.3  561 # NP4JV W5FDB EM40
0135  -4  0.2  962 # K8STS N6RBW -11
0135  -1 -0.1 1565 # PY7VI K4KFN RRR
0135 -18  1.1 2170 # KG5TED UN7FU -14
""" ],
[ False, "160424_1146.wav", """
1146  -2  0.1  576 # VA3WLD W5ARX FM16
1146  -6  0.3  868 # N9BUB QSO B4
1146 -17 -0.1 1265 # CQ KG5INX EM20
1146  -4 -0.5 1401 # KC2PQ KA8YYY 73
1146  -8  0.5 1823 # AG6RS KB3OZC -11
""" ],
[ False, "160424_1147.wav", """
1147  -1 -0.5  346 # CQ KD8ZEF FN13
1147  -1 -0.9  575 # W5ARX VA3WLD R-15
1147  -2 -0.9 1402 # KA8YYY KC2PQ 73
1147 -15 -1.7 1829 # KB3OZC AG6RS R-12
""" ],
[ False, "160424_1230.wav", """
1230  -4 -1.3  873 # KD8ZEF K3JSE R-05
1230 -22 -0.1 1238 # K3VAT KG5INX -15
1230  -1  0.9 1639 # WB8FVB KI8DU EM88
1230  -1  0.2 1824 # CQ KB3OZC FN20
1230  -1 -0.8 2113 # CQ VE1JBC FN73
""" ],
[ False, "160424_1231.wav", """
1231 -19 -0.2  604 # CQ XE2YWH DL92
1231  -6 -0.7  873 # K3JSE KD8ZEF RR73
1231  -1  0.2 1239 # KG5INX K3VAT R-06
1231 -19 -0.0 1639 # KI8DU WB8FVB -05
""" ],
[ True, "160424_1314.wav", """
1314  -1  0.1  296 # AE7JP WB2SMK FN32
1314 -19  0.0  504 # CQ AA1XQ FN41
1314  -6  0.1 1686 # DE N8DEA/QRP 73
1314  -1 -0.1 1995 # CQ VA3WLD FN03
""" ],
[ False, "160424_1315.wav", """
1315  -1  0.5  508 # AA1XQ N3GGT FN21
1315 -23  0.0  792 # VE5TLW W9MDB -07
1315 -18  0.0 1995 # VA3WLD KC0DE EL96
""" ],
[ False, "160424_1438.wav", """
1438 -19  0.1  561 # CQ N1PBC FN42
1438 -15  0.0 1177 # AK4FC AJ4TF 73
1438  -4 -0.0 1583 # N8DEA N8JK R-07
""" ],
[ False, "160424_1439.wav", """
1439  -5  0.1 1585 # DE N8DEA/QRP 73
""" ],
[ False, "160424_1556.wav", """
1556 -15 -1.0  382 # RA9FGW IU2EWY -10
1556  -6 -1.2  649 # KB0PPQ K5KNM 73
1556 -18 -1.6 1078 # CQ HA0ML KN17
1556  -9 -0.8 1321 # IZ1KGY 9A0W -06
1556  -2 -0.9 1517 # KB3JSV AB4QS R-15
1556  -3 -2.9 1588 # KJ4TX N9PBD EM58
1556  -1 -1.1 1768 # CQ KK4RDI EM90
1556 -24 -1.1 2062 # EM30UT MI0TBV
""" ],
[ False, "160424_1557.wav", """
1557  -2  1.3  650 # K5KNM KB0PPQ 73
1557  -4 -0.1  736 # KB0DNP W4LVH FM02
1557 -13  0.1  938 # K1GG W7UT DM37
1557 -14 -0.3 1145 # VU2TS R3NA LO07
1557 -13  0.2 1320 # 9A0W IZ1KGY R-06
1557  -1  0.6 1774 # YV5ARM K4YYL EM84
1557 -10 -0.1 2061 # CQ EM30UT KO4
""" ],
[ False, "160424_1611.wav", """
1611 -11 -0.1  780 # YC8RBI OE3UKW -19
1611  -1 -0.0  929 # CQ N9MUF EN51
""" ],
[ False, "160424_1612.wav", """
1612 -14 -0.1  980 # CQ AM1MDC IN52
1612 -16  1.5 1770 # HB9OBX IT9SUQ R-13
1612 -20  0.9 1965 # CQ 9A6T JN75
""" ],
[ True, "160424_1638.wav", """
1638 -12 -0.5  500 # IU2EWY R3NA 73
1638 -13  0.6  777 # CQ YL3ID KO07
1638 -14  0.3  956 # IU2EWY SQ9OUM -11
1638 -12 -0.1 1253 # CQ KW4HQ EM74
1638 -19  0.7 1357 # F6GIG 9A6T -11
1638 -17 -0.2 1540 # VE4RK WA4UT EM66
1638 -11 -0.2 1624 # WY1C F5BWS RRR
1638  -9  0.3 1731 # DL9SAD IT9SUQ 73
1638 -16  0.8 1801 # ZS6AI UR5WET KN19
1638 -13  0.4 1878 # RD3FC AM1MDC -21
1638  -1  1.2 2152 # N5RLM N4MRM -04
""" ],
[ False, "160424_1639.wav", """
1639 -16 -0.2  486 # KW4CR N6YG RRR
1639  -1 -0.2  778 # YL3ID K1JSN EM63
1639 -10  0.3 1159 # CQ DX IK0XFD JN61
1639 -21  0.1 1338 # EA3AMP RD3FC KO95
1639  -1 -0.2 1624 # F5BWS WY1C 73
1639 -11  0.4 2155 # N4MRM N5RLM R-09
1639 -15  0.2 2257 # WA1PCY W0OS -10
""" ],
[ False, "160424_1640.wav", """
1640 -21  0.4  373 # CQ PE1PNO JO21
1640 -17  0.0  551 # RN4ABD SV6COH KM09
1640  -3  0.8  607 # CQ W4DRK EL95
1640 -10  0.7  774 # VE3TKB YL3ID -19
1640 -12 -0.1  969 # CQ KK9G EN61
1640 -16 -0.2 1361 # WY1C K7ADD CN97
1640  -3 -0.3 1539 # VE4RK WA4UT EM66
1640  -9 -0.2 1623 # WY1C F5BWS 73
1640 -11  0.3 1729 # CQ IT9SUQ JM77
1640 -11  0.1 1911 # CQ K4PPY FM03
1640  -1  1.2 2152 # N5RLM N4MRM RR73
""" ],
[ False, "160424_1656.wav", """
1656  -5 -0.2  283 # WB9VGJ KE0EFX R-06
1656  -3  1.3  547 # CQ KB0PPQ EM29
1656  -3 -0.2 1161 # DL2IAU WS5L EM12
1656 -13 -0.4 1212 # N5RLM KE0HQZ -02
1656 -18 -0.1 1403 # CQ DG2PHE JO51
1656 -18 -0.6 1568 # CQ EA7FUW IM76
1656  -9  0.3 1729 # CQ IT9SUQ JM77
1656  -1  0.4 2152 # DJ0QO N4MRM -09
1656  -7 -0.2 2478 # K8EAC F5BWS 73
""" ],
[ True, "160424_1657.wav", """
1657  -8  0.2  282 # KB7MYO W0OS EN16
1657 -19 -0.5  333 # EA3BJO EA1HMT 73
1657 -12  0.4  509 # YE6YE DG9AK JN49
1657 -10 -0.1  552 # N7LVS WT9WT RRR
1657 -21 -0.1  853 # UR3LC M1WPB JO01
1657  -4  0.7 1020 # VE4RK KC0FGX EN30
1657  -5  0.4 1212 # KE0HQZ N5RLM R-05
1657  -8  0.7 1243 # K6PVA KB0NRG EM27
1657 -18 -0.6 1573 # EA7FUW R3NA LO07
1657 -13  0.3 1910 # CQ K4PPY FM03
1657  -1 -0.2 2055 # MM0LGS K1JSN -20
1657  -3 -0.1 2152 # N4MRM DJ0QO JN39
""" ],
[ False, "160424_1725.wav", """
1725 -20 -0.2  406 # TU 73 FROM NV
1725 -10  0.7  492 # RA3QEH DL8ZBA R-08
1725  -2  1.3  583 # CQ KB0PPQ EM29
1725  -1 -0.2  738 # VA7REH KE0EFX R-15
1725 -14 -1.8 1260 # DK3XX KJ0B EN36
1725 -12 -0.1 1498 # SP5AN F5OJN -09
1725  -1 -0.2 1866 # W0OS N1CZZ -01
""" ],
[ False, "160424_1726.wav", """
1726 -18 -0.2  352 # CQ KK7X DN17
1726  -7 -0.2  579 # DL2IAU WS5L EM12
1726 -12  0.3  704 # HB9POU YL3ID -13
1726 -10 -0.1  971 # VE6UX KK9G RRR
1726 -16  0.8 1269 # CQ KP3IV FK68
1726 -14  7.0 1426 # CQ KB0DNP EN10
1726 -13  0.3 1625 # DL1MRD 2W0NAW -11
1726 -10  0.2 1761 # CQ IT9SUQ JM77
1726  -9  1.0 1865 # N1CZZ KA5JTM EL29
1726 -11  0.2 2092 # ZS6AI DL1BQR JO73
1726 -15  0.4 2290 # RA3TAT DC4DO R-15
1726  -5 -0.2 2481 # KM4NHW OE3UKW -20
""" ],
[ False, "160424_1727.wav", """
1727  -6 -0.1  492 # RA3QEH DL8ZBA RRR
1727  -4  1.3  583 # NE1I KB0PPQ -03
1727  -5 -0.2  738 # VA7REH KE0EFX 73
1727 -18 -0.1  988 # HA7CH M6POB IO83
1727  -8  0.2 1123 # KF5RSA KC0FGX EN30
1727  -9 -0.2 1269 # KP3IV DL3NOC JO63
1727  -8 -0.2 1430 # KB0DNP N4STV EL98
1727  -9 -0.1 1499 # SP5AN F5OJN RR73
1727  -2 -0.2 1866 # W0OS N1CZZ -01
""" ],
[ False, "160424_1728.wav", """
1728 -14 -0.2  352 # CQ KK7X DN17
1728  -6 -0.2  579 # DL2IAU WS5L EM12
1728 -11 -0.1  971 # CQ KK9G EN61
1728 -11  0.1  989 # M6POB HA7CH -01
1728 -16 -0.1 1210 # CQ 2W0NAW IO81
1728  -8  0.2 1761 # CQ IT9SUQ JM77
1728 -18  0.1 1906 # AF5ZJ K4PPY RRR
1728 -15  0.2 2095 # ZS6AI DL1BQR JO73
1728 -19  0.4 2290 # RA3TAT DC4DO 73
1728  -3 -0.2 2481 # KM4NHW OE3UKW -20
""" ],
[ True, "160424_1729.wav", """
1729 -12 -0.2  268 # ALC CHECK
1729 -11  0.3  349 # KK7X N0MEU DM79
1729 -10 -0.1  492 # RA3QEH DL8ZBA 73
1729  -4  1.3  582 # NE1I KB0PPQ RRR
1729  -8 -0.2  745 # CQ KW4FQ EM75
1729 -20 -0.2  988 # HA7CH M6POB R-01
1729  -4  0.1 1123 # KF5RSA KC0FGX R-11
1729  -2 -0.2 1269 # KP3IV DL3NOC R-17
1729  -1 -0.2 1430 # KB0DNP N4STV EL98
1729 -16  0.1 1625 # 2W0NAW DL1MRD 73
1729 -14 -0.3 1906 # K4PPY AF5ZJ 73
""" ],
[ False, "160424_1730.wav", """
1730  -5 -0.2  579 # DL2IAU WS5L EM12
1730  -7 -0.2  971 # CQ KK9G EN61
1730 -16  0.1  988 # M6POB HA7CH RRR
1730 -14 -2.1 1255 # CQ PD5JVD JO32
1730 -11  1.1 1426 # CQ KB0DNP EN10
1730  -6 -0.1 1502 # F5OJN KE0EFX EM28
1730  -4 -0.4 1762 # R9AT IT9SUQ R-17
1730 -13 -0.0 1866 # CQ SQ9OUM JO90
1730 -15  0.1 1906 # AF5ZJ K4PPY 73
1730 -17 -0.1 2486 # F5BWS N4THG EM86
""" ],
[ True, "160424_1826.wav", """
1826  -2 -0.3  317 # US7IS WA2DX FN42
1826 -25 -0.2  602 # SQ5WAJ M6RUG -01
1826 -25 -0.3  790 # 2D0YLX SP5AN KO02
1826 -15 -0.3  954 # CQ W4ORS EM64
1826  -2  0.2 1064 # KC2NEO W0NEO -20
1826  -9 -0.2 1107 # SA5REK G3MZV -20
1826  -9  0.5 1273 # AA2DP KP3IV RR73
1826  -1 -0.4 1519 # K3CWF KD0YTE EN30
1826 -18  0.3 1785 # 2D0YLX SQ9OUM 73
1826  -5  0.1 1907 # N5TF PA3AJH JO32
1826 -11 -0.3 2082 # CQ PD0ONJ JO22
1826 -22 -0.3 2358 # 2D0YLX KM2S FN20
""" ],
[ False, "160424_1827.wav", """
1827 -11  0.1  316 # CQ US7IS KN98
1827 -17 -0.2  456 # OH8MXJ EA3AMP JN01
1827 -11 -1.9  602 # M6RUG SQ5WAJ R-12
1827  -7  0.1  850 # RA6ATV DL1BQR 73
1827  -1  1.7 1109 # CQ N5TF EM50
1827  -6 -0.3 1517 # VK2POP DK4CF R-15
1827 -13 -0.1 1541 # CQ F5OJN JN05
1827 -18 -0.3 1957 # CQ OE6KLG JN77
""" ],
[ False, "160424_1909.wav", """
1909 -13  0.6  182 # 2D0YLX OZ1PGB
1909  -1 -0.1  350 # KK7X KW4HQ R-26
1909  -5  0.8  568 # CQ W0NEO EM36
1909  -5  0.5  791 # CQ DL8ZBA JN49
1909 -13 -0.5 1257 # CQ M1WPB JO01
1909 -20 -2.5 1331 # CQ EA8CDW IL28
1909 -11  1.8 1483 # 2D0YLX HA3LI JN96
1909 -20  0.3 1555 # CQ ON4CJU JO20
1909  -9 -1.0 1757 # OE6KLG PA2BT JO21
1909  -1 -1.5 1861 # MM0HVU W9RY EM57
1909  -4 -0.3 2153 # CQ AC9HP EM69
""" ],
[ True, "160424_1910.wav", """
1910 -21 -0.3  459 # CQ SQ3SKN JO72
1910 -19  0.1  568 # W0NEO N7RJN DM33
1910 -10  0.3  645 # CQ PA1JT JO21
1910  -8 -0.0  785 # VE7JH W8OU EM15
1910 -13  0.1  984 # N1PBC SP7SMF -12
1910 -19 -0.3 1119 # CQ NC7L DM33
1910  -5 -0.2 1256 # N6YG WP4PGY 73
1910 -10  0.1 1384 # YE6YE PD5MJF 73
1910  -1 -0.2 1530 # CQ K9EEH VY0
1910  -1 -0.3 1678 # CQ KD9EOT EN62
1910 -17 -0.6 1893 # KE5YTA 73
1910 -15  0.8 2307 # PSE MY RAPORT
""" ],
[ False, "160424_1952.wav", """
1952  -6  0.1  813 # VE7JH KD0JHZ EN41
1952  -1 -0.4  180 # GI7UGV RN2F -04
1952 -16 -0.2  394 # NO7P KC7EQL -17
1952  -6 -0.4  419 # CT1FBK VP9NO 73
1952 -10  0.2  478 # IW4EJK KC0FGX R-10
1952 -15 -0.6  605 # KK6ILV W7IWW DN55
1952  -1 -0.3  882 # M1WPB K0ZRK EM36
1952 -15  0.3 1226 # CQ DX XE2SIV DM22
1952 -12 -0.3 1249 # KB3X W7YLQ R-02
1952 -12 -0.6 1588 # CQ SQ9OUM JO90
1952  -2 -0.2 1615 # CQ KD9DHT EM69
1952  -1 -0.2 1938 # K4KFN IT9SUQ 73
1952 -16 -0.6 2288 # KE4ZUN K0GRC R-16
""" ],
[ False, "160424_1953.wav", """
1953  -5 -0.1  477 # KC0FGX IW4EJK 73
1953 -11 -0.3  628 # VP9NO HA3LI JN96
1953 -12 -0.4  882 # CQ M1WPB JO01
1953  -1 -0.7 1227 # XE2SIV W9RY EM57
1953  -2 -0.4 1481 # VE3TMT KB0NNV EM38
1953 -21  0.1 1717 # NC7L N5OHM 73
1953  -7 -1.0 1753 # PA3GPT PA2BT RR73
1953  -9 -0.4 2289 # K0GRC 73
""" ],
[ True, "160424_2036.wav", """
2036  -7 -1.0  810 # CQ M6DHV IO84
2036  -9 -0.2  363 # CQ SP8AWL KO11
2036 -14 -0.2  433 # CQ KC7EQL CN88
2036 -22 -0.8  644 # K9GVM K7QDX DM34
2036 -17 -0.3  972 # KD2HNS ON8ON R-09
2036 -12 -0.3 1039 # CQ IK1MDH JN35
2036 -22 -0.4 1260 # CQ LZ1UBO KN12
2036 -13 -0.6 1404 # K2DAR S55RD 73
2036  -3 -0.0 1663 # WD4GBW MM0HVU 73
2036  -7  1.6 1975 # M0NMC IT9SUQ 73
2036  -1 -0.8 2412 # WD8LJP KG4NMS 73
""" ],
[ False, "160424_2037.wav", """
2037  -1 -0.1  811 # M6DHV KB5IKR EM70
2037  -8  1.4  250 # VE4RK KB0PPQ 73
2037  -1 -0.8  643 # K7QDX K9GVM -16
2037  -6 -0.4 1039 # IK1MDH KB1CTC EM92
2037 -10 -0.3 1217 # LU1XP HA5FTL JN97
2037 -23 -0.3 1608 # CQ DX K3JZ CN87
2037  -3 -0.1 1970 # CQ KW4HQ EM74
2037  -1  0.8 2408 # KG4NMS WD8LJP 73
""" ],
[ True, "160424_2054.wav", """
2054  -8 -1.0  815 # VA3TX M6DHV RRR
2054 -16  0.8  453 # CQ M6MKF IO94
2054 -11 -0.2  553 # CQ IK3PQG JN55
2054  -1  1.6 1024 # EA1EI N5TF R-10
2054 -16 -0.2 1204 # WA2HIP HA5FTL 73
2054 -17  0.4 1320 # CO2BG DJ5CS JN49
2054  -7  0.8 1477 # CQ YV5FRD FK60
2054 -12 -0.8 1712 # KC3AK SP5FCZ 73
2054  -6  0.5 1945 # K9ZJ 9A3BDE JN74
2054  -8 -0.9 2029 # AGN PLS MM0HV
2054 -11 -0.5 2213 # LA3LUA EB5AG -06
2054 -19 -0.0 2378 # CQ EA1FFT IN73
""" ],
[ True, "160424_2055.wav", """
2055 -13 -0.9  829 # CQ K9GVM EN61
2055  -1  0.1  319 # KA1MR DF4WQ JN39
2055 -25 -0.4  688 # YV5FRD SP5XSD KO03
2055 -22 -0.3  758 # CQ DX K3JZ CN87
2055  -1 -0.1 1025 # CQ EA1EI IN63
2055  -1 -0.4 1322 # CO2BG WA4UT EM66
2055 -17 -0.3 1806 # CQ IK1MDH JN35
2055 -11 -0.1 1948 # KF7WNX K9ZJ R-01
2055 -16 -0.4 2037 # CQ DX DL7UHD JO62
""" ],
[ False, "160424_2056.wav", """
2056  -6 -1.0  815 # VA3TX M6DHV 73
2056 -12  0.8  453 # CQ M6MKF IO94
2056  -1  1.6  552 # ON8ON N5TF R-17
2056 -12 -0.9 1014 # WA3NSM MM0HVU IO85
2056  -9 -0.5 1209 # CQ PH7Y JO21
2056 -13 -0.8 1252 # CO6CG SP5FCZ KO02
2056 -15  2.2 1318 # IW2DIW CO2BG R-05
2056  -4  0.8 1477 # CQ YV5FRD FK60
2056 -18  0.0 1712 # KC3AK DL1MRD JO43
2056 -16 -0.4 2029 # VP9NO OE3UKW -14
2056 -11 -0.5 2211 # LA3LUA EB5AG 73
""" ],
[ False, "160424_2108.wav", """
2108 -14  1.7  352 # R3KF PB1HF -05
2108 -13 -0.5  565 # CQ G4TZX JO01
2108 -17 -0.5  946 # IU2EWY EI3CTB R-20
2108 -18 -1.8  985 # CQ KB1ESX FN42
2108 -16 -0.2 1139 # S56RGA EA3CH 73
2108  -9  0.0 1604 # PD5CVK R3KF KO91
2108 -18  0.0 1819 # CQ GM4ZET IO86
""" ],
[ False, "160424_2109.wav", """
2109 -10 -0.6  680 # CQ I1RJP JN45
2109  -1 -0.5  985 # KB1ESX N3DGE FN20
2109 -18 -0.1 1340 # CQ IT9KPE JM68
2109 -21 -0.4 1819 # GM4ZET DL1ZBB JO40
""" ],
[ False, "160424_2125.wav", """
2125  -5 -0.7  470 # VP9NO SQ6WZ R-15
2125  -6 -1.0  681 # CQ I1RJP JN45
2125  -9 -0.1  929 # AX2LAW CT1EKU IM58
2125 -11 -0.5 1106 # SP5FCZ IT9KPE JM68
2125 -13 -0.3 1140 # CQ S56RGA JN65
2125 -11 -0.8 1818 # GM4ZET LA2VRA R-04
2125 -11 -0.8 2010 # PB1HF GK4KAW RRR
""" ],
[ False, "160424_2126.wav", """
2126  -1 -0.5  466 # SQ6WZ VP9NO 73
2126 -17  0.2 1818 # LA2VRA GM4ZET RR73
""" ],
[ False, "160424_2232.wav", """
2232  -1 -0.6  876 # EI3CTB WX2H FN20
2232 -17 -0.4 1103 # CQ K1KEN EL87
2232  -1 -0.6 1376 # M0XAG IK2WSO JN45
2232  -5 -1.1 1504 # KA4RSZ K4HVF -08
2232 -18 -0.2 1694 # PF7M EA3HXB RR73
2232  -4 -0.2 1861 # CO6CG K9EEH EN51
2232  -1 -0.6 2074 # CQ DX K6EID EM73
2232 -16 -0.5 2318 # CQ SP5AN KO02
""" ],
[ False, "160424_2233.wav", """
2233 -18 -0.5  812 # CQ G1VIF IO93
2233  -1 -0.8  455 # PY2JEA I1RJP JN45
2233  -6 -0.2  982 # GK4KAW OE6KLG 73
2233 -16 -0.6 1152 # CQ W1FNB FN33
2233 -18 -0.3 1178 # ZP9MCE UN1L R-08
2233 -12 -0.8 1376 # CQ M0XAG IO83
2233  -7 -0.5 1505 # K4HVF KA4RSZ R-04
2233  -7 -0.7 1694 # CQ PF7M JO33
2233  -2 -0.5 1920 # CQ G3MZV IO81
""" ],
[ False, "160425_0000.wav", """
0000  -7 -1.8  388 # CQ K0NJR EN26
0000  -1 -0.7  629 # CQ K9JKM EN52
0000  -2 -0.3  969 # WX2H IK3PQG RR73
0000 -16 -0.6 1172 # W0NEO KD9EOT EN62
0000  -7 -3.2 1275 # UN1L AB1KW FN43
0000 -14  0.3 1615 # CQ SP6ECQ JO71
0000  -2 -0.4 1731 # KC1EDK KD9DHT -18
0000  -1 -0.5 1981 # K8KJG VE3RUV 73
0000  -1  0.5 2181 # CQ YV4DHS FK60
""" ],
[ False, "160425_0001.wav", """
0001  -1 -0.2  388 # K0NJR KD8BIN EN81
0001  -1 -0.9  614 # K9JKM I1RJP JN45
0001  -1 -0.7  968 # IK3PQG WX2H -08
0001 -10 -0.3 1173 # CQ W0NEO EM36
0001  -6 -0.4 1460 # CQ WB0ZYU EM29
0001  -1 -0.4 1731 # KD9DHT KC1EDK R-05
0001 -13 -0.1 1959 # CQ CA3ECM FF46
0001  -1 -1.4 2181 # YV4DHS K8KJG FM09
""" ],
[ False, "160425_0124.wav", """
0124  -1 -0.6  291 # YV6GM W0JMP EN34
0124  -4 -1.8  419 # CQ K0NJR EN26
0124  -9  0.4  585 # CQ NP4AM FK68
0124  -1 -0.6  902 # KC1EDK K0DMW -08
0124  -9 -0.4 1200 # CQ DX YV5KG FK60
0124 -10 -0.3 1403 # TU PAUL 73
0124 -18 -0.7 1597 # VE3NLS KD9EOT 73
0124  -4 -0.7 1809 # VE3RUV KO4LZ R-15
0124 -11  0.4 2096 # CQ IW0GBO JN61
""" ],
[ False, "160425_0125.wav", """
0125 -11 -0.5  902 # K0DMW KC1EDK R-05
0125  -1  0.1 1203 # SV1CIF KD5OSN 73
0125  -6 -0.6 1403 # KC3AK KM4LLF 73
0125  -1 -0.1 1598 # CQ VE3NLS FN04
0125  -2 -0.7 1810 # KO4LZ VE3RUV 73
0125  -1  0.3 2097 # IW0GBO KW4QY FM05
""" ],
[ False, "160425_0229.wav", """
0229 -20 -0.3  363 # PY2DPM K7EMI R-22
0229 -13  0.7  588 # UA9SHH UY2IS 73
0229  -8 -2.0  680 # NC7L KE5SV EM10
0229  -2 -0.1  774 # CQ YV5KAJ FK60
0229  -1 -0.6 1018 # KE7II NW9F -15
0229 -16 -0.1 1477 # VE3KAO PY2RJ GG66
0229  -1 -0.7 1778 # W0OS KF2T RRR
0229  -1 -1.1 2058 # PY7ZZ KG4NMS EM60
""" ],
[ True, "160425_0230.wav", """
0230  -9  1.0  364 # K7EMI RRR 73
0230  -4 -1.2  465 # WZ4K I1RJP JN45
0230 -12 -0.7  680 # KE5SV NC7L -03
0230 -17 -0.4  776 # YV5KAJ RM7L LN07
0230 -21 -0.7 1017 # NW9F KE7II R-10
0230  -8 -0.7 1268 # CQ OP4A JO21
0230  -1 -0.1 1481 # KG4NMS VE3KAO -17
0230 -10 -0.2 1715 # N8FKF N4NQY -20
0230  -4 -0.3 1777 # KF2T W0OS 73
0230  -9 -0.9 2055 # KG4NMS PY7ZZ -14
""" ],
[ False, "160425_0411.wav", """
0411 -25 -0.7  596 # KG7VGD W1FIT -06
0411 -19  0.1 1306 # CQ OE6ATD JN76
0411  -9 -0.1 1341 # KK4RDI SP2CDN R-20
0411  -9 -0.8 1701 # CQ N3BEN CM99
0411  -1 -1.3 2107 # WA2HIP IK5BCM JN53
0411  -1 -1.3 2331 # VA3MJR KG4NMS 73
""" ],
[ False, "160425_0412.wav", """
0412  -1 -1.4  395 # W5TT I1RJP JN45
0412  -9 -0.0  595 # W1FIT KB0DNP EN10
0412 -13 -0.8  859 # W9EO NC7L RRR
0412 -22 -0.8  990 # KB0DNP KQ2Z FN30
0412 -12 -0.7 1220 # UA3GDJ IZ3XJM -06
0412  -1  0.3 1307 # OE6ATD KA9HQE EN54
0412  -9 -0.8 1700 # N3BEN KK4RDI EM90
0412 -16 -0.8 2107 # CQ WA2HIP FN54
0412  -7 -0.7 2330 # KG4NMS EA3CFV -16
""" ],
[ False, "160425_0517.wav", """
0517  -2 -1.5  558 # CQ I1RJP JN45
0517  -6 -0.3  769 # CQ PY2DPM GG66
0517  -7 -0.9  793 # N6YFM K5VP CN85
0517  -6 -3.0 1048 # CQ IW2MYH JN45
0517  -7 -0.9 1849 # CQ K0FG EM38
0517  -8 -0.9 2058 # DL6TY W9HZ RRR
0517  -2 -1.1 2256 # F5BWS KF5YDG R-18
""" ],
[ False, "160425_0518.wav", """
0518 -11 -1.0  375 # CQ IU4APO JN54
0518  -4 -1.4  768 # PY2DPM IK2IWT JN55
0518 -18 -0.9 1050 # IW2MYH WA2HIP FN54
0518  -7  0.2 1365 # CQ IK8IJN JM78
0518  -1 -0.7 1634 # SV5AZK IT9FGA JM67
0518 -19 -0.9 2059 # RRR MITCH 73
0518 -12 -1.1 2256 # KF5YDG F5BWS RRR
""" ],
[ False, "160425_0620.wav", """
0620  -6 -1.6  431 # AX5AW I1RJP JN45
0620  -9 -0.9  774 # EA5WO KC7UBS -13
0620  -4 -1.4  970 # IZ0MIO KG4NMS -18
0620 -17 -1.0 1725 # PD0LK IU1BOW 73
0620  -1 -1.0 2169 # PY2DPM N5BCA EM12
0620 -13 -0.4 2324 # CQ PA2WCB JO21
""" ],
[ False, "160425_0621.wav", """
0621  -1 -0.6  775 # KC7UBS EA5WO IM99
0621 -17 -0.8  972 # KG4NMS IZ0MIO R-15
0621  -8 -2.5 1725 # IU1BOW PD0LK 73
0621  -5 -0.5 2169 # AE4NT PY2DPM -16
""" ],
[ False, "160425_0723.wav", """
0723 -11 -1.1  382 # FB QRP73
0723 -19 -0.9  596 # AX2VLT AX7RB -01
0723  -7  0.2  685 # CQ KT4DLB EM61
0723  -5 -0.1  847 # F4GCZ DJ6JJ -03
0723 -14  0.3 1761 # CQ PF3X JO21
0723  -8 -1.0 2076 # G8HXE KD6RF EM22
""" ],
[ False, "160425_0724.wav", """
0724 -12 -1.0  810 # N9TES KC7UBS RRR
0724  -6  0.5  685 # KT4DLB 9Y4NW FK90
0724 -20 -1.6 1157 # CQ IV3IIM JN65
0724  -5 -0.8 1695 # KO3F W6DLM CN87
0724 -10 -0.1 2076 # CQ G8HXE VK
""" ],
[ False, "160425_0820.wav", """
0820  -1 -1.4  863 # CE3RR KG4NMS EM60
0820 -20 -1.8 1098 # AE4NT AC8SM -08
0820  -1 -0.6 1741 # KH6NX W0HUR EN41
0820  -4 -1.0 2198 # JF1XUD WB5TOI -18
""" ],
[ False, "160425_0821.wav", """
0821 -11 -0.6  827 # F4GCZ PA2WCB JO21
0821  -7 -1.0 1098 # AC8SM AE4NT R-04
0821 -12 -1.0 1172 # KG4NMS GK4KAW IO70
0821  -9 -1.1 1741 # W0HUR KH6NX -14
""" ],
[ False, "160425_0838.wav", """
0838 -14 -1.0 1739 # KH6NX VK3FZ QF22
0838  -4 -1.0 2198 # JH1XVQ WB5TOI -09
""" ],
[ False, "160425_0839.wav", """
0839 -12 -1.1 1097 # CQ KC7UBS DN74
0839  -7 -1.1 1266 # CQ KH6NX BL11
0839  -1 -1.4 1711 # W6MRR KG4NMS 73
""" ],
[ False, "160425_0930.wav", """
0930  -4 -0.9  577 # WJ1B TU 73
0930  -2 -1.0  994 # WH6HI K8AJX R-10
0930  -8 -1.2 2103 # KG4NMS KA5JTM R-20
0930  -1 -0.7 2221 # VK6DW KG4NMS -20
""" ],
[ False, "160425_0931.wav", """
0931 -20 -0.9  578 # K5NJ WJ1B 73
0931 -11 -0.7  775 # AA1XQ WB6PVU CM87
0931  -9 -1.0  994 # ALOHA73 WH6HI
0931  -9 -1.2 1628 # W2GLH WB5GM EM10
""" ],
[ False, "160425_1055.wav", """
1055 -19 -1.0  781 # W0WKO VK2DX R-20
1055 -14 -0.2 1252 # AC8SM ZL3TE -10
1055  -6 -1.2 1592 # CQ AX3BL QF22
""" ],
[ False, "160425_1056.wav", """
1056 -17 -0.8  780 # VK2DX W0WKO RRR
1056  -1 -1.4 1251 # ZL3TE AC8SM R-20
1056  -1 -1.4 1592 # AX3BL NS9I EN64
1056  -1 -1.1 2367 # CQ DX KD4MDC EM78
""" ],
[ False, "160425_1122.wav", """
1122 -14 -1.5  913 # VE2GHI N5BCA EM12
1122  -5 -1.2  969 # JA1PLT KF4RWA EM63
1122  -3 -0.6 1253 # ZL3TE N5RLM R-10
1122 -19 -0.8 1600 # AX3BL KA5JTM EL29
1122  -1 -1.2 2237 # VK2DX K8EAC FM18
""" ],
]

def benchmark(verbose):
    global chan
    chan = 0
    score = 0 # how many we decoded
    wanted = 0 # how many wsjt-x decoded
    for bf in bfiles:
        if bf[0] == False:
            continue
        if verbose:
            print bf[1]
        wsa = bf[2].split("\n")

        filename = "jt65files/" + bf[1]
        r = JT65()
        r.verbose = False

        if True:
            # test r.hints[]
            hints = [ ]
            for wsx in wsa:
                wsx = wsx.strip()
                m = re.search(r' ([0-9]+) +# (.*)', wsx)
                if m != None:
                    hz = float(m.group(1))
                    txt = m.group(2)
                    txt = txt.strip()
                    txt = re.sub(r'  *', ' ', txt)
                    a = txt.split(' ')
                    while len(a) > 0 and (a[0] == 'CQ' or a[0] == 'DX'):
                        a = a[1:]
                    hints.append([ hz, a[0] ])
            r.hints = hints

        r.gowav(filename, chan)

        all = r.get_msgs()
        # each msg is [ minute, hz, msg, decode_time, nerrs, snr ]

        got = { } # did wsjt-x see this? indexed by msg.

        for wsx in wsa:
            wsx = wsx.strip()
            m = re.search(r'# (.*)', wsx)
            if m != None:
                wanted += 1
                wsmsg = m.group(1)
                wsmsg = wsmsg.replace(" ", "")
                found = False
                for x in all:
                    mymsg = x[2]
                    mymsg = mymsg.replace(" ", "")
                    if mymsg == wsmsg:
                        found = True
                        got[x[2]] = True
                if found:
                    score += 1
                    if verbose:
                        print "yes %s" % (m.group(1))
                else:
                    if verbose:
                        print "no %s" % (m.group(1))
                sys.stdout.flush()
        if True and verbose:
            for x in all:
                if x[4] < 25 and not (x[2] in got):
                    print "MISSING: %6.1f %d %.0f %s" % (x[1], x[4], x[5], x[2])
    if verbose:
        print "score %d of %d" % (score, wanted)
    return [ score, wanted ]

def smallmark():
    global budget, noffs, off_scores, pass1_frac, hetero_thresh, soft_iters

    o_budget = budget
    o_noffs = noffs
    o_off_scores = off_scores
    o_pass1_frac = pass1_frac
    o_hetero_thresh = hetero_thresh
    o_soft_iters = soft_iters

    for pass1_frac in [ .7, .8, 0.9, 1.0 ]:
        sc = benchmark(False)
        print "%d : pass1_frac=%.2f" % (sc[0], pass1_frac)
    pass1_frac = o_pass1_frac

    for soft_iters in [ 35, 50, 75, 100, 125, 150, 200, 250 ]:
        sc = benchmark(False)
        print "%d : soft_iters=%d" % (sc[0], soft_iters)
    soft_iters = o_soft_iters

    for off_scores in [ 3, 4, 5, 6, 7 ]:
        sc = benchmark(False)
        print "%d : off_scores=%d" % (sc[0], off_scores)
    off_scores = o_off_scores

    for hetero_thresh in [ 3, 5, 7, 9, 11 ]:
        sc = benchmark(False)
        print "%d : hetero_thresh=%d" % (sc[0], hetero_thresh)
    hetero_thresh = o_hetero_thresh

    for budget in [ 4, 5, 6, 7, 8]:
        sc = benchmark(False)
        print "%d : budget=%d" % (sc[0], budget)
    budget = o_budget

    for noffs in [ 3, 4, 5, 6 ]:
        sc = benchmark(False)
        print "%d : noffs=%d" % (sc[0], noffs)
    noffs = o_noffs

filename = None
desc = None
bench = None
small = None
send_msg = None

def main():
  global filename, desc, bench, send_msg, small
  i = 1
  while i < len(sys.argv):
    if sys.argv[i] == "-in":
      desc = sys.argv[i+1]
      i += 2
    elif sys.argv[i] == "-file":
      filename = sys.argv[i+1]
      i += 2
    elif sys.argv[i] == "-bench":
      bench = True
      i += 1
    elif sys.argv[i] == "-small":
      small = True
      i += 1
    elif sys.argv[i] == "-send":
      send_msg = sys.argv[i+1]
      i += 2
    else:
      usage()
  
  if send_msg != None:
      js = JT65Send()
      js.send(send_msg)
      sys.exit(0)
  
  if bench == True:
    benchmark(True)
    sys.exit(0)

  if small == True:
    smallmark()
    sys.exit(0)
  
  if filename != None and desc == None:
    r = JT65()
    r.verbose = True
    r.gowav(filename, 0)
  elif filename == None and desc != None:
    r = JT65()
    r.verbose = True
    r.opencard(desc)
    r.gocard()
  else:
    usage()

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
