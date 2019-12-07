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
step_frac = 2.5 # fraction of FFT bin for frequency search (4).
#ngoff = 3 # look at this many of guess_offset()'s results (6, 4).
goff_step = 64 # guess_offset search interval.
fano_limit = 30000 # how hard fano will work (10000).
fano_delta = 17
fano_bias = 0.5
fano_floor = 0.005
fano_scale = 4.5
statruns = 3
driftmax = 1.0 # look at drifts from -driftmax to +driftmax (2).
ndrift = 3 # number of drifts to try (including drift=0)
coarse_steps = 4 # coarse() search for start offset at this many points per symbol time.
coarse_hzsteps = 4 # look for signals at this many freq offsets per bin.
coarse_top1 = 2
coarse_top2 = 5
phase0_budget = 0.4  # fraction of remaining budget for pre-subtraction
subslop = 0.01 # granularity (in symbols) of subtraction phase search
start_slop = 4.0 # pad to this many seconds before 0:01
end_slop = 5.0 # pad this many seconds after end of nominal end time
band_order = 6 # order of bandpass filter
subgap = 0.4  # extra subtract()s this many hz on either side of main bin
ignore_thresh = -30 # ignore decodes with lower SNR than this

# information about one decoded signal.
class Decode:
    def __init__(self,
                 hza,
                 msg,
                 snr,
                 msgbits):
        self.hza = hza
        self.msg = msg
        self.snr = snr
        self.msgbits = msgbits # output of Fano decode
        self.minute = None
        self.start = None # sample number
        self.dt = None # dt in seconds
        self.decode_time = None # unix time of decode
        self.phase0 = False
        self.phase1 = False
        self.drift = self.hz() - hza[0] # Hz per minute

    def hz(self):
        return numpy.mean(self.hza)

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
    global fano_limit, fano_delta

    #for i in range(0, len(in0)):
    #    print "%d %d %d" % (i, in0[i], in1[i])
    #sys.exit(1)

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

    delta = c_int()
    delta = fano_delta

    ok = libfano.nfano_decode(xin0, xin1, n_out, out_array, limit, metric_out, delta)
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
class FFTCache:
    def __init__(self, samples, jrate, jblock):
        self.jrate = jrate
        self.jblock = jblock
        self.samples = samples
        self.memo = { }
        self.granule = int(self.jblock / 8)

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
    # return [ bin, full FFT at offset ].
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
        return [ bin, a ]

    # hza is [ hz0, hzN ] -- at start and end.
    # offset is 0..self.jblock.
    # return buckets[0..162ish][4] -- i.e. a mini-FFT per symbol.
    def get(self, hza, offset):
        return self.getmore(hza, offset, 0)

    # hza is [ hz0, hzN ] -- at start and end.
    # offset is 0..self.jblock.
    # return buckets[0..162ish][4 +/- more] -- i.e. a mini-FFT per symbol.
    # ordinarily more is 0. it's 1 for guess_freq().
    def getmore(self, hza, offset, more):
        # round offset to 1/8th of self.jblock.
        offset = int(offset / self.granule) * self.granule

        bin_hz = self.jrate / float(self.jblock)
        nsyms = (len(self.samples) - offset) // self.jblock
        out = numpy.zeros((nsyms, 4+more+more))
        for i in range(0, nsyms):
            ioff = i * self.jblock + offset
            [ bin, m ] = self.fft1(ioff, hza)
            assert bin - more >= 0
            assert bin+4+more <= len(m)
            m = m[bin-more:bin+4+more]
            out[i] = m
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
          if samples_time == None:
              sec = 0
          else:
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

              self.process(samples[i0:], t)

              bufbuf = [ ]
              nsamples = 0

  # received a message, add it to the list.
  # offset in seconds.
  # drift in hz/minute.
  def got_msg(self, dec):
      if self.verbose:
          drift = dec.hza[1] - dec.hz()
          print("%6.1f %.1f %.1f %d %s" % (dec.hz(), dec.dt, drift, dec.snr, dec.msg))
      dec.decode_time = time.time()
      self.msgs_lock.acquire()
      self.msgs.append(dec)
      self.msgs_lock.release()

  # someone wants a list of all messages received,
  # as array of Decode.
  def get_msgs(self):
      self.msgs_lock.acquire()
      a = copy.copy(self.msgs)
      self.msgs_lock.release()
      return a

  def process(self, samples, samples_time):
    global budget, step_frac, goff_step, fano_limit, driftmax, ndrift

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
    filter = weakutil.butter_bandpass(1380, 1620, self.cardrate, band_order)
    samples = lfilter(filter[0], filter[1], samples)
    # down-convert from 1400 to 100.
    samples = weakutil.freq_shift(samples, -self.downhz, 1.0/self.cardrate)
    # down-sample.
    samples = samples[0::16]

    #
    # pad at start+end b/c transmission might have started early or late.
    # I've observed dt's from -2.8 to +3.4.
    # there's already two seconds of slop at start b/c xmission starts
    # at xx:01 but file seems to start at xx:59.
    # and we're going to trim a second either side after AGC.
    #
    sm = numpy.mean(samples) # pad with plausible signal levels
    sd = numpy.std(samples)

    assert start_slop >= 2.0
    startpad = int((start_slop - 2.0) * self.jrate) # samples
    samples = numpy.append(numpy.random.normal(sm, sd, startpad), samples)

    endpad = int((start_slop*self.jrate + 162.0*self.jblock + end_slop*self.jrate) - len(samples))
    if endpad > 0:
        samples = numpy.append(samples, numpy.random.normal(sm, sd, endpad))
    elif endpad < 0:
        samples = samples[0:endpad]

    bin_hz = self.jrate / float(self.jblock)

    # store each message just once, to suppress duplicates.
    # indexed by message text; value is [ samples_minute, hz, msg, snr, offset, drift ]
    msgs = { }

    ssamples = numpy.copy(samples) # for subtraction
    # phase 0: decode and subtract, but don't use the subtraction.
    # phase 1: decode from subtracted samples.
    phase0_start = time.time()
    phase1_end = t0 + budget
    for phase in range(0, 2):
        if phase == 0:
            phasesamples = samples
        else:
            phasesamples = ssamples # samples with phase0 decodes subtracted
        [ fine_rank, noise ] = self.coarse(phasesamples)
        xf = FFTCache(phasesamples, self.jrate, self.jblock)

        already = { }
        for rr in fine_rank:
            # rr is [ drift, hz, offset, strength ]
            if phase == 0 and len(msgs) > 0:
                if time.time() - phase0_start >= phase0_budget*(phase1_end-phase0_start):
                    break
            else:
                if time.time() - t0 >= budget:
                    break

            drift = rr[0]
            hz = rr[1]
            offset = rr[2]

            #if int(hz / bin_hz) in already:
            #    continue

            hza = [ hz - drift, hz + drift ]
            offset = self.guess_start(xf, hza, offset)
            hza = self.guess_freq(xf, hza, offset)
            ss = xf.get(hza, offset)
        
            # ss has one element per symbol time.
            # ss[i] is a 4-element FFT.
        
            # first symbol is in ss[0]
            # return is [ hza, msg, snr ]
            assert len(ss[0]) >= 4
            dec = self.process1(samples_minute, ss[0:162], hza, noise)
    
            if False:
                if dec != None:
                    print("%.1f %d %.1f %.1f %s -- %s" % (hz, phase, drift, numpy.mean(hza), rr, dec.msg))
                else:
                    print("%.1f %d %.1f %.1f %s" % (hz, phase, drift, numpy.mean(hza), rr))
    
            if dec != None:
                dec.minute = samples_minute
                dec.start = offset
                if not (dec.msg in msgs):
                    already[int(hz / bin_hz)] = True
                    dec.phase0 = (phase == 0)
                    dec.phase1 = (phase == 1)
                    msgs[dec.msg] = dec
                    if phase == 0:
                        ssamples = self.subtract(ssamples, dec, numpy.add(dec.hza, 0.0))
                        if subgap > 0.0001:
                            ssamples = self.subtract(ssamples, dec, numpy.add(dec.hza, subgap))
                            ssamples = self.subtract(ssamples, dec, numpy.add(dec.hza, -subgap))
                #elif dec.snr > msgs[dec.msg].snr:
                #    # we have a higher SNR.
                #    msgs[dec.msg] = dec
    
            sys.stdout.flush()

    for txt in msgs:
        dec = msgs[txt]
        dec.hza[0] += self.downhz
        dec.hza[1] += self.downhz
        dec.dt = (dec.start / float(self.jrate)) - 2.0 # convert to seconds
        self.got_msg(dec)

  def subtract(self, osamples, dec, hza):
      padded = dec.msgbits + ([0] * 31)
      encbits = fano_encode(padded)
      # len(encbits) is 162, each element 0 or 1.

      # interleave encbits, by bit-reversal of index.
      ibits = numpy.zeros(162, dtype=numpy.int32)
      p = 0
      for i in range(0, 256):
          j = bit_reverse(i, 8)
          if j < 162:
              ibits[j] = encbits[p]
              p += 1

      # combine with sync pattern to generate 162 symbols of 0..4.
      four = numpy.multiply(ibits, 2)
      four = numpy.add(four, numpy.divide(numpy.add(pattern, 1), 2))

      bin_hz = self.jrate / float(self.jblock)

      samples = numpy.copy(osamples)

      assert dec.start >= 0

      samples = samples[dec.start:]

      bigslop = int(self.jblock * subslop)

      # find amplitude of each symbol.
      amps = [ ]
      offs = [ ]
      tones = [ ]
      i = 0
      while i < len(four):
          nb = 1
          while i+nb < len(four) and four[i+nb] == four[i]:
              nb += 1

          hz0 = hza[0] + (i / float(len(four))) * (hza[1] - hza[0])
          hz = hz0 + four[i] * bin_hz
          tone = weakutil.costone(self.jrate, hz, self.jblock*nb)

          # nominal start of symbol in samples[]
          i0 = i * self.jblock
          i1 = i0 + nb*self.jblock
          
          # search +/- slop.
          # we search separately for each symbol b/c the
          # phase may drift over the minute, and we
          # want the tone to match exactly.
          i0 = max(0, i0 - bigslop)
          i1 = min(len(samples), i1 + bigslop)
          cc = numpy.correlate(samples[i0:i1], tone)
          mm = numpy.argmax(cc) # thus samples[i0+mm]

          # what is the amplitude?
          # if actual signal had a peak of 1.0, then
          # correlation would be sum(tone*tone).
          cx = cc[mm]
          c1 = numpy.sum(tone * tone)
          a = cx / c1

          amps.append(a)
          offs.append(i0+mm)
          tones.append(tone)

          i += nb

      ai = 0
      while ai < len(amps):
          a = amps[ai]
          off = offs[ai]
          tone = tones[ai]
          samples[off:off+len(tone)] -= tone * a
          ai += 1

      nsamples = numpy.append(osamples[0:dec.start], samples)

      return nsamples

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
      cpu = 0.0
      justone = False
      starttime = time.time()
      while time.time() < starttime + 10:
          hz = 80 + random.random() * 240
          if justone:
              nstart = 0
          else:
              nstart = int(random.random() * 3000)
          nend = 2000 + int(random.random() * 5000)
          symbols = [ ]
          for p in pattern:
              if justone:
                  sym = 0
              else:
                  sym = 2 * random.randint(0, 1)
              if p > 0:
                  sym += 1
              symbols.append(sym)
          samples = numpy.random.normal(0, 0.5, nstart)
          samples = numpy.append(samples, weakutil.fsk(symbols, [ hz, hz ], bin_hz, self.jrate, self.jblock))
          samples = numpy.append(samples, numpy.random.normal(0, 0.5, nend))
          if justone == False:
              samples = samples * 1000
          xf = FFTCache(samples, self.jrate, self.jblock)
          self.guess_offset(xf, [hz+4*bin_hz,hz+4*bin_hz]) # prime the cache, for timing
          t0 = time.time()
          xa = self.guess_offset(xf, [hz,hz])
          t1 = time.time()
          x0start = xa[0][0]
          x0abs = abs(nstart - x0start)
          #print("%.1f %d: %d %d" % (hz, nstart, x0start, x0abs))
          if abs(xa[1][0] - nstart) < x0abs:
              print("%.1f %d: %d %d -- %d" % (hz, nstart, x0start, x0abs, xa[1][0]))
          mo = max(mo, x0abs)
          sumabs += x0abs
          sum += x0start - nstart
          cpu += t1 - t0
          n += 1
          if justone:
              sys.exit(1)
      # jul 18 2016 -- max diff was 53.
      # jun 16 2017 -- max diff was 77 (but with different hz range).
      # jun 16 2017 -- max diff 52, avg abs diff 17, avg diff -1
      # jun 16 2017 -- max diff 33, avg abs diff 16, avg diff 0 (using tone, not bandpass filter)
      # jul  8 2017 -- max diff 33, avg abs diff 16, avg diff 0, avg CPU 0.022
      # jul  8 2017 -- max diff 32, avg abs diff 15, avg diff 0, avg CPU 0.114
      #                but this one uses FFTCache...
      # jul  9 2017 -- max diff 32, avg abs diff 17, avg diff -2, avg CPU 0.006
      print("max diff %d, avg abs diff %d, avg diff %d, avg CPU %.3f" % (mo, sumabs/n, sum/n, cpu / n))

  # a signal starts at roughly offset=start,
  # to within self.jblock/coarse_steps.
  # return a better start.
  def guess_start(self, xf, hza, start):
      candidates = [ ]

      step = int(self.jblock / coarse_steps)
      start0 = start - step // 2
      start0 = max(start0, 0)
      start1 = start + step // 2
      # the "/ 8" here is the FFTCache granule.
      for start in range(start0, start1, self.jblock // 8):
          # tones[0..162][0..4]
          if start + len(pattern)*self.jblock > xf.len():
              continue
          tones = xf.get(hza, start)
          tones = tones[0:162,:]
          tone0 = tones[:,0]
          tone1 = tones[:,1]
          tone2 = tones[:,2]
          tone3 = tones[:,3]

          # we just care about sync vs no sync,
          # so combine tones 0 and 2, and 1 and 3
          syncs0 = numpy.maximum(tone0, tone2)
          syncs1 = numpy.maximum(tone1, tone3)

          # now syncs1 - syncs0.
          # yields +/- that should match pattern.
          tt = numpy.subtract(syncs1, syncs0)

          strength = numpy.sum(numpy.multiply(tt, pattern))

          candidates.append([ start, strength ])

      candidates = sorted(candidates, key = lambda e : -e[1])

      return candidates[0][0]

  # returns an array of [ offset, strength ], sorted
  # by strength, most-plausible first.
  # xf is an FFTCache.
  def guess_offset(self, xf, hza):
      ret = [ ]

      for off in range(0, self.jblock, goff_step):
          # tones[0..162][0..4]
          tones = xf.get(hza, off)
          tone0 = tones[:,0]
          tone1 = tones[:,1]
          tone2 = tones[:,2]
          tone3 = tones[:,3]

          # we just care about sync vs no sync,
          # so combine tones 0 and 2, and 1 and 3
          syncs0 = numpy.maximum(tone0, tone2)
          syncs1 = numpy.maximum(tone1, tone3)

          # now syncs1 - syncs0.
          # yields +/- that should match pattern.
          tt = numpy.subtract(syncs1, syncs0)

          cc = numpy.correlate(tt, pattern)

          indices = list(range(0, len(cc)))
          indices = sorted(indices, key=lambda i : -cc[i])
          indices = indices[0:ngoff]
          offsets = numpy.multiply(indices, self.jblock)
          offsets = offsets + off

          both = [ [ offsets[i], cc[indices[i]] ] for i in range(0, len(offsets)) ]
          ret += both

      ret = sorted(ret, key = lambda e : -e[1])

      return ret

  # returns an array of [ offset, strength ], sorted
  # by strength, most-plausible first.
  # oy: this version is the same quality as guess_offset(),
  # but noticeably slower.
  def fft_guess_offset(self, samples, hz):
      bin_hz = self.jrate / float(self.jblock)
      bin = int(round(hz / bin_hz))

      # shift freq so hz in the middle of a bin
      #samples = weakutil.freq_shift(samples,
      #                              bin * bin_hz - hz,
      #                              1.0/self.jrate)

      tones = [ [], [], [], [] ]
      for off in range(0, len(samples), goff_step):
          if off + self.jblock > len(samples):
              break
          a = numpy.fft.rfft(samples[off:off+self.jblock])
          a = a[bin:bin+4]
          a = abs(a)
          tones[0].append(a[0])
          tones[1].append(a[1])
          tones[2].append(a[2])
          tones[3].append(a[3])

      if False:
          for ti in range(0, 4):
              for i in range(8, 24):
                  sys.stdout.write("%.0f " % (tones[ti][i]))
              sys.stdout.write("\n")

      # we just care about sync vs no sync,
      # so combine tones 0 and 2, and 1 and 3
      syncs0 = numpy.maximum(tones[0], tones[2])
      syncs1 = numpy.maximum(tones[1], tones[3])

      # now syncs1 - syncs0.
      # yields +/- that should match pattern.
      tt = numpy.subtract(syncs1, syncs0)

      if False:
              for i in range(0, 24):
                  sys.stdout.write("%.0f " % (tt[i]))
                  if (i % 8) == 7:
                      sys.stdout.write("| ")
              sys.stdout.write("\n")

      #z = numpy.repeat(pattern, int(self.jblock / goff_step))
      z = [ ]
      nfill = int(self.jblock / goff_step) - 1
      for x in pattern:
          z.append(x)
          z = z + [0]*nfill
      cc = numpy.correlate(tt, z)

      if False:
              for i in range(0, min(len(cc), 24)):
                  sys.stdout.write("%.0f " % (cc[i]))
                  if (i % 8) == 7:
                      sys.stdout.write("| ")
              sys.stdout.write("\n")

      indices = list(range(0, len(cc)))
      indices = sorted(indices, key=lambda i : -cc[i])
      offsets = numpy.multiply(indices, goff_step)
      #offsets = numpy.subtract(offsets, ntaps / 2)

      both = [ [ offsets[i], cc[indices[i]] ] for i in range(0, len(offsets)) ]

      return both

  # returns an array of [ offset, strength ], sorted
  # by strength, most-plausible first.
  def convolve_guess_offset(self, samples, hz):
      bin_hz = self.jrate / float(self.jblock)

      ntaps = self.jblock

      # average y down to a much lower rate to make the
      # correlate() go faster. 64 works well.

      # filter each of the four tones
      tones = [ ]
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

          # scipy.signal.resample(yx, len(yx) / goff_step) works, but too slow.
          # re = weakutil.Resampler(goff_step*64, 64)
          # yx = re.resample(yx)
          yx = weakutil.moving_average(yx, goff_step)
          yx = yx[0::goff_step]

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

      z = numpy.repeat(pattern, int(self.jblock / goff_step))
      cc = numpy.correlate(tt, z)

      indices = list(range(0, len(cc)))
      indices = sorted(indices, key=lambda i : -cc[i])
      offsets = numpy.multiply(indices, goff_step)
      offsets = numpy.subtract(offsets, ntaps / 2)

      both = [ [ offsets[i], cc[indices[i]] ] for i in range(0, len(offsets)) ]

      return both

  # given hza[hz0,hz1], return a new hza adjusted to
  # give stronger tones.
  # start is offset in samples[].
  def guess_freq(self, xf, hza, start):
      more = 1
      ss = xf.getmore(hza, start, more)
      # ss has one element per symbol time.
      # ss[i] is a 6-element FFT, with ss[i][1] as lowest tone.
      # first symbol is in ss[0]
      bin_hz = self.jrate / float(self.jblock)
      diffs = [ ]
      for pi in range(0, len(pattern)):
          if pattern[pi] > 0:
              sync = 1
          else:
              sync = 0
          fft = ss[pi]
          sig0 = fft[sync+more]
          sig1 = fft[2+sync+more]
          if sig0 > sig1:
              bin = sync+more
          else:
              bin = 2+sync+more
          if fft[bin] > fft[bin-1] and fft[bin] > fft[bin+1]:
              xp = weakutil.parabolic(numpy.log(fft), bin) # interpolate
              # xp[0] is a better bin number (with fractional bin)
              diff = (xp[0] - bin) * bin_hz
              if diff > bin_hz / 2:
                  diff = bin_hz / 2
              elif diff < -bin_hz / 2:
                  diff = -bin_hz / 2
          else:
              diff = 0.0
          diffs.append(diff)

      nhza = [
          hza[0] + numpy.mean(diffs[0:80]),
          hza[1] + numpy.mean(diffs[81:162])
          ]
      return nhza

  # do a coarse pass over the band, looking for
  # possible signals.
  # bud is budget in seconds.
  # returns [ fine_rank, noise ]
  def coarse(self, samples):
    bin_hz = self.jrate / float(self.jblock)

    # WSPR signals officially lie between 1400 and 1600 Hz.
    # we've down-converted to 100 - 300 Hz.
    # search a bit wider than that.
    min_hz = self.lowhz-20
    max_hz = self.lowhz+200+20

    # generate a few copies of samples corrected for various amounts of drift.
    if ndrift <= 1:
        drifts = [ 0.0 ]
    else:
        drifts = [ ]
        driftstart = -driftmax
        driftend = driftmax+0.001
        driftinc = 2.0*driftmax / (ndrift - 1)
        for drift in numpy.arange(driftstart, driftend, driftinc):
            drifts.append(drift)

    # sum FFTs over the whole two minutes to find likely frequencies.
    # coarse_rank[i] is the sum of the four tones starting at bin i,
    # so that the ranks refer to a signal whose base tone is in bin i.
    clusters = { }
    for hzoff in numpy.arange(0, bin_hz, 0.001 + bin_hz/coarse_hzsteps):
        for drift in drifts:
            dsamples = weakutil.freq_shift_ramp(samples, [hzoff+drift,hzoff-drift], 1.0/self.jrate)
            for off in numpy.arange(0, self.jblock, (self.jblock / float(coarse_steps)) + 1, dtype=numpy.int32):
                nbins = (self.jblock // 2) + 1
                coarse = numpy.zeros(nbins) # for noise
                ncoarse = 0 # for noise
                nsyms = (len(dsamples) - off) // self.jblock
                mat = numpy.zeros((nbins, nsyms))
                for sym in range(0, nsyms):
                    start = off + (sym * self.jblock)
                    if start+self.jblock > len(dsamples):
                        break
                    block = dsamples[start:start+self.jblock]
                    a = numpy.fft.rfft(block)
                    a = abs(a)
                    mat[:,sym] = a
                    coarse = numpy.add(coarse, a)
                    ncoarse = ncoarse + 1
                coarse = coarse / ncoarse # sum -> average, for noise calculation
                for bin in range(0, nbins-4):
                    tone0 = mat[bin+0,:]
                    tone1 = mat[bin+1,:]
                    tone2 = mat[bin+2,:]
                    tone3 = mat[bin+3,:]
                    syncs0 = numpy.maximum(tone0, tone2)
                    syncs1 = numpy.maximum(tone1, tone3)
                    tt = numpy.subtract(syncs1, syncs0)

                    # normalize to emphasize correlation rather than random loudness
                    tt = tt / numpy.mean(abs(tt))

                    cc = numpy.correlate(tt, pattern)
                    indices = list(range(0, len(cc)))
                    indices = sorted(indices, key=lambda i : -cc[i])
                    indices = indices[0:coarse_top1]
                    offsets = numpy.multiply(indices, self.jblock)
                    offsets = offsets + off
                    hz = bin*bin_hz - hzoff
                    both = [ [ drift, hz, offsets[i], cc[indices[i]] ] for i in range(0, len(offsets)) ]
                    for e in both:
                        k = str(int(e[1] / (bin_hz/2))) + "-" + str(int(e[2] / (self.jblock/2)))
                        #k = bin
                        if not k in clusters:
                            clusters[k] = [ ]
                        clusters[k].append(e)

    # just the best few from each bin
    coarse_rank = [ ]
    for k in clusters:
        v = clusters[k]
        v = sorted(v, key = lambda e : -e[3])
        v = v[0:coarse_top2]
        coarse_rank += v
    
    # sort coarse bins, biggest signal first.
    # coarse_rank[i] = [ drift, hz, start, strength ]
    coarse_rank = [ e for e in coarse_rank if (e[1] >= min_hz and e[1] < max_hz) ]
    coarse_rank = sorted(coarse_rank, key = lambda e : -e[3])

    # calculate noise for snr, mimicing wsjtx wsprd.c.
    # first average in freq domain over 7-bin window.
    # then noise from 30th percentile.
    nn = numpy.convolve(coarse, [ 1, 1, 1, 1, 1, 1, 1 ])
    nn = nn / 7.0
    nn = nn[6:]
    nns = sorted(nn[int(min_hz/bin_hz):int(max_hz/bin_hz)])
    noise = nns[int(0.3*len(nns))]

    return [ coarse_rank, noise ]

  # returns None or a Decode
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
    runlen = len(pattern) // statruns
    winners = [ ]
    losers = [ ]
    for pi1 in range(0, len(pattern)):
        sig0 = levels[pi1][0]
        sig1 = levels[pi1][1]
        runi = pi1 // runlen
        if len(winners) < runi+1:
            winners.append([])
            losers.append([])
        winners[runi].append(max(sig0, sig1))
        losers[runi].append(min(sig0, sig1))
    winmean = [ numpy.mean(x) for x in winners ]
    winstd = [ numpy.std(x) for x in winners ]
    losemean = [ numpy.mean(x) for x in losers ]
    losestd = [ numpy.std(x) for x in losers ]

    # power rather than voltage.
    rawsnr = (numpy.mean(winmean)*numpy.mean(winmean)) / (noise*noise)
    # the "-1" turns (s+n)/n into s/n
    rawsnr -= 1
    if rawsnr < 0.1:
        rawsnr = 0.1
    rawsnr /= (2500.0 / 1.5) # 1.5 hz noise b/w -> 2500 hz b/w
    snr = 10 * math.log10(rawsnr)

    if snr < ignore_thresh:
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
          # http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf
          # this works the best of all these approaches.

          runi = pi // runlen

          # if a 0 were sent, how likely is v0? (it's the "0" FSK bin)
          p00 = problt(v0, winmean[runi], winstd[runi])
          # if a 1 were sent, how likely is v0?
          p01 = probgt(v0, losemean[runi], losestd[runi])

          # if a 0 were sent, how likely is v1? (it's the "1" FSK bin)
          p10 = probgt(v1, losemean[runi], losestd[runi])
          # if a 1 were sent, how likely is v1?
          p11 = problt(v1, winmean[runi], winstd[runi])

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
        if p0 < fano_floor:
            p0 = fano_floor
        logp0 = math.log(2*p0, 2) - fano_bias
        logp0 = int(round(logp0 * fano_scale))
        if p1 < fano_floor:
            p1 = fano_floor
        logp1 = math.log(2*p1, 2) - fano_bias
        logp1 = int(round(logp1 * fano_scale + 0.5))

        softsyms.append( [ logp0, logp1, v0, v1 ] )

    #for i in range(0, len(softsyms)):
    #    print "%d %d %d" % (i, softsyms[i][0], softsyms[i][1])
    #sys.exit(1)

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

    [ msgbits, metric ] = nfano_decode(sym0, sym1)

    if msgbits == None:
        # Fano could not decode
        return None

    if numpy.array_equal(msgbits[0:80], [0]*80):
        # all bits are zero
        return None

    msgbits = msgbits[0:-31] # drop the 31 bits of padding, yielding 50 bits

    msg = self.unpack(msgbits)
    if msg == None:
        return None

    dec = Decode(hza, msg, snr, msgbits)
    return dec

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
                  pfx[i] = chr(ord('A') + (nc - 10))
              else:
                  pfx[i] = " "
              n3 /= 37
          return "%s%s%s/%s" % (pfx[0], pfx[1], pfx[2], call)
      else:
          # suffix of 1 or 2 characters.
          nc = n3 - 60000
          if nc >= 0 and nc <= 9:
              return "%s/%s" % (call, chr(ord('0')+nc))
          if nc >= 10 and nc <= 35:
              return "%s/%s" % (call, chr(ord('A')+(nc-10)))
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
    extra = 0 # how many decodes that seem spurious
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
                for dec in all:
                    mymsg = dec.msg
                    mymsg = mymsg.strip()
                    mymsg = re.sub(r'  *', ' ', mymsg)
                    if mymsg in wsx:
                        found = dec
                        got[dec.msg] = True

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
                        ph = ""
                        if found.phase0:
                            ph += "0"
                        if found.phase1:
                            ph += "1"
                        print("yes %4.0f %s (%.1f %.1f) %s %s" % (float(whz), wa[2], found.hz(), found.dt, ph, wmsg))
                else:
                    any_no = True
                    if verbose:
                        print("no  %4.0f %s %s" % (float(whz), wa[2], wmsg))
                sys.stdout.flush()
        for dec in all:
            if not (dec.msg in got):
                # only increase extra if looks like a bad decode.
                extraok = False
                oklist = [ "<...>", "WD4AHB", "OM1AI", "VE8TEA", "CG3EXP", "K0WFS", "VK7DD",
                           "N3SZ", "G4DJB", "KV0S", "W0BY", "M0RTP", "VE3DXK", "KK4YEL",
                           "DL0DFF", "VA3ROM", "EA5CYA", "K4EH", "G4IUP", "VE3CJE", "DL6NL",
                           "IK1WVQ" ]
                for call in oklist:
                    if call in dec.msg:
                        extraok = True
                if extraok == False:
                    extra += 1
                if verbose:
                    if extraok:
                        sys.stdout.write("OK ")
                    else:
                        sys.stdout.write("BAD ")
                    print("EXTRA: %6.1f %s" % (dec.hz(), dec.msg))
    if verbose:
        print("score %d of %d -- %d extra" % (score, wanted, extra))
    return [ score, wanted, extra ]

vars = [
    [ "fano_scale", [ 2, 2.5, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5 ] ],
    [ "fano_floor", [ 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.01, 0.025, 0.05, 0.1 ] ],
    [ "fano_limit", [ 5000, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000 ] ],
    [ "fano_bias", [ 0.3, 0.35, 0.4, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.6 ] ],
    [ "start_slop", [ 2, 3, 4, 5, 6, 7 ] ],
    [ "end_slop", [ 2, 3, 4, 5, 6, 7 ] ],
    [ "coarse_top1", [ 1, 2, 3, 4 ] ],
    [ "coarse_top2", [ 1, 2, 3, 4, 5, 6, 7 ] ],
    [ "coarse_hzsteps", [ 1, 2, 3, 4, 5, 6 ] ],
    [ "coarse_steps", [ 1, 2, 3, 4, 6, 8 ] ],
    [ "step_frac", [ 1, 1.5, 2, 2.5, 3, 4 ] ],
    [ "band_order", [ 2, 3, 4, 5, 6, 7, 8 ] ],
    [ "subslop", [ 0.005, 0.01, 0.02 ] ],
    [ "ndrift", [ 1, 2, 3, 4, 5 ] ],
    [ "phase0_budget", [ 0.3, 0.4, 0.5, 0.6 ] ],
    [ "subgap", [ 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2 ] ],
    [ "driftmax", [ 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 3 ] ],
    [ "goff_step", [ 32, 64, 128, 256 ] ],
    [ "statruns", [ 1, 2, 3, 4, 8 ] ],
    [ "fano_delta", [ 7, 17, 27, 37, 47, 57 ] ],
    [ "ignore_thresh", [ -26, -28, -30, -32, -34, -40 ] ],
    [ "budget", [ 9, 20, 50 ] ],
#    [ "ngoff", [ 1, 2, 3, 4, 6 ] ],
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

            [ score, wanted, extra ] = benchmark(wsjtfile, False)
            exec("%s%s = old" % (xglob, v[0]))
            sys.stdout.write("%s=%s : " % (v[0], val))
            sys.stdout.write("%d %d %.1f\n" % (score, extra, score - extra/2.0))
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
