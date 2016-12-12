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
# Robert Morris, AB1HL
#

import numpy
import wave
import weakaudio
import weakutil
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
import threading
import re
import random
import multiprocessing
from scipy.signal import lfilter
import ctypes
from ctypes import c_int, byref, cdll
import resource
import collections
import gc

#
# performance tuning parameters.
#
budget = 9     # CPU seconds (9)
noffs = 3      # look for sync every jblock/noffs (2)
off_scores = 1 # consider off_scores*noffs starts per freq bin (3, 4)
pass1_frac = 0.1 # fraction budget to spend before subtracting (0.5, 0.9, 0.5)
hetero_thresh = 6 # zero out bin that wins too many times (9, 5, 7)
soft_iters = 75 # try r-s soft decode this many times (35, 125, 75)
min_soft = 10  # always designate at least this many errors for soft decode
max_soft = 40  # never more than this many soft error symbols

# information about one decoded signal.
class Decode:
    def __init__(self,
                 hza,
                 nerrs,
                 msg,
                 snr,
                 minute,
                 start,
                 twelve,
                 decode_time):
        self.hza = hza
        self.nerrs = nerrs
        self.msg = msg
        self.snr = snr
        self.minute = minute
        self.start = start
        self.twelve = twelve
        self.decode_time = decode_time

    def hz(self):
        return numpy.mean(self.hza)

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
             "TG7HQQ", "475IVR", "L16RAH", "XO2QLH", "5E8HML", "HF7VBA",
             "F11XTN", "7T4EUZ", "EF5KYD", "A80CCM", "HF7VBA",
             "VV3EZD", "DT8ZBT", "8Z9RTD", "7U0NNP", "6P8CGY", "WH9ASY",
             "V96TCU", "BF3AUF", "7B5JDP", "1HFXR1", "28NTV" ]
    for bad in bads:
        if bad in msg:
            return True
    return False

very_first_time = True

class JT65:
  debug = False

  offset = 0

  def __init__(self):
      self.done = False
      self.msgs_lock = thread.allocate_lock()
      self.msgs = [ ]
      self.verbose = False
      self.enabled = True # True -> run process(); False -> don't

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
    bufbuf = None
    self.process(samples, 0)

  def opencard(self, desc):
      # self.cardrate = 11025 # XXX
      self.cardrate = 11025 / 2 # XXX jrate
      self.audio = weakaudio.new(desc, self.cardrate)

  def gocard(self):
      samples_time = time.time()
      bufbuf = [ ]
      nsamples = 0
      while self.done == False:
          sec = self.second(samples_time)
          if sec < 48 or nsamples < 48*self.cardrate:
              # give lower-level audio a chance to use
              # bigger batches, may help resampler() quality.
              time.sleep(1.0)
          else:
              time.sleep(0.2)

          [ buf, buf_time ] = self.audio.read()

          if len(buf) > 0:
              bufbuf.append(buf)
              nsamples += len(buf)
              samples_time = buf_time
              if numpy.max(buf) > 30000 or numpy.min(buf) < -30000:
                  sys.stderr.write("!")

          # wait until we have enough samples through 49th second of minute.
          # we want to start on the minute (i.e. a second before nominal
          # start time), and end a second after nominal end time.
          # thus through 46.75 + 2 = 48.75.

          sec = self.second(samples_time)
          if sec >= 49 and nsamples >= 49*self.cardrate:
              # we have >= 49 seconds of samples, and second of minute is >= 49.

              samples = numpy.concatenate(bufbuf)
              bufbuf = [ ]

              # sample # of start of minute.
              i0 = len(samples) - self.cardrate * self.second(samples_time)
              i0 = int(i0)
              t = samples_time - (len(samples)-i0) * (1.0/self.cardrate)
              self.process(samples[i0:], t)

              samples = None
              nsamples = 0

  def close(self):
      # ask gocard() thread to stop.
      self.done = True 

  # received a message, add it to the list.
  # dec is a Decode.
  def got_msg(self, dec):
      self.msgs_lock.acquire()

      # already in msgs with worse nerrs?
      found = False
      for i in range(max(0, len(self.msgs)-40), len(self.msgs)):
          xm = self.msgs[i]
          if xm.minute == dec.minute and abs(xm.hz() - dec.hz()) < 10 and xm.msg == dec.msg:
              # we already have this msg
              found = True
              if dec.nerrs < xm.nerrs:
                  self.msgs[i] = dec
                  
      if found == False:
          self.msgs.append(dec)

      self.msgs_lock.release()

  # someone wants a list of all messages received.
  # each msg is a Decode.
  # # each msg is [ minute, hz, msg, decode_time, nerrs, snr ]
  def get_msgs(self):
      self.msgs_lock.acquire()
      a = copy.copy(self.msgs)
      self.msgs_lock.release()
      return a

  # fork the real work, to try to get more multi-core parallelism.
  def process(self, samples, samples_time):
      global budget
      global very_first_time
      if very_first_time:
          # warm things up.
          very_first_time = False
          thunk = (lambda dec : self.got_msg(dec))
          self.process0(samples, samples_time, thunk, 0, 2580)
          return

      sys.stdout.flush()

      # parallelize the work by audio frequency: one thread
      # gets the low half, the other thread gets the high half.
      # the ranges have to overlap so each can decode
      # overlapping (interfering) transmissions.

      rca = [ ] # connections on which we'll receive
      pra = [ ] # child processes
      tha = [ ] # a thread to read from each child
      txa = [ ]
      npr = 2
      for pi in range(0, npr):
          min_hz = pi * (2580 / npr)
          min_hz = max(min_hz - 50, 0)

          max_hz = (pi + 1) * (2580 / npr)
          max_hz = min(max_hz + 175, 2580)

          txa.append(time.time())

          recv_conn, send_conn = multiprocessing.Pipe(False)
          p = multiprocessing.Process(target=self.process00,
                                      args=[samples, samples_time, send_conn,
                                            min_hz, max_hz])
          p.start()
          send_conn.close()
          pra.append(p)
          rca.append(recv_conn)

          th = threading.Thread(target=lambda c=recv_conn: self.readchild(c))
          th.start()
          tha.append(th)

      for pi in range(0, len(rca)):
          t0 = time.time()
          pra[pi].join(budget+2.0)
          if pra[pi].is_alive():
              print "\n%s child process still alive, enabled=%s\n" % (self.ts(time.time()),
                                                                      self.enabled)
              pra[pi].terminate()
              pra[pi].join(2.0)
          t1 = time.time()
          tha[pi].join(2.0)
          if tha[pi].isAlive():
              t2 = time.time()
              print "\n%s reader thread still alive, enabled=%s, %.1f %.1f %.1f\n" % (self.ts(t2),
                                                                                      self.enabled,
                                                                                      t0-txa[pi],
                                                                                      t1-t0,
                                                                                      t2-t1)
          rca[pi].close()

  def readchild(self, recv_conn):
      while True:
          try:
              dec = recv_conn.recv()
              # x is a Decode
              self.got_msg(dec)
          except:
              break

  # in child process.
  def process00(self, samples, samples_time, send_conn, min_hz, max_hz):
      gc.disable() # no point since will exit soon
      thunk = (lambda dec : send_conn.send(dec))
      self.process0(samples, samples_time, thunk, min_hz, max_hz)
      send_conn.close()

  # for each decode, call thunk(Decode).
  # only look at sync tones from min_hz .. max_hz.
  def process0(self, samples, samples_time, thunk, min_hz, max_hz):
    global budget, noffs, off_scores, pass1_frac

    if self.enabled == False:
        return

    if self.verbose:
        print "len %d %.1f, type %s, rates %.1f %.1f" % (len(samples),
                                                         len(samples) / float(self.cardrate),
                                                         type(samples[0]),
                                                         self.cardrate,
                                                         self.jrate)
        sys.stdout.flush()

    if type(samples[0]) != numpy.int16 and type(samples[0]) != numpy.float32:
        print "samples %s" % (type(samples[0]))

    if False:
        print "saving to aaa.wav, max %.0f" % (numpy.max(samples))
        weakutil.writewav1(samples, "aaa.wav", self.cardrate)

    # for budget.
    t0 = time.time()

    # samples_time is UNIX time that samples[0] was
    # sampled by the sound card.
    samples_minute = self.minute(samples_time + 30)

    if self.cardrate != self.jrate:
      # reduce rate from self.cardrate to self.jrate.
      assert self.jrate >= 2 * 2500
      if False:
          filter = weakutil.butter_lowpass(2500.0, self.cardrate, order=10)
          samples = scipy.signal.lfilter(filter[0],
                                         filter[1],
                                         samples)
          samples = weakutil.resample(samples, self.cardrate, self.jrate)
      else:
          # resample in pieces so that we can preserve float32,
          # since lfilter insists on float64.
          rs = weakutil.Resampler(self.cardrate, self.jrate)
          resampleblock = self.cardrate # exactly one second works best
          si = 0
          ba = [ ]
          while si < len(samples):
              block = samples[si:si+resampleblock]
              nblock = rs.resample(block)
              nblock = nblock.astype(numpy.float32)
              ba.append(nblock)
              si += resampleblock
          samples = numpy.concatenate(ba)

    # pad at start+end b/c transmission might have started early
    # or late.
    #sm = numpy.mean(abs(samples))
    sm = numpy.mean(abs(samples[2000:5000]))
    r0 = (numpy.random.random(self.jrate*3) - 0.5) * sm * 2
    r0 = r0.astype(numpy.float32)
    r1 = (numpy.random.random(self.jrate*4) - 0.5) * sm * 2
    r1 = r1.astype(numpy.float32)
    samples = numpy.concatenate([ r0, samples, r1 ])

    bin_hz = self.jrate / float(self.jblock)
    minbin = max(5, int(min_hz  / bin_hz))
    maxbin = int(max_hz / bin_hz)

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
          # a = numpy.fft.rfft(block)
          # a = abs(a)
          a = weakutil.arfft(block)
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
        indices = indices[0:off_scores]
        for ii in indices:
          scores.append([ j, cc[ii], True, offs[oi] + ii*self.jblock ])

    # release m[]'s memory.
    m = None

    # scores[i] = [ bin, correlation, valid, start ]

    if False:
        want_hz = 571
        bin_hz = self.jrate / float(self.jblock)
        for e in scores:
            ehz = e[0] * bin_hz
            if abs(ehz - want_hz) < 2.5:
                print e
                e[1] *= 100

    # highest scores first.
    scores = sorted(scores, key=lambda sc : -sc[1])

    ssamples = numpy.copy(samples) # subtracted
    already = { } # suppress duplicate msgs
    decodes = 0

    # first without subtraction.
    # don't blow the whole budget, to ensure there's time
    # to start decoding on subtracted signals.
    i = 0
    while i < len(scores) and (time.time() - t0) < budget * pass1_frac:
        hz = scores[i][0] * (self.jrate / float(self.jblock))
        dec = self.process1(samples, hz, noise, scores[i][3], already)
        if dec != None:
            decodes += 1
            dec.minute = samples_minute
            thunk(dec)
            scores[i][2] = False
            ssamples = self.subtract_v3(ssamples, dec.hza, dec.start, dec.twelve)
        i += 1
    nfirst = i

    # filter out entries that we just decoded.
    scores = [ x for x in scores if x[2] ]

    # now try again, on subtracted signal.
    # we do a complete new pass since a strong signal might have
    # been unreadable due to another signal at a somewhat higher
    # frequency.
    i = 0
    while i < len(scores) and (time.time() - t0) < budget:
        hz = scores[i][0] * (self.jrate / float(self.jblock))
        dec = self.process1(ssamples, hz, noise, scores[i][3], already)
        if dec != None:
            decodes += 1
            dec.minute = samples_minute
            thunk(dec)
            # this subtract() is important for performance.
            ssamples = self.subtract_v3(ssamples, dec.hza, dec.start, dec.twelve)
        i += 1

    if self.verbose:
        print "%d..%d, did %d of %d, %d hits, maxrss %.1f MB" % (
            min_hz,
            max_hz,
            nfirst+i,
            len(scores),
            decodes,                                              
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0*1024.0))

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
  # this is not used any more; it's a better use of CPU time
  # to just look for sync correlation at a bunch of sub-symbol offsets.
  #
  def guess_offset(self, samples, hz):
      if True:
          # FIR filter so we can predict delay through the filter.
          ntaps = 1001 # XXX 1001 works well
          fdelay = ntaps / 2
          taps = weakutil.bandpass_firwin(ntaps, hz-4, hz+4, self.jrate)

          # filtered = lfilter(taps, 1.0, samples)

          # filtered = numpy.convolve(samples, taps, mode='valid')
          # filtered = scipy.signal.convolve(samples, taps, mode='valid')
          filtered = scipy.signal.fftconvolve(samples, taps, mode='valid')
          # hack to match size of lfilter() output
          filtered = numpy.append(numpy.zeros(ntaps-1), filtered)
      else:
          # butterworth IIR
          fdelay = 920 # XXX I don't know how to predict filter delay.
          filter = weakutil.butter_bandpass(hz - 4, hz + 4, self.jrate, 3)
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
      ya = weakutil.moving_average(y, downfactor)
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
              ff = weakutil.freq_from_fft(sx, self.jrate,
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
  # returns None or a Decode
  def process1(self, samples, xhz, noise, start, already):
    if len(samples) < 126*self.jblock:
        return None

    bin_hz = self.jrate / float(self.jblock) # FFT bin size, in Hz

    dec = self.process1a(samples, xhz, start, noise, already)

    return dec

  # returns a Decode, or None
  def process1a(self, samples, xhz, start, noise, already):
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
    if self.sync_bin(hza, 0) < 5:
        return None
    if self.sync_bin(hza, 125) < 5:
        return None
    if self.sync_bin(hza, 0) + 2+64 > self.jblock/2:
        return None
    if self.sync_bin(hza, 125) + 2+64 > self.jblock/2:
        return None

    m = [ ]
    for i in range(0, 126):
      # block = block * scipy.signal.blackmanharris(len(block))
      sync_bin = self.sync_bin(hza, i)
      sync_hz = self.sync_hz(hza, i)
      freq_off = sync_hz - (sync_bin * bin_hz)
      block = samples[i*self.jblock:(i+1)*self.jblock]

      # block = weakutil.freq_shift(block, -freq_off, 1.0/self.jrate)
      # a = numpy.fft.rfft(block)
      a = weakutil.fft_of_shift(block, -freq_off, self.jrate)

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
          if j < len(m[pi]) and (bestj == None or m[pi][j] > bestv):
            bestj = j
            bestv = m[pi][j]
        if bestj != None:
          wins[bestj-sync_bin] += 1

    # zero out bins that win too often. a given symbol
    # (bin) should only appear two or three times in
    # a transmission.
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

        # mean of bins in same time slot
        s1 = numpy.mean(m[pi][sync_bin+2:sync_bin+2+64])

        if s1 != 0.0:
            strength.append(s0 / s1)
        else:
            strength.append(0.0)

    [ nerrs, msg, twelve ] = self.process2(sa, strength)

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

    return Decode(hza, nerrs, msg, snr, None,
                  start, twelve, time.time())

  # sa[] is 63 channel symbols, each 0..63.
  # it needs to be un-gray-coded, un-interleaved,
  # un-reed-solomoned, &c.
  # strength[] indicates how sure we are about each symbol
  #   (ratio of winning FFT bin to second-best bin).
  def process2(self, sa, strength):
    global soft_iters

    # un-gray-code
    for i in range(0, len(sa)):
      sa[i] = weakutil.gray2bin(sa[i], 6)

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

        best = None # [ matching_symbols, nerrs, msg, twelve ]

        # try various numbers of erasures.
        for iter in range(0, soft_iters):
            #nera = (iter % 35) + 5
            nera = int((random.random() * (max_soft - min_soft)) + min_soft)
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
                # re-encode, count symbols that match input to decoder, as score.
                sy1 = self.rs_encode(twelve)
                eqv = numpy.equal(sy1, sa)
                neq = collections.Counter(eqv)[True]
                if best == None or neq > best[0]:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    if best != None:
                        sys.stdout.write("nerrs=%d neq=%d %s -> " % (best[1], best[0], best[2]))
                        sys.stdout.write("nera=%d nerrs=%d neq=%d %s\n" % (nera, nerrs, neq, msg))
                    sys.stdout.flush()
                    best = [ neq, nerrs, msg, twelve ]
            
        if best != None:
            sys.stdout.flush()
            return best[1:]

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
        # sys.exit(1)
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
        gray = [ weakutil.bin2gray(x, 6) for x in inter ]

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
  sys.stderr.write("       jt65.py -nbench dir\n")
  sys.stderr.write("       jt65.py -send msg\n")
  # list sound cards
  weakaudio.usage()
  sys.exit(1)

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

def benchmark1(dir, bfiles, verbose):
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

        filename = dir + "/" + bf[1]
        r = JT65()
        r.verbose = False

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
                    mymsg = x.msg
                    mymsg = mymsg.replace(" ", "")
                    if mymsg == wsmsg:
                        found = True
                        got[x.msg] = True
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
                if x.nerrs < 25 and not (x.msg in got):
                    print "MISSING: %6.1f %d %.0f %s" % (x.hz(), x.nerrs, x.snr, x.msg)
    if verbose:
        print "score %d of %d" % (score, wanted)
    return [ score, wanted ]

# given a file with wsjt-x 1.6.0 results, sitting in a directory
# fill of the corresponding .wav files, do our own decoding and
# compare results with wsjt-x.
# e.g. benchmark("nov/wsjt.txt") or benchmark("jt65files/big.txt").
# wsjt-x output is cut-and-paste from wsjt-x decode display.
# 2211  -1 -0.3  712 # S56IZW WB8CQV R-13
# 2211 -12  0.0  987 # VE2NCG K8GLC EM88
# wav file names look like 161122_2211.wav
def benchmark(wsjtfile, verbose):
    dir = os.path.dirname(wsjtfile)
    minutes = { } # keyed by hhmm
    wsjtf = open(wsjtfile, "r")
    for line in wsjtf:
        line = re.sub(r'\xA0', ' ', line) # 0xA0 -> space
        line = re.sub(r'[\r\n]', '', line)
        m = re.match(r'^([0-9]{4}) +[0-9.-]+ +[0-9.-]+ +[0-9]+ +# *(.*)$', line)
        if m == None:
            print "oops: " + line
            continue
        hhmm = m.group(1)
        if not hhmm in minutes:
            minutes[hhmm] = ""
        minutes[hhmm] += line + "\n"
    wsjtf.close()

    info = [ ]
    for hhmm in minutes:
        ff = [ x for x in os.listdir(dir) if re.match('......_' + hhmm + '.wav', x) != None ]
        if len(ff) == 1:
            filename = ff[0]
            info.append([ True, filename, minutes[hhmm] ])
        elif len(ff) == 0:
            sys.stderr.write("could not find .wav file in %s for %s\n" % (dir, hhmm))
        else:
            sys.stderr.write("multiple files in %s for %s: %s\n" % (dir, hhmm, ff))

    return benchmark1(dir, info, verbose)

def optimize(wsjtfile):
    vars = [
        # [ "weakutil.use_numpy_rfft", [ False, True ] ],
        # [ "weakutil.use_numpy_arfft", [ False, True ] ],
        # [ "weakutil.fos_threshold", [ 0.125, 0.25, 0.5, 0.75, 1.0 ] ],
        [ "noffs", [ 1, 2, 3, 4, 6, 8 ] ],
        [ "pass1_frac", [ 0.05, 0.1, 0.2, 0.3 ] ],
        [ "min_soft", [ 5, 10, 15, 20, 25, 30, 35 ] ],
        [ "max_soft", [ 30, 35, 40, 45 ] ],
        [ "budget", [ 6, 9, 15 ] ],
        [ "soft_iters", [ 0, 25, 50, 75, 100, 200, 400, 800 ] ],
        # [ "off_scores", [ 1, 2, 4 ] ],
        # [ "hetero_thresh", [ 4, 6, 8, 10 ] ],
        ]

    sys.stdout.write("# ")
    for v in vars:
        sys.stdout.write("%s=%s " % (v[0], eval(v[0])))
    sys.stdout.write("\n")

    # warm up any caches, JIT, &c.
    r = JT65()
    r.verbose = False
    r.gowav("jt65files/j7.wav", 0)

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

            sc = benchmark(wsjtfile, False)
            exec "%s%s = old" % (xglob, v[0])
            sys.stdout.write("%s=%s : " % (v[0], val))
            sys.stdout.write("%d\n" % (sc[0]))
            sys.stdout.flush()

def main():
    # gc.set_debug(gc.DEBUG_STATS)
    thv = gc.get_threshold()
    gc.set_threshold(10*thv[0], 10*thv[1], 10*thv[2])

    filenames = [ ]
    desc = None
    bench = None
    opt = None
    send_msg = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-in":
            desc = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-file":
            filenames.append(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "-bench":
            bench = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-opt":
            opt = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-send":
            send_msg = sys.argv[i+1]
            i += 2
        else:
            usage()

    if send_msg != None:
        js = JT65Send()
        js.send(send_msg)
        sys.exit(0)
        
    if bench != None:
        benchmark(bench, True)
        sys.exit(0)

    if opt != None:
        optimize(opt)
        sys.exit(0)
  
    if len(filenames) > 0 and desc == None:
        r = JT65()
        r.verbose = True
        for filename in filenames:
            r.gowav(filename, 0)
    elif len(filenames) == 0 and desc != None:
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
