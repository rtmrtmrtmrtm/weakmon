#!/usr/local/bin/python

#
# WWVB phase-shift keying sound-card demodulator
#
# set radio to 59 khz, upper side band.
# radio must be within 10 hz of correct frequency.
#
# my setup uses a Z10024A low-pass filter to keep out AM broadcast.
# an outdoor dipole or indoor W1VLF antenna work well.
#
# Robert Morris, AB1HL
#

import numpy
import wave
import weakaudio
import weakcat
import weakutil
import weakargs
import weakaudio
import scipy
import scipy.signal
import sys
import os
import math
import time
import calendar
import subprocess
import thread
import argparse

# a[] and b[] are -1/0/1 bit sequences.
# in how many bits are they identical?
def bitmatch(a, b):
  n = 0
  i = 0
  while i < len(a) and i < len(b):
    if a[i] != 0 and b[i] != 0 and a[i] == b[i]:
      n += 1
    i += 1
  return n

# invert a -1/1 bit sequence.
def invert(a):
  b = a[:]
  for i in range(0, len(b)):
    b[i] *= -1
  return b

# part of wwvb checksum work.
# tm[] is 0/1 array of the 26 time bits.
# a[] is the array of indices of bits to xor.
def xsum(tm, a):
  z = 0
  for i in range(0, len(a)):
    b = tm[a[i]]
    z ^= b
  if z == 0:
    return -1
  else:
    return 1

# http://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
def smooth(values, window):
  weights = numpy.repeat(1.0, window)/window
  sma = numpy.convolve(values, weights, 'valid')
  return sma

class WWVB:
  center = 1000     # 60 khz shifted to here in audio
  filterwidth = 20  # bandpass filter width in hertz
  searchhz = 10     # only look for WWVB at +/- searchhz

  # set these to True in order to search for the signal in
  # time and frequency. set them to False if the PC clock is
  # correct, the radio frequency is accurate, and the goal
  # is to measure reception quality rather than to learn the time.
  searchtime = True
  searchfreq = True

  debug = False

  filter = None
  c2filter = None
  c3filter = None
  samples = numpy.array([0])
  offset = 0
  flywheel = 0
  flyfreq = None # carrier frequency of last good ecc

  # remember all the good CRC offset/minute pairs,
  # to try to guess the most likely correct time for
  # any given minute with a bad CRC.
  timepairs = numpy.zeros([0,2]) # each is [ offset, minute ]

  def __init__(self):
    pass

  def openwav(self, filename):
    self.wav = wave.open(filename)
    self.wav_channels = self.wav.getnchannels()
    self.wav_width = self.wav.getsampwidth()
    self.rate = self.wav.getframerate()

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
    while True:
      buf = self.readwav(chan)
      if buf.size < 1:
        break
      self.gotsamples(buf, 0)
      while self.process(False):
        pass
    self.process(True)

  def opencard(self, desc):
      self.rate = 8000
      self.audio = weakaudio.new(desc, self.rate)

  def gocard(self):
      while True:
          [ buf, buf_time ] = self.audio.read()

          if len(buf) > 0:
              mx = numpy.max(numpy.abs(buf))
              if mx > 30000:
                  sys.stderr.write("!")
              self.gotsamples(buf, buf_time)
              while self.process(False):
                  pass
          else:
              time.sleep(0.2)

  def gotsamples(self, buf, time_of_last):
    # the band-pass filter.
    if self.filter == None:
      self.filter = weakutil.butter_bandpass(self.center - self.filterwidth/2,
                                             self.center + self.filterwidth/2,
                                             self.rate, 3)
      self.zi = scipy.signal.lfiltic(self.filter[0],
                                     self.filter[1],
                                     [0])

    zi = scipy.signal.lfilter(self.filter[0], self.filter[1], buf, zi=self.zi)
    self.samples = numpy.concatenate((self.samples, zi[0]))
    self.zi = zi[1]

    # remember time of self.sample[0]
    # XXX off by filter delay
    self.samples_time = time_of_last - len(self.samples) / float(self.rate)

  def guess1(self, a, center, width):
    fx = weakutil.freq_from_fft(a, self.rate, center - width/2, center + width/2)
    return fx
    
  # guess the frequency of the WWVB carrier.
  # only looks +/- 10 hz.
  def guess(self):
    # apply FFT to abs(samples) then divide by two,
    # since psk has no energy at "carrier".
    sa = numpy.abs(self.samples)
    
    n = 0
    fx = 0
    sz = 59*self.rate
    while (n+1)*sz <= len(sa) and (n+1)*sz <= 60*self.rate:
      xx = self.guess1(sa[n*sz:(n+1)*sz], 2*self.center, self.searchhz * 2.0)
      fx += xx
      n += 1
    fx /= n

    return fx / 2.0

  # guess what the minute must be for a given sample offset,
  # based on past decoded minutes and their sample offsets.
  def guessminute(self, offset):
    if len(self.timepairs) < 1:
      return -1

    offsets = self.timepairs[:,0]
    minutes = self.timepairs[:,1]
    xx = numpy.subtract(offset, offsets)
    xx = numpy.divide(xx, self.rate * 60.0)
    guesses = numpy.add(minutes, xx)
    m2 = numpy.median(guesses)

    return m2

  # minute m at sample offset passed CRC.
  # add it to timepairs.
  def addpair(self, offset, m):
    self.timepairs = numpy.concatenate((self.timepairs, [[ offset, m ]]))

  # what bits do we expect for this minute?
  # return -1, 0, 1 for bit=0, bit=unknown, bit=1.
  def guessbits(self, m):
    # sync
    bits = [ -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1 ]
    if m >= 0:
      # time crc
      bits += self.crc(m)[::-1]
      # the time
      for i in range(25, -1, -1):
        if (m & (1 << i)) != 0:
          bits += [ 1 ]
        else:
          bits += [ -1 ]
        if i == 25 or i == 16 or i == 7:
          bits += [ 0 ]
    else:
      # don't know the time or CRC.
      for i in range(13, 47):
        bits += [ 0 ]
    # dst/ls and notice XXX DST off
    bits += [ -1, 1, 0, -1, -1, -1 ]
    # dst XXX 2nd sun of march, 1st sun of nov
    bits += [ -1, 1, 1, -1, 1, 1 ]
    # final sync bit
    bits += [ -1 ]
    return bits

  def process(self, eof):
    bitsamples = float(self.rate)
    if eof:
      if len(self.samples) < 65 * bitsamples:
        return False
    elif self.flywheel > 0 and len(self.timepairs) > 1:
      if len(self.samples) < 65 * bitsamples:
        return False
    else:
      if len(self.samples) < 130 * bitsamples:
        return False

    if False:
        print "saving to aaa.wav, max %.0f" % (numpy.max(self.samples))
        weakutil.writewav1(self.samples, "aaa.wav", self.rate)

    # pad at start and end in case samples started late / ended early.
    #sm = numpy.mean(self.samples)
    #sd = numpy.std(self.samples)
    #self.samples = numpy.append(numpy.random.normal(sm, sd, self.rate), self.samples)
    #self.samples = numpy.append(self.samples, numpy.random.normal(sm, sd, self.rate))

    # the main mode of failure seems to be that the
    # measured frequency is wrong, perhaps because of
    # heterodynes very close to 60 khz. so try
    # a few different carrier frequencies.
    # a frequency error of 0.008 Hz (in 1000 Hz) is enough to
    # shift the phase by 180 degrees by the end of the minute.
    # sadly the ECC is weak enough that you can't try too
    # many things before getting a falsely correct ECC.

    guessfx = self.guess()
    if self.searchfreq:
      # we're not sure what the correct frequency is
      fx = guessfx
    else:
      # self.center is the exact frequency to look for
      fx = self.center

    # try to decode.
    # match is sample number.
    # matchscore is -1 or (more or less) the number of matching bits.
    # bits is the 60 demodulated bits.
    # eccok is True iff the error-correcting code was correct.
    # m is 0 or the decoded minute.
    (match, matchscore, bits, eccok, m) = self.tryfreq(fx)

    if self.searchfreq and self.flyfreq != None:
      (match2, matchscore2, bits2, eccok2, m2) = self.tryfreq(self.flyfreq)
      if matchscore2 > matchscore:
        # the match at the flywheel frequency was stronger than at
        # the guessed frequency.
        matchscore = matchscore2
        fx = self.flyfreq
        match = match2
        bits = bits2
        eccok = eccok2
        m = m2

    if self.searchtime:
      guessm = self.guessminute(self.offset + match)
      if guessm > 0 and match >= 0 and len(bits) >= 60:
        guessbits = self.guessbits(int(round(guessm)))
        ngood = bitmatch(guessbits, bits)
      else:
        ngood = 0
    else:
      guessm = self.unix2minute(time.time())
      guessbits = self.guessbits(guessm)
      if m > 0 and len(bits) >= 60:
        ngood = bitmatch(guessbits, bits)
      else:
        ngood = 0

    matchsec = 0
    if match >= 0:
      tt = self.samples_time + match / float(self.rate)
      ttt = time.gmtime(int(tt))
      matchsec = ttt.tm_sec + (tt - int(tt))

    # print this minute's results:
    # time from UNIX clock.
    # time decoded from WWVB.
    # audio frequency (nominally 1000 hz).
    # time offset (in seconds) between UNIX clock and WWVB signal.

    if match >= 0:
        unix = self.samples_time + match / float(self.rate)
        unixts = self.ts(unix + 30)
    else:
        unixts = self.ts(time.time())
    if eccok:
        wwvbts = self.ts(self.minute2unix(m))
    else:
        wwvbts = "-"
    print "%s %s %.3f %.2f" % (unixts,
                               wwvbts,
                               guessfx,
                               matchsec)

    sys.stdout.flush()

    if match >= 0:
      if eccok:
        self.flywheel = 5
        self.flyfreq = fx
        self.addpair(self.offset + match, m)
      consume = match + 59*bitsamples
    else:
      if self.flywheel > 0:
        consume = 60*bitsamples
      else:
        consume = len(self.samples) - 60*bitsamples
        if consume > 50*bitsamples:
          consume = 50*bitsamples

    consume = int(round(consume))
    self.samples = self.samples[consume:]
    self.offset += consume
    self.samples_time += consume / float(self.rate)
    self.flywheel -= 1

    if False:
      sys.stderr.write("quitting after first minute\n")
      sys.exit(0)

    return True

  # current UNIX time.time() to real minute of century.
  def unix2minute(self, now):
    century = time.strptime("1 jan 2000 UTC", "%d %b %Y %Z")
    csec = calendar.timegm(century)
    mins = (now - csec) / 60
    mins -= len(self.samples) / (self.rate * 60.0)
    mins = int(round(mins))
    return mins

  # convert WWVB minute of century to UNIX seconds-since-1970.
  def minute2unix(self, min):
    century = time.strptime("1 jan 2000 UTC", "%d %b %Y %Z")
    csec = calendar.timegm(century)
    csec += min * 60
    return csec

  # turn UNIX seconds into UTC hh:mm for printing.
  def ts(self, now):
      tm = time.gmtime(int(now))
      s = time.strftime("%H:%M", tm)
      return s

  # try to decode, given a guess at the carrier audio frequency.
  def tryfreq(self, fx):
    bitsamples = float(self.rate)
    secondspersample = 1.0 / float(self.rate)
    cyclespersample = secondspersample * fx
    phasepersample = cyclespersample * 2.0 * numpy.pi

    # synthetic carrier at frequency fx.
    i0 = 0
    i1 = len(self.samples)
    n = i1 - i0
    ph0 = i0 * phasepersample
    ph1 = ph0 + n * phasepersample
    qq = numpy.pi / 2.0
    ref = numpy.sin(numpy.linspace(ph0, ph1 - phasepersample, n))
    refq = numpy.sin(numpy.linspace(ph0 + qq, ph1 - phasepersample + qq, n))

    # product of carrier and signal.
    # this yields a steady-ish > 0 when in phase,
    # and < 0 when out of phase.
    prod = numpy.multiply(self.samples, ref)
    prodq = numpy.multiply(self.samples, refq)
    smoothwindow = int(round(bitsamples))
    prod = smooth(prod, smoothwindow)
    prodq = smooth(prodq, smoothwindow)
    if numpy.mean(numpy.abs(prodq)) > numpy.mean(numpy.abs(prod)):
      # 90 degree shift of carrier gave stronger product
      prod = prodq

    # indices of samples that are just before each
    # zero crossing in either direction.
    # this finds bit boundaries.
    zc = numpy.where(numpy.diff(numpy.sign(prod)))[0]
    za = [ ]
    for zz in zc:
      # reference to start
      bn = zz / bitsamples
      ri = (bn - int(bn)) * bitsamples
      za.append(ri)
    if len(za) < 1:
      starti = 0
    else:
      starti = numpy.median(za) # good start of bit
    samples_starti = starti + int(smoothwindow/2)
    midi = starti + (bitsamples / 2) # good start of bit

    # decode bits. may have zero and one reversed; will
    # fix after we find the sync sequence.
    bits = [ ]
    i = midi
    while round(i) < len(prod):
      if prod[int(round(i))] > 0:
        bb = -1
      else:
        bb = 1
      bits += [ bb ]
      i += bitsamples

    # do we trust guess at where minute starts and
    # what the current time is?
    flying = (self.flywheel > 0) and (len(self.timepairs) > 1)

    # generate a set of known bits to look for,
    # including current time+CRC if known.
    if self.searchtime == False:
      # look for current real time
      pat = self.guessbits(self.unix2minute(time.time()))
    elif flying:
      # we probably know the current time bits to expect.
      guessm = int(round(self.guessminute(self.offset)))
      pat = self.guessbits(guessm)
    else:
      # we may not know the current time.
      pat = self.guessbits(-1)

    # also an inverted copy of the search pattern.
    patr = invert(pat)

    # look for offset of sync sequence &c in bits[].
    if self.searchtime == False or flying or len(bits) >= 120:
      # there must be a match in these samples, either because
      # flywheel says a minute starts at bits[1], or becuse
      # we have two entire minutes of bits.
      if self.searchtime == False:
        # real-time samples, so we know where minute started.
        tt = self.samples_time + samples_starti / float(self.rate)
        tti = int(round(tt))
        ttt = time.gmtime(tti)
        # samples[0] arived at ttt.tm_sec -- second within minute
        i0 = 60 - ttt.tm_sec
        if i0 >= 60:
          i0 -= 60
        if i0 < 0:
          print "oops i0 %d" % (i0)
          i0 = 0
        i1 = i0 + 1
        # print "starti %d tt %.3f tm_sec %d i0 %d" % (starti, tt, ttt.tm_sec, i0)
      elif flying:
        # we expect minute to start at bits[1], but
        # might have gradually slipped over time.
        i0 = 0
        i1 = 3
      else:
        # we don't know where second should start.
        i0 = 0
        i1 = len(bits) - 60
      match = -1
      matchscore = -1
      matchrev = False
      matchecc = False
      for i in range(i0, i1):
        score1 = bitmatch(pat, bits[i:])
        if score1 > matchscore:
          matchscore = score1
          match = i
          matchrev = False
          (matchecc, xm) = self.decode(bits[match:match+60])
          if matchecc:
            matchscore += 10
        score2 = bitmatch(patr, bits[i:])
        if score2 > matchscore:
          matchscore = score2
          match = i
          matchrev = True
          xbits = bits[:]
          for j in range(0, len(xbits)):
            xbits[j] *= -1
          (matchecc, xm) = self.decode(xbits[match:match+60])
          if matchecc:
            matchscore += 10
      if matchrev:
        for j in range(0, len(bits)):
          bits[j] *= -1
    else:
      # we don't know where second should start.
      match = -1
      i = 0
      while match < 0 and i <= len(bits) - 60:
        if bits[i:i+13] == pat:
          match = i
        if bits[i:i+13] == patr:
          for j in range(0, len(bits)):
            bits[j] *= -1
          match = i
        i += 1

    if match >= 0:
      (eccok, m) = self.decode(bits[match:match+60])
      return (samples_starti + match*self.rate, matchscore, bits[match:match+60], eccok, m)
    else:
      return (samples_starti + match*self.rate, -1, bits, False, 0)

  # return CRC for a given minute.
  # returns the five CRC bits in an array, as -1 or 1.
  def crc(self, m):
    tm = []
    for i in range(0, 26):
      if m & (1 << i):
        tm.append(1)
      else:
        tm.append(0)

    # calculate expected hamming ECC code on time
    p = [ 0, 0, 0, 0, 0 ]
    p[0] = xsum(tm, [ 23, 21, 20, 17, 16, 15, 14, 13, 9, 8, 6, 5, 4, 2, 0 ])
    p[1] = xsum(tm, [ 24, 22, 21, 18, 17, 16, 15, 14, 10, 9, 7, 6, 5, 3, 1 ])
    p[2] = xsum(tm, [ 25, 23, 22, 19, 18, 17, 16, 15, 11, 10, 8, 7, 6, 4, 2 ])
    p[3] = xsum(tm, [ 24, 21, 19, 18, 15, 14, 13, 12, 11, 7, 6, 4, 3, 2, 0 ])
    p[4] = xsum(tm, [ 25, 22, 20, 19, 16, 15, 14, 13, 12, 8, 7, 5, 4, 3, 1 ])

    return p

  def decode(self, bits):
    if len(bits) < 60:
      return ( False, 0 )
    if self.debug:
      sys.stdout.write("sync:   ")
      for i in range(0, 13):
        sys.stdout.write("%d " % ((bits[i]+1)/2))
      sys.stdout.write("\n")

      sys.stdout.write("ecc:    ")
      for i in range(13, 18):
        sys.stdout.write("%d " % ((bits[i]+1)/2))
      sys.stdout.write("\n")

      sys.stdout.write("time:   ")
      for i in range(18, 47):
        if i == 19 or i == 29 or i == 39:
          sys.stdout.write("(%d) " % ((bits[i]+1)/2))
        else:
          sys.stdout.write("%d " % ((bits[i]+1)/2))
      sys.stdout.write("\n")

      sys.stdout.write("dst/ls: ")
      for i in range(47, 53):
        sys.stdout.write("%d " % ((bits[i]+1)/2))
      sys.stdout.write("\n")

      sys.stdout.write("dst:    ")
      for i in range(53, 59):
        sys.stdout.write("%d " % ((bits[i]+1)/2))
      sys.stdout.write("\n")

      sys.stdout.write("end:    ")
      for i in range(59, 60):
        sys.stdout.write("%d " % ((bits[i]+1)/2))
      sys.stdout.write("\n")

    # binary minute of century
    m = 0
    e = 25
    i = 18
    while i < 47:
      if bits[i] > 0:
        m += 2**e
      i += 1
      e -= 1
      if i == 19 or i == 29 or i == 39:
        i += 1

    p = self.crc(m)
    
    if m != 0 and p[4] == bits[13] and p[3] == bits[14] and p[2] == bits[15] and p[1] == bits[16] and p[0] == bits[17]:
      eccok = True
    else:
      eccok = False
    
    return (eccok, m)

def main():
  parser = weakargs.stdparse('Decode phase-shift WWVB.')
  parser.add_argument("-center", metavar='Hz', default=1000.0, type=float)
  parser.add_argument("-file")
  args = weakargs.parse_args(parser)

  if args.cat != None:
    cat = weakcat.open(args.cat)
    cat.set_usb_data()
    cat.setf(0, 59000)

  if (args.card == None) == (args.file == None):
    parser.error("one of -card and -file are required")

  if args.file != None:
    r = WWVB()
    r.center = args.center
    r.gowav(args.file, 0)
    sys.exit(0)

  if args.card != None:
    r = WWVB()
    r.center = args.center
    r.opencard(args.card)
    r.gocard()
    sys.exit(0)

  parser.error("one of -card, -file, or -levels is required")

  sys.exit(1)

main()
