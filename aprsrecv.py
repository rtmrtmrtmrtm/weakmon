#!/usr/local/bin/python

#
# APRSRecv
#

import numpy
import wave
import weakaudio
import time
import scipy
import sys
import os
import math
from scipy.signal import butter, lfilter, filtfilt

# make a butterworth bandpass filter
def butter_bandpass(lowcut, highcut, samplerate, order=5):
  # http://wiki.scipy.org/Cookbook/ButterworthBandpass
  nyq = 0.5 * samplerate
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='bandpass')
  return b, a

# http://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
def smooth (values, window):
  weights = numpy.repeat(1.0, window)/window
  sma = numpy.convolve(values, weights, 'valid')
  return sma

# https://github.com/tcort/va2epr-tnc/blob/master/firmware/aprs.c
# Thomas Cort <linuxgeek@gmail.com>
# update a CRC with one new byte.
# initial crc should be 0xffff.
def crciter(crc, byte):
  byte &= 0xff
  crc ^= byte
  for i in range(0, 8):
    if crc & 1:
      crc = (crc >> 1) ^ 0x8408
    else:
      crc >>= 1
  return crc

def crc16(bytes):
  crc = 0xffff
  for b in bytes:
    crc = crciter(crc, b)
  return crc

class APRSRecv:
  debug = False
  
  baud = 1200 # bits per second

  # Bell-202
  mark = 1200
  space = 2200

  rate = 0 # samples/second

  # buffer of samples waiting to be processed,
  # and where we are in the buffer. need to keep
  # a window before/after where we're looking so
  # we can find a good slicing level.
  # a buffer for each of mark/space, filtered
  mark_smooth = numpy.array([0])
  space_smooth = numpy.array([0])
  seq0 = 0 # global sample # of mark/space_smooth[0]
  nextseq = 0.0 # global sample # of start of next symbol to ingest

  shifted = False # figures shift
  column = 0

  markfilter = None
  spacefilter = None

  laststart = 0
  emitted = "" # accumulate output in this string
  emitf = None # write output here
  muted = False
  no_markleft = True

  symbols = [ ]

  charsdir = None

  marknoise = 1.0 # last noise sample, from mark_smoothed
  spacenoise = 1.0
  marksnrsum = 0.0 # running sum of mark snrs
  marknsnr = 0 # number of samples in marksnrsum
  spacesnrsum = 0.0
  spacensnr = 0

  def __init__(self, rate):
    self.rate = rate
    self.resetsnr()
    for name in [ "dump", "mark", "space", "m1", "s1", "ss",
                  "in", "inmark", "inspace" ]:
      try:
        os.remove(name)
      except:
        pass

  def openwav(self, filename):
    self.wav = wave.open(filename)
    self.wav_channels = self.wav.getnchannels()
    self.wav_width = self.wav.getsampwidth()
    self.rate = self.wav.getframerate()
    sys.stdout.write("file=%s chans=%d width=%d rate=%d\n" % (filename,
                                                              self.wav_channels,
                                                              self.wav_width,
                                                              self.rate))

  def readwav(self):
    z = self.wav.readframes(1024)
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

  def gotsamples(self, buf):
    if self.debug:
      f = open("in", "a")
      numpy.savetxt(f, buf)
      f.close()
    
    # the two tone filters.
    if self.markfilter == None:
      shift = abs(self.mark - self.space)
      width = 600 # originally 1000
      order = 1 # originally 3
      self.markfilter = butter_bandpass(self.mark - width/2,
                                        self.mark + width/2,
                                        self.rate, order)
      self.markzi = scipy.signal.lfiltic(self.markfilter[0],
                                         self.markfilter[1],
                                         [0])
      self.spacefilter = butter_bandpass(self.space - width/2,
                                         self.space + width/2,
                                         self.rate, order)
      self.spacezi = scipy.signal.lfiltic(self.spacefilter[0],
                                          self.spacefilter[1],
                                          [0])

    mzi = lfilter(self.markfilter[0], self.markfilter[1], buf, zi=self.markzi)
    mark_fil = mzi[0]
    self.markzi = mzi[1]

    szi = lfilter(self.spacefilter[0], self.spacefilter[1], buf, zi=self.spacezi)
    space_fil = szi[0]
    self.spacezi = szi[1]

    if self.debug:
      f = open("inmark", "a")
      numpy.savetxt(f, mark_fil)
      f.close()
      f = open("inspace", "a")
      numpy.savetxt(f, space_fil)
      f.close()

    # detection: find amplitude of signal.
    mark_det = numpy.abs(mark_fil)
    space_det = numpy.abs(space_fil)

    # smooth; this is the low-pass filter one needs post-detection.
    # needed so that 1) fewer false zeroes and 2) sampling
    # doesn't encounter so much noise, and uses most of
    # bit's energy.
    # the complexity here is carrying over the end of the
    # previous buffer.
    # XXX there is probably a more scientific approach.

    win = int(self.rate / self.baud)
    if self.no_markleft:
      self.markleft = numpy.zeros(win)
      self.spaceleft = numpy.zeros(win)
      self.no_markleft = False
    mmm = numpy.concatenate((self.markleft[0:win], mark_det))
    sss = numpy.concatenate((self.spaceleft[0:win], space_det))
    mark_smooth = smooth(mmm, win)[0:len(mark_det)]
    space_smooth = smooth(sss, win)[0:len(mark_det)]
    self.markleft = mark_det[-win:]
    self.spaceleft = space_det[-win:]

    self.mark_smooth = numpy.concatenate((self.mark_smooth, mark_smooth))
    self.space_smooth = numpy.concatenate((self.space_smooth, space_smooth))

  # find a good slicing level.
  # call with mark_smooth or space_smooth.
  # return value is half-way between min and max.
  # don't use average since mark and space aren't equally popular.
  # already filtered/smoothed so no point in using percentiles.
  def mid(self, smoothed):
    hi = numpy.max(smoothed)
    lo = numpy.min(smoothed)
    return (hi + lo) / 2

  def process(self):
    bitsamples = self.rate / float(self.baud)
    flagsamples = bitsamples * 9 # HDLC 01111110 flag (9 b/c NRZI)

    while self.mark_smooth.size - (self.nextseq - self.seq0) > 3*flagsamples+1:
      # while loop assures we have 3*flagsamples into the future.
      # don't keep more than 2*flagsamples in the past
      trim = int(self.nextseq - self.seq0 - 1 - 2*flagsamples)
      if trim > 0:
        self.mark_smooth = self.mark_smooth[trim:]
        self.space_smooth = self.space_smooth[trim:]
        self.seq0 += trim
      
      # find good slicing levels, over a window to
      # left and right of current position.
      off = int(self.nextseq - self.seq0)
      w1 = int(max(off - 13.5*bitsamples, 0))
      w2 = int(off + 18*bitsamples)
      mmid = self.mid(self.mark_smooth[w1:w2])
      smid = self.mid(self.space_smooth[w1:w2])
      
      # slice
      m1 = numpy.subtract(self.mark_smooth, mmid)
      s1 = numpy.subtract(self.space_smooth, smid)
      sliced = numpy.subtract(m1, s1)

      # decode one bit
      middle = int(off + 0.5*bitsamples)
      self.gotsymbol(self.seq0 + middle, sliced[middle])

      onext = int(self.nextseq)

      # does a flag start in this symbol?
      # the first symbol of the terminating flag is the last
      # symbol of the packet due to NRZI. so we actually check
      # if the flag started at the symbol *before* the one we
      # just decoded. it's safe if we mistakenly decode one too
      # many symbols b/c finish frame ignores partial bytes.
      flagoff = off - bitsamples
      if flagoff < 0:
        flagoff = 0 # only happens when program first starts
      ff = self.findflag(sliced[int(flagoff):int(flagoff+flagsamples+bitsamples)])
      if ff != None:
        self.finishframe(self.seq0 + flagoff)
        self.resetsnr()
        # try to find a more exact start for this flag, by looking
        # over a wider window.
        if self.debug:
          print "flag %d" % (self.seq0 + ff + flagoff)
        # advance only flagsamples - bitsamples so we
        # don't eat the first 0 of the next flag (if any).
        # this means gotsymbol() sees last symbol of flag
        # before first symbol of frame.
        # self.nextseq will be start of first payload symbol,
        # i.e. bitsamples/2 before the middle of the symbol.
        self.nextseq = self.seq0 + ff + flagoff + flagsamples - bitsamples
        self.nextseq = self.adjust(sliced, self.nextseq)
      else:
        # just advance over the one bit
        self.nextseq += bitsamples

      if self.debug:
        n = int(self.nextseq) - onext
        off = onext - self.seq0
        f = open("dump", "a")
        f.write("# %d\n" % (onext))
        numpy.savetxt(f,
                      numpy.column_stack([ numpy.arange(onext,
                                                        onext+n),
                                           sliced[off:off+n] ]),
                      "%d %f")
        f.close()
        f = open("mark", "a")
        numpy.savetxt(f, self.mark_smooth[off:off+n], "%f")
        f.close()
        f = open("space", "a")
        numpy.savetxt(f, self.space_smooth[off:off+n], "%f")
        f.close()
        f = open("m1", "a")
        numpy.savetxt(f, m1[off:off+n], "%f")
        f.close()
        f = open("s1", "a")
        numpy.savetxt(f, s1[off:off+n], "%f")
        f.close()

  # process one raw mark/space. x > 0 if mark, x < 0 if space.
  # center of symbol is at self.mark/space_smooth[off-self.seq0].
  def gotsymbol(self, off, x):
    if x > 0:
      self.symbols.append(1) # mark
      zz = 50
      if self.marknoise > 0:
        snr = self.mark_smooth[off-self.seq0] / self.marknoise
      else:
        snr = 0.0001
      self.marksnrsum += snr
      self.marknsnr += 1
      self.spacenoise = self.space_smooth[off-self.seq0]
    else:
      self.symbols.append(0) # space
      zz = -50
      if self.spacenoise > 0:
        snr = self.space_smooth[off-self.seq0] / self.spacenoise
      else:
        snr = 0.0001
      self.spacesnrsum += snr
      self.spacensnr += 1
      self.marknoise = self.mark_smooth[off-self.seq0]
    if self.debug:
      f = open("ss", "a")
      f.write("%d %d\n" % (off, zz))
      f.close()

  # symbols[0] contains 0=space 1=mark.
  # symbols[0] should be the last symbol of the hdlc flag.
  # decode NRZI.
  # de-bit-stuff.
  # endseq is sample # of last symbol, for reporting
  # back to callback().
  def finishframe(self, endseq):
    # un-nrzi symbols to get bits.
    bits = [ ]
    for i in range(1, len(self.symbols)):
      if self.symbols[i] == self.symbols[i-1]:
        bits.append(1) # no change in tone -> 1
      else:
        bits.append(0) # change -> 0

    # un-bit-stuff. every sequence of five ones must be followed
    # by a zero, which we should delete.
    nones = 0
    nbits = [ ]
    for i in range(0, len(bits)):
      if nones == 5:
        #if bits[i] != 0:
        #  sys.stderr.write("stuff oops %d %d\n" % (self.seq0, i))
        nones = 0
        # don't append the zero...
      elif bits[i] == 1:
        nones += 1
        nbits.append(bits[i])
      else:
        nbits.append(bits[i])
        nones = 0
    bits = nbits

    bytes = [ ]
    by = 0
    shf = 0
    for i in range(0, len(bits)):
      by |= (bits[i] << shf) # least-significant bit first
      shf += 1
      if (i % 8) == 7:
        #for j in range(7, -1, -1):
        #  sys.stdout.write("%d" % ((by >> j) & 1))
        #sys.stdout.write(" %02x " % (by))
        #if by > 32 and by < 127:
        #  sys.stdout.write(" %c " % chr(by))
        #else:
        #  sys.stdout.write(" ? ")
        #xx = (by >> 1)
        #if xx > 32 and xx < 127:
        #  sys.stdout.write(" %c " % chr(xx))
        #sys.stdout.write("\n")
        bytes.append(by)
        by = 0
        shf = 0

    ok = self.checkpacket(bytes)
    if ok > 0 or len(bytes) > 16:
      msg = self.printpacket(bytes)
      bitsamples = self.rate / float(self.baud)
      pktsamples = int(len(self.symbols) * bitsamples)
      self.callback(endseq - pktsamples, pktsamples, ok, msg)
      #print "%d bits, %d mod 8" % (len(bits), len(bits) % 8)
      if self.debug:
        for bi in range(0, len(bytes)):
          sys.stdout.write("%3d: " % (bi))
          xx = bytes[bi]
          for i in range(0, 8):
            sys.stdout.write("%d" % ((xx >> 7) & 1))
            xx <<= 1
          by = bytes[bi]
          sys.stdout.write(" %02x " % (by))
          xx = by >> 1
          if xx > 32 and xx < 127:
            sys.stdout.write("%c" % (chr(xx)))
          sys.stdout.write("\n")
      sys.stdout.flush()

    self.symbols = [ ]

  # 0: syntactically unlikely to be a packet
  # 1: syntactically plausible but crc failed
  # 2: crc is correct
  def checkpacket(self, bytes):
    if len(bytes) < 18 or len(bytes) > 500:
      return 0

    crc = crc16(bytes[0:-2])
    crc ^= 0xffff
    # last two packet bytes are crc-low crc-high
    if bytes[-2] == (crc & 0xff) and bytes[-1] == ((crc >> 8) & 0xff):
      return 2
    
    i = 0

    # addresses
    addrsdone = False
    while i+7 < len(bytes) and addrsdone == False:
      for j in range(i, i+6):
        if bytes[j] & 1:
          if self.debug:
            print "early address termination"
          return 0
      i += 6
      ssid = bytes[i]
      i += 1
      x = (ssid >> 1) & 0xf
      if ssid & 1:
        addrsdone = True
    if addrsdone == False:
      if self.debug:
        print "no address termination"
      return 0

    if i + 4 > len(bytes):
      if self.debug:
        print "too short"
      return 0
    if bytes[i] != 0x03:
      if self.debug:
        print "control not 0x03"
      return 0
    if bytes[i+1] != 0xf0:
      if self.debug:
        print "PID not 0xf0"
      return 0
    i += 2

    return 1

  def printpacket(self, bytes):
    i = 0

    msg = ""
    # msg += "%.1f " % (self.getsnr()) # useless

    # addresses
    addrsdone = False
    while i+7 < len(bytes) and addrsdone == False:
      for j in range(i, i+6):
        x = bytes[j] >> 1
        if x > 32 and x < 127:
          msg += "%c" % (chr(x))
        elif x == 32:
          pass
        else:
          msg += "."
      i += 6
      ssid = bytes[i]
      i += 1
      x = (ssid >> 1) & 0xf
      if x > 0:
        msg += "-%d" % (x)
      if ssid & 1:
        addrsdone = True
      msg += " "

    if i + 2 < len(bytes):
      # sys.stdout.write("ctl=%02x pid=%02x " % (bytes[i], bytes[i+1]))
      i += 2

    return msg

  # look for an HDLC flag -- 01111110.
  # it is NRZI, so it is really msssssssm or smmmmmmms.
  # check that it really is a flag.
  # sliced is an array of smoothed and sliced samples.
  # returns the index in slice of the start of the first
  # flag bit -- i.e. the zero crossing.
  def findflag(self, sliced):
    bitsamples = self.rate / float(self.baud)
    flagsamples = bitsamples * 9 # HDLC 01111110 flag (9 b/c NRZI)

    pat = [ -1, 1, 1, 1, 1, 1, 1, 1, -1 ]
    a = numpy.zeros(int(round(flagsamples)))
    for i in range(0, len(pat)):
      ai = int((i + 0.5) * bitsamples)
      a[ai] = pat[i]
    cc = numpy.correlate(sliced, a)
    mm1 = numpy.argmax(cc)
    mm2 = numpy.argmin(cc)

    # thresholds to ensure strong bits
    #shi = numpy.percentile(sliced, 90)
    #slo = numpy.percentile(sliced, 10)
    shi = numpy.max(sliced)
    slo = numpy.min(sliced)
    thresh = min(shi, -slo) / 4.0

    # check that the samples really do form a flag.
    if abs(cc[mm1]) > abs(cc[mm2]):
      mul = 1
      start = mm1
    else:
      mul = -1
      start = mm2
    ok = True
    for i in range(0, len(pat)):
      ai = (i + 0.5) * bitsamples
      x = sliced[int(ai+start)] * mul
      if((pat[i] < 0 and x > -thresh) or
         (pat[i] > 0 and x < thresh)):
         ok = False

    if ok:
      return start
    else:
      return None

  #
  # improve our notion of where a symbol starts.
  # "start" is current belief about the global sample number
  # of the start of a symbol. use zero-crossings
  # to make a new start that will cause
  # subsequent bits to be well framed. this solves problems
  # with skewed bit sampling points caused by the two tones
  # having different amplitudes.
  # XXX doesn't help.
  #
  def adjust(self, sliced, start):
    start -= self.seq0
    bitsamples = self.rate / float(self.baud)
    zc = numpy.where(numpy.diff(numpy.sign(sliced)))[0]
    sum = start
    n = 1
    for z in zc:
      nominal = bitsamples * round((z - start) / bitsamples)
      sum += z - nominal
      n += 1
    nstart = int(sum / n)
    return nstart + self.seq0

  def resetsnr(self):
    self.marksnrsum = 0
    self.marknsnr = 0
    self.spacesnrsum = 0
    self.spacensnr = 0

  # calculate and reset SNR.
  def getsnr(self):
    snr = 0
    if self.marknsnr > 0 and self.spacensnr > 0:
      marksnr = self.marksnrsum / float(self.marknsnr)
      spacesnr = self.spacesnrsum / float(self.spacensnr)
      snr = (marksnr + spacesnr) / 2.0
      snr = math.log10(snr) * 10
    self.resetsnr()
    return snr

  oneret = None
  def onecb(self, start, n, fate, msg):
    info = [ int(start), n, fate, msg ]
    o = self.oneret
    if o != None and o[2] > 0 and info[2] == 0:
      return
    if o != None and o[2] > 0 and info[2] > 0:
      print "double %s / %s" % (repr(self.oneret),
                                               repr(info))
    self.oneret = info

  #
  # decode just one packet, contained in samples[].
  # return [ startseq, n, fate, msg ]
  #
  def one(self, buf):
    self.callback = self.onecb
    self.oneret = None

    # process() wants 2*flagsamples in the past and
    # 3*flagsamples into the future.
    bitsamples = self.rate / float(self.baud)
    flagsamples = int(bitsamples * 9) # HDLC 01111110 flag (9 b/c NRZI)
    zeros = numpy.zeros([4*flagsamples], dtype=numpy.float64)

    self.gotsamples(zeros)
    self.gotsamples(buf)
    self.gotsamples(zeros)
    self.process()

    if self.oneret != None:
      self.oneret[0] -= len(zeros)
      
    return self.oneret

  def gofile(self, filename):
    self.openwav(filename)
    while True:
      buf = self.readwav()
      if buf.size < 1:
        break
      self.gotsamples(buf)
      self.process()

  def opencard(self, desc):
      self.audio = weakaudio.open(desc, self.rate)

  def gocard(self):
    while True:
      [ buf, buf_time ] = self.audio.read()
      if len(buf) > 0:
        self.gotsamples(buf)
        self.process()
      else:
        time.sleep(0.2)
