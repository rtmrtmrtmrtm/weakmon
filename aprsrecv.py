#!/usr/local/bin/python

#
# sound-card APRS decoder
#
# Robert Morris, AB1HL
#

import numpy
import wave
import weakaudio
import weakutil
import time
import scipy
import sys
import os
import math
from scipy.signal import lfilter, filtfilt
import numpy.lib.stride_tricks

# optimizable tuning parameters.
smoothwindow = 2.0 # symbols, 1.0 0.8 0.7 1.7(for hamming smoother)
slicewindow = 25.0 # symbols, 20 30 20
tonegain = 2.0     # 2.0 (useful for track 02)
advance = 8.0      # symbols 1.0 8.0

# http://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
def smooth(values, window):
  #weights = numpy.repeat(1.0, window)/window
  weights = numpy.hamming(window)
  sma = numpy.convolve(values, weights, 'valid')
  sma = sma[0:len(values)]
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

# https://witestlab.poly.edu/blog/capture-and-decode-fm-radio/
def deemphasize(samples, rate):
  d = rate * 750e-6   # Calculate the # of samples to hit the -3dB point  
  x = numpy.exp(-1/d)   # Calculate the decay between each sample  
  b = [1-x]          # Create the filter coefficients  
  a = [1,-x]  
  out = scipy.signal.lfilter(b,a,samples) 
  return out

class APRSRecv:

  def __init__(self):
    self.rate = None

    # Bell-202
    self.baud = 1200 # bits per second
    self.mark = 1200
    self.space = 2200

    self.off = 0
    self.raw = numpy.array([0])
    self.flagpat = None

  def openwav(self, filename):
    self.wav = wave.open(filename)
    self.wav_channels = self.wav.getnchannels()
    self.wav_width = self.wav.getsampwidth()
    self.rate = self.wav.getframerate()
    if False:
      sys.stdout.write("file=%s chans=%d width=%d rate=%d\n" % (filename,
                                                                self.wav_channels,
                                                                self.wav_width,
                                                                self.rate))

  def readwav(self):
    z = self.wav.readframes(4096)
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
    self.raw = numpy.append(self.raw, buf)

  # slice one tone, yielding a running <0 for low and >0 for high.
  # slicing level is a running midway between local min and max.
  # don't use average since mark and space aren't equally popular.
  # already filtered/smoothed so no point in using percentiles.
  def sliceone(self, smoothed):
      global slicewindow

      bitsamples = int(self.rate / float(self.baud))

      win = int(slicewindow * bitsamples)

      # average just to the right at start of packet,
      # and just to the left towards end of packet.
      # by inserting sliceshift samples from the packet.
      # to avoid averaging in a lot of non-packet noise.
      # this is important for packets that end with a
      # single flag and that are followed by lots of noise
      # (happens a lot in track 02).
      sliceshift = int(win/2 + bitsamples)

      z = numpy.concatenate((smoothed[0:win],
                             smoothed[win:win+sliceshift],
                             smoothed[win:win+64*bitsamples],
                             smoothed[win+64*bitsamples:win+64*bitsamples+sliceshift],
                             smoothed[win+64*bitsamples:]))

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

      if len(maxv) < len(smoothed):
        maxv = numpy.append(maxv, maxv[0:(len(smoothed)-len(maxv))])
        minv = numpy.append(minv, minv[0:(len(smoothed)-len(minv))])
      elif len(maxv) > len(smoothed):
        maxv = maxv[0:len(smoothed)]
        minv = minv[0:len(smoothed)]

      if False:
        # agc -- normalize so that min..max is -0.5..0.5
        # XXX this does not help.
        sliced = numpy.subtract(smoothed, minv)
        sliced = numpy.divide(sliced, maxv - minv)
        sliced = sliced - 0.5
        return sliced
      else:
        midv = (maxv + minv) / 2.0
        sliced = numpy.subtract(smoothed, midv)
        return sliced

  # correlate against a tone (1200 or 2200 Hz).
  # idea from Sivan Toledo's QEX article.
  # (used to use butterworth bandpass filters of order 3
  #  and width 1100 hz, but the following is slightly better).
  def corr(self, samples, tone):
      global smoothwindow
      win = int(smoothwindow * self.rate / self.baud)
      xsin = weakutil.sintone(self.rate, tone, len(samples))
      xcos = weakutil.costone(self.rate, tone, len(samples))
      c = numpy.sqrt(numpy.add(numpy.square(smooth(xsin * samples, win)),
                               numpy.square(smooth(xcos * samples, win))))
      return c

  # correlate, slice, generate +/- for each sample.
  def slice(self, samples):
    markcorr = self.corr(samples, self.mark)
    spacecorr = self.corr(samples, self.space)
    m1 = self.sliceone(markcorr)
    s1 = self.sliceone(spacecorr)
    return numpy.subtract(m1, s1)

  def process(self, eof):
    global tonegain, advance

    bitsamples = self.rate / float(self.baud)
    flagsamples = bitsamples * 9 # HDLC 01111110 flag (9 b/c NRZI)
    maxpacket = 340 * 8 * bitsamples # guess at number of samples if longest possible packet

    if self.raw.size < maxpacket and eof == False:
      return

    # set up to try multiple emphasis setups.
    sliced = [ ]

    # no change in emphasis; best when receiver doesn't de-emph,
    # but not right for senders that pre-emph.
    sliced.append( self.slice(self.raw) )

    # de-emphasize, for a receiver that doesn't de-emph,
    # but senders that do.
    # doesn't seem to help for track 01...
    #sliced.append( self.slice(deemphasize(self.raw, self.rate)) )

    while self.raw.size >= maxpacket or (eof and self.raw.size > 20*bitsamples):
      # sliced[0] is a candidate for start of packet,
      # i.e. first sample of first flag.

      bestok = 0
      bestmsg = None
      bestnsymbols = 0
      beststart = 0

      for sl in sliced:
        [ ok, msg, nsymbols, start ] = self.process1(sl)
        if ok > bestok:
          bestok = ok
          bestmsg = msg
          bestnsymbols = nsymbols
          beststart = self.off + start 

      if bestok > 0 and self.callback:
        # compute space-to-mark tone strength ratio, to help understand emphasis.
        # space is 2200 hz, mark is 1200 hz.
        start = beststart - self.off

        #indices = numpy.arange(0, bestnsymbols*bitsamples, bitsamples)
        #indices = indices + (start + 0.5*bitsamples)
        #indices = numpy.rint(indices).astype(int)
        #rawsymbols = sliced[0][indices]
        #rawmark = markcorr[indices]
        #rawspace = spacecorr[indices]
        #meanmark = numpy.mean(numpy.where(rawsymbols > 0, rawmark, 0))
        #meanspace = numpy.mean(numpy.where(rawsymbols <= 0, rawspace, 0))
        #ratio = meanspace / meanmark
        ratio = 1.0

        self.callback(bestok, bestmsg, beststart, ratio)
        sys.stdout.flush()

      if bestok == 2:
        trim = int(bestnsymbols * bitsamples) # skip packet
      else:
        trim = int(advance * bitsamples) # skip a few symbols

      self.off += trim
      self.raw = self.raw[trim:]
      #markcorr = markcorr[trim:] # just for debug
      #spacecorr = spacecorr[trim:] # just for debug
      for i in range(0, len(sliced)):
        sliced[i] = sliced[i][trim:]

  # does a packet start at sliced[0:] ?
  # sliced[] likely has far more samples than needed.
  # returns [ ok, msg, nsymbols, flagstart ]
  # flag starts at sliced[flagstart].
  def process1(self, sliced):
    global advance

    bitsamples = self.rate / float(self.baud)
    flagsamples = bitsamples * 9 # HDLC 01111110 flag (9 b/c NRZI)

    ff = self.findflag(sliced[0:int(round(flagsamples+advance*bitsamples+2))])
    if ff != None:
      indices = numpy.arange(0, len(sliced) - (ff+2*bitsamples), bitsamples)
      indices = indices + (ff + 0.5*bitsamples)
      indices = numpy.rint(indices).astype(int)
      rawsymbols = sliced[indices]
      symbols = numpy.where(rawsymbols > 0, 1, -1)

      [ ok, msg, nsymbols ] = self.finishframe(symbols[8:])
      if ok >= 1:
        return [ ok, msg, nsymbols, ff ]

    return [ 0, None, 0, 0 ]

  def dump(self, mark, space, sliced, raw):
    f = open("mark", "w")
    numpy.savetxt(f, mark, "%f")
    f.close()
    f = open("space", "w")
    numpy.savetxt(f, space, "%f")
    f.close()
    f = open("sliced", "w")
    numpy.savetxt(f, sliced, "%f")
    f.close()
    f = open("raw", "w")
    numpy.savetxt(f, raw, "%f")
    f.close()

    # sample points
    bitsamples = self.rate / float(self.baud)
    f = open("points", "w")
    off = 0 + 0.5*bitsamples
    while off < len(sliced):
      f.write("%f 0\n" % (off-1))
      if sliced[int(off)] > 0:
        f.write("%f 1\n" % (off))
      else:
        f.write("%f -1\n" % (off))
      f.write("%f 0\n" % (off+1))
      off += bitsamples # critical that it's floating point
    f.close()

  # symbols[0] contains -1=space 1=mark.
  # symbols[0] should be the last symbol of the hdlc flag.
  # there are likely many excess symbols, so need to look for the end flag.
  # decode NRZI.
  # de-bit-stuff.
  # returns [ ok, msg, nsymbols ]
  # ok is 0/1/2 for no, bad crc, good crc.
  def finishframe(self, symbols):
    # look for flag at end.
    flagcorr = numpy.correlate(symbols, [-1, 1, 1, 1, 1, 1, 1, 1, -1])
    cimax = numpy.argmax(flagcorr) # index of first flag
    cimin = numpy.argmin(flagcorr) # index of first inverted flag
    if flagcorr[cimax] == 9 and flagcorr[cimin] == -9:
      # they are both proper flags
      ci = min(cimax, cimin)
    elif flagcorr[cimax] == 9:
      ci = cimax
    else:
      ci = cimin

    symbols = symbols[0:ci+1]

    # un-nrzi symbols to get bits.
    bits = numpy.where(numpy.equal(symbols[:-1], symbols[1:]), 1, 0)

    # un-bit-stuff. every sequence of five ones must be followed
    # by a zero, which we should delete.
    nones = 0
    nbits = [ ]
    for i in range(0, len(bits)):
      if nones == 5:
        nones = 0
        # assuming bits[i] == 0
        # don't append the zero...
      elif bits[i] == 1:
        nones += 1
        nbits.append(bits[i])
      else:
        nbits.append(bits[i])
        nones = 0
    bits = nbits

    if len(bits) < 8:
      return [ 0, None, 0 ]

    # convert bits to bytes.
    # bits[] is least-significant-first, but
    # numpy.packbits() wants MSF.
    bits = numpy.array(bits)
    bits = bits[0:(len(bits)/8)*8]
    assert (len(bits)%8) == 0
    bits = bits[::-1]
    bytes = numpy.packbits(bits)
    bytes = bytes[::-1]

    msg = None

    ok = self.checkpacket(bytes)
    if ok > 0 or len(bytes) > 16:
      msg = self.printpacket(bytes)

    return [ ok, msg, len(symbols) ]

  # 0: syntactically unlikely to be a packet
  # 1: syntactically plausible but crc failed
  # 2: crc is correct
  def checkpacket(self, bytes):
    debug = False

    if len(bytes) < 18 or len(bytes) > 500:
      if debug:
        print "bad len %d" % (len(bytes))
      return 0

    crc = crc16(bytes[0:-2])
    crc ^= 0xffff
    # last two packet bytes are crc-low crc-high
    if bytes[-2] == (crc & 0xff) and bytes[-1] == ((crc >> 8) & 0xff):
      # CRC is good.
      return 2
    
    i = 0

    # addresses
    addrsdone = False
    while i+7 < len(bytes) and addrsdone == False:
      for j in range(i, i+6):
        if bytes[j] & 1:
          if debug:
            print "early address termination"
          return 0
      i += 6
      ssid = bytes[i]
      i += 1
      x = (ssid >> 1) & 0xf
      if ssid & 1:
        addrsdone = True
    if addrsdone == False:
      if debug:
        print "no address termination"
      return 0

    if i + 4 > len(bytes):
      if debug:
        print "too short"
      return 0
    if bytes[i] != 0x03:
      if debug:
        print "control not 0x03"
      return 0
    if bytes[i+1] != 0xf0:
      if debug:
        print "PID not 0xf0"
      return 0
    i += 2

    return 1

  # the standard format:
  # SRC>DST,PATH:msg
  # KB1LNA-4>T2RP4R,WIDE1-1,WIDE2-1:`c&nl"4>/'"4]}|![&4'C|!wwa!|3
  def printpacket(self, bytes):
    i = 0

    #msg = ""

    addrs = [ ]
    lastH = -1 # last relay with H bit

    # addresses
    addrsdone = False
    addrnum = 0 # 0=dst, 1=src, 2...=relays
    while i+7 < len(bytes) and addrsdone == False:
      addr = ""
      for j in range(i, i+6):
        x = bytes[j] >> 1
        if x > 32 and x < 127:
          addr += "%c" % (chr(x))
        elif x == 32:
          pass
        else:
          addr += "."
      i += 6
      ssid = bytes[i]
      i += 1
      x = (ssid >> 1) & 0xf
      if x > 0:
        addr += "-%d" % (x)
      if addrnum > 1 and (ssid & 0x80) != 0:
          # H (has-been-repeated)
          #addr += "*"
          lastH = len(addrs)
      if ssid & 1:
        addrsdone = True
      addrs.append(addr)
      addrnum += 1

    if len(addrs) >= 2:
      msg = "%s>%s" % (addrs[1], addrs[0])
    elif len(addrs) == 1:
      msg = "%s>???" % (addrs[0])
    else:
      msg = "???"
    for ai in range(2, len(addrs)):
      msg += ",%s" % (addrs[ai])
      if ai == lastH:
        msg += "*"

    msg += ":"

    if i + 2 < len(bytes):
      # sys.stdout.write("ctl=%02x pid=%02x " % (bytes[i], bytes[i+1]))
      i += 2

    if True:
      while i < len(bytes) - 2:
        c = bytes[i]
        if c >= ord(' ') and c < 0x7f:
            msg += chr(c)
        else:
            msg += "<0x%02x>" % (c)
        i = i + 1

    # remaining 3 bytes are FCS and flag

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
    
    # pre-compute self.flagsamples
    if self.flagpat == None:
      self.flagpat = [ -1, 1, 1, 1, 1, 1, 1, 1, -1 ]
      a = numpy.array([])
      for i in range(0, len(self.flagpat)):
        ai = int(round((i + 0.5) * bitsamples))
        a = numpy.append(a, numpy.zeros(ai - len(a) + 2))
        a[ai-1] = self.flagpat[i] / 2.0
        a[ai] = self.flagpat[i]
        a[ai+1] = self.flagpat[i] / 2.0
      self.flagsamples = a

    cc = numpy.correlate(sliced, self.flagsamples)
    mm1 = numpy.argmax(cc)
    mm2 = numpy.argmin(cc)
    #print "len(sliced) %d len(a) %d len(cc) %d; m1=%d m2=%d" % (len(sliced), len(a), len(cc), mm1, mm2)

    # check that the samples really do form a flag.
    if abs(cc[mm1]) > abs(cc[mm2]):
      mul = 1
      start = mm1
    else:
      mul = -1
      start = mm2
    score = 0
    for i in range(0, len(self.flagpat)):
      ai = (i + 0.5) * bitsamples
      x = sliced[int(round(ai+start))] * mul
      if((self.flagpat[i] < 0 and x > 0) or
         (self.flagpat[i] > 0 and x < 0)):
         pass
      else:
        score += 1

    if score >= 8:
      return start
    else:
      return None

  def gofile(self, filename):
    self.openwav(filename)
    while True:
      buf = self.readwav()
      if buf.size < 1:
        break
      self.gotsamples(buf)
      self.process(False)
    self.process(True)

  def opencard(self, desc):
      self.audio = weakaudio.new(desc, None)
      self.rate = self.audio.rate

  def gocard(self):
    while True:
      [ buf, buf_time ] = self.audio.read()
      if len(buf) > 0:
        self.gotsamples(buf)
        self.process(False)
      # sleep so that samples accumulate, which makes
      # resample() higher quality.
      time.sleep(0.2)

wins = 0

# process a .wav file, return number of correct CRCs.
def benchmark(wavname, verbose):
  global wins
  wins = 0
  def cb(fate, msg, start, space_to_mark):
    global wins
    if fate == 2:
      wins = wins + 1
  ar = APRSRecv()
  ar.callback = cb
  ar.gofile(wavname)
  if verbose:
    print "%d %s" % (wins, wavname)
  return wins

def optimize(wavname):
    vars = [
      [ "smoothwindow", [ 1.5, 1.6, 1.7, 1.8, 2.0, 2.2 ] ],
      [ "slicewindow", [ 20, 25, 30, 40 ] ],
      [ "tonegain", [ 1.5, 2.0, 2.5, 3.0, 4.0 ] ],
      # [ "advance", [ 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0 ] ],
    ]

    sys.stdout.write("# ")
    for v in vars:
        sys.stdout.write("%s=%s " % (v[0], eval(v[0])))
    sys.stdout.write("\n")
    sys.stdout.flush()

    # warm up any caches, JIT, &c.
    ar = APRSRecv()
    ar.callback = None
    ar.gofile(wavname)

    for v in vars:
        for val in v[1]:
            old = None
            if "." in v[0]:
                xglob = ""
            else:
                xglob = "global %s ; " % (v[0])
            exec "%sold = %s" % (xglob, v[0])
            exec "%s%s = %s" % (xglob, v[0], val)

            sc = benchmark(wavname, False)
            exec "%s%s = old" % (xglob, v[0])
            sys.stdout.write("%s=%s : " % (v[0], val))
            sys.stdout.write("%d\n" % (sc))
            sys.stdout.flush()

def cb(fate, msg, start, space_to_mark):
    # fate=0 -- unlikely to be correct.
    # fate=1 -- CRC failed but syntax look OK.
    # fate=2 -- CRC is correct.
    if fate >= 1:
        print "%d %d %.1f %s" % (fate, start, space_to_mark, msg)

def usage():
  sys.stderr.write("Usage: aprsrecv.py [file...] [-sound]\n")
  sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "-sound":
            ar = APRSRecv()
            ar.callback = cb
            ar.gocard()
        elif a == "-bench":
            benchmark(sys.argv[i+1], True)
            i += 1
        elif a == "-optim":
            optimize(sys.argv[i+1])
            i += 1
        else:
            ar = APRSRecv()
            ar.callback = cb
            ar.gofile(sys.argv[i])
        i += 1

if __name__ == '__main__':
    if False:
        pfile = "cprof.out"
        sys.stderr.write("aprs: cProfile -> %s\n" % (pfile))
        import cProfile
        import pstats
        cProfile.run('main()', pfile)
        p = pstats.Stats(pfile)
        p.strip_dirs().sort_stats('time')
        # p.print_stats(10)
        p.print_callers()
    else:
        main()
