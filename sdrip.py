#
# Control an RFSpace SDR-IP (and probably a NetSDR).
#

import socket
import sys
import os
import numpy
import scipy
import scipy.signal
import thread
import threading
import time
import struct

def x8(x):
  s = chr(x & 0xff)
  return s

def x16(x):
  s = ""
  s += chr(x & 0xff) # least-significant first
  s += chr((x >> 8) & 0xff)
  return s

def x32(x):
  s = ""
  s += chr(x & 0xff) # least-significant first
  s += chr((x >> 8) & 0xff)
  s += chr((x >> 16) & 0xff)
  s += chr((x >> 24) & 0xff)
  return s

# 40-bit frequency in Hz, lsb first
# but argument must be an int
def x40(hz):
  s = ""
  for i in range(0, 5):
    s = s + chr(hz & 0xff)
    hz >>= 8
  return s

# turn a string into hex digits
def hx(s):
  buf = ""
  for i in range(0, len(s)):
    buf += "%02x " % (ord(s[i]))
  return buf

def butter_lowpass(cut, samplerate, order=5):
  nyq = 0.5 * samplerate
  cut = cut / nyq
  b, a = scipy.signal.butter(order, cut, btype='lowpass')
  return b, a

# IQ -> USB
def iq2usb(iq):
    ii = iq.real
    qq = iq.imag
    ii = numpy.real(scipy.signal.hilbert(ii)) # delay to match hilbert(Q)
    qq = numpy.imag(scipy.signal.hilbert(qq))
    ssb = numpy.subtract(ii, qq) # usb from phasing method
    #ssb = numpy.add(ii, qq) # lsb from phasing method
    assert len(iq) == len(ssb)
    return ssb

mu = thread.allocate_lock()

#
# if already connected, return existing SDRIP,
# otherwise a new one.
#
sdrips = { }
def open(ipaddr):
    global sdrips, mu
    mu.acquire()
    if not (ipaddr in sdrips):
        sdrips[ipaddr] = SDRIP(ipaddr)
    sdr = sdrips[ipaddr]
    mu.release()
    return sdr

class SDRIP:
  # SDR-IP's IP address
  ipaddr = None
  
  # data UDP socket
  ds = None

  # control TCP socket
  cs = None

  # sample rate
  rate = None

  # frequency in Hz
  frequency = None

  # 16 or 24
  # only 24 seems useful
  samplebits = 24

  # iq? i think only True works.
  iq = True

  # optionally record/playback iq in a file
  iqout = None
  iqin = None

  nextseq = 0
  
  def __init__(self, ipaddr):
    # ipaddr is SDR-IP's IP address e.g. "192.168.3.123"
    self.ipaddr = ipaddr
    self.mu = thread.allocate_lock()

    if ipaddr != None:
      self.connect()

  def connect(self):
    # allocate a UDP socket and port for incoming data from the SDR-IP.
    self.ds = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.ds.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
    self.ds.bind(('', 0)) # ask kernel to choose a free port
    hostport = self.ds.getsockname() # hostport[1] is port number

    # commands over TCP to port 50000
    self.cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.cs.connect((self.ipaddr, 50000))

    # fork() a sub-process to read the data UDP socket, since
    # the Python thread scheduler doesn't run us often enough
    # if WSPR is compute-bound for tens of seconds.
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        self.reader(w)
        os._exit(0)
    else:
        self.pipe = r
        os.close(w)

    # tell the SDR-IP where to send UDP packets
    self.setudp(hostport[1])

    # boilerplate
    self.setad()
    self.setfilter()
    #self.setgain(0)
    self.setgain(-20)

  def reader1(self):
    while True:
        buf = self.ds.recv(4096)
        self.packets_mu.acquire()
        self.packets.append(buf)
        self.packets_mu.release()

  # read the data UDP socket in a separate process and
  # send the results on the pipe w.
  def reader(self, w):
      ww = os.fdopen(w, 'wb')

      # spawn a thread that just keeps reading from the socket
      # and appending packets to packets[].

      self.packets = [ ]
      self.packets_mu = thread.allocate_lock()

      th = threading.Thread(target=lambda : self.reader1())
      th.daemon = True
      th.start()

      # move packets from packets[] to the UNIX pipe.
      # the pipe write() calls may block, but it's OK because
      # the reader1() thread keeps draining the UDP socket.
      while True:
          self.packets_mu.acquire()
          ppp = self.packets
          self.packets = [ ]
          self.packets_mu.release()

          if len(ppp) < 1:
              time.sleep(0.005) # we expect 100 pkts/second

          for pkt in ppp:
              try:
                  ww.write(struct.pack('I', len(pkt)))
                  ww.write(pkt)
                  ww.flush()
              except:
                  os._exit(1)

  # read a 16-bit int from TCP control socket
  def read16(self):
    x0 = self.cs.recv(1) # least-significant byte
    x1 = self.cs.recv(1) # most-significant byte
    return (ord(x0) & 0xff) | ((ord(x1) << 8) & 0xff00)

  # read a reply from the TCP control socket
  # return [ type, item, data ]
  def readctl(self):
    len = self.read16() # overall length and msg type
    mtype = (len >> 13) & 0x7
    len &= 0x1fff
    item = self.read16() # control item
    data = ""
    xlen = len - 4
    while xlen > 0:
      dd = self.cs.recv(1)
      data += dd
      xlen -= 1
    return [ mtype, item, data ]

  # read tcp control socket until we get a reply
  # that matches type and item.
  def readreply(self, mtype, item):
    while True:
      reply = self.readctl()
      if reply[0] == 0 and reply[1] == item:
        return reply[2]
      if reply[0] == 1 and reply[1] == 5:
        # A/D overload
        continue
      if reply[0] != 0:
        sys.stderr.write("readreply oops1 %d %d\n" % (mtype, reply[0]))
      if reply[1] != item:
        sys.stderr.write("readreply oops2 wanted=%04x got=%04x\n" % (item,
                                                                   reply[1]))
      # print "reply: %04x %s" % (reply[1], hx(reply[2]))
    sys.exit(1)

  # send a Request Control Item, wait for and return the result
  def getitem(self, item):
    mtype = 1 # type=request control item
    buf = ""
    buf += x8(4) # overall length, lsb
    buf += x8((mtype << 5) | 0) # 0 is len msb
    buf += x16(item)
    self.cs.send(buf)
    return self.readreply(mtype, item)

  def setitem(self, item, data):
    self.mu.acquire()
    mtype = 0 # set item
    lx = 4 + len(data)
    buf = ""
    buf += x8(lx)
    buf += x8((mtype << 5) | 0)
    buf += x16(item)
    buf += data
    self.cs.send(buf)
    ret = self.readreply(mtype, item)
    self.mu.release()
    return ret

  # set Frequency
  def setfreq1(self, display, hz):
    hz = int(hz)
    data = ""
    data += chr(display) # 1=display, 0=actual receiver DDC
    data += x40(hz)
    self.setitem(0x0020, data)

  def setfreq(self, hz):
    if self.ipaddr != None:
      self.setfreq1(0, hz) # DDC
      self.setfreq1(1, hz) # display
    self.frequency = hz

  # set Receiver State to Run
  # only I/Q seems to work, not real.
  def setrun(self):
    data = ""
    if self.iq:
      data += x8(0x80) # 0x80=I/Q, 0x00=real
    else:
      data += x8(0x00) # 0x80=I/Q, 0x00=real
    data += x8(0x02) # 1=idle, 2=run
    if self.samplebits == 16:
      data += x8(0x00) # 80=24 bit continuous, 00=16 bit continuous
    else:
      data += x8(0x80) # 80=24 bit continuous, 00=16 bit continuous
    data += x8(0x00) # unused
    self.setitem(0x0018, data)
    self.nextseq = 0

  # stop receiver
  def stop(self):
    data = ""
    if self.iq:
      data += x8(0x80) # 0x80=I/Q, 0x00=real
    else:
      data += x8(0x00) # 0x80=I/Q, 0x00=real
    data += x8(0x01) # 1=idle, 2=run
    if self.samplebits == 16:
      data += x8(0x00) # 80=24 bit continuous, 00=16 bit continuous
    else:
      data += x8(0x80) # 80=24 bit continuous, 00=16 bit continuous
    data += x8(0x00) # unused
    self.setitem(0x0018, data)

  # DDC Output Sample Rate
  # rate is samples/second
  # must be an integer x4 division of 80 million.
  # the minimum is 32000.
  def setrate(self, rate):
    self.rate = rate
    if self.ipaddr != None:
      data = ""
      data += x8(0) # ignored
      data += x32(rate)
      self.setitem(0x00B8, data)

  # A/D Modes
  # always sets dither and A/D gain 1.5
  def setad(self):
    data = ""
    data += x8(0) # ignored
    #data += x8(0x3) # bit zero is dither, bit 1 is A/D gain 1.5
    data += x8(0x1) # bit zero is dither, bit 1 is A/D gain 1.5
    self.setitem(0x008A, data)

  # RF Filter Select
  # always sets automatic
  # 0=automatic
  # 11=bypass
  def setfilter(self):
    data = ""
    data += x8(0) # ignored
    data += x8(0) # automatic
    self.setitem(0x0044, data)

  # RF Gain
  # gain is 0, -10, -20 -30 dB
  def setgain(self, gain):
    if self.ipaddr != None:
      data = ""
      data += x8(0) # ignored
      data += x8(gain)
      self.setitem(0x0038, data)

  # Data Output UDP IP and Port Address
  # just set the port, not the host address.
  def setudp(self, port):
    # find host's IP address.
    hostport = self.cs.getsockname()
    ipaddr = socket.inet_aton(hostport[0]) # yields a four-byte string, wrong order

    data = ""
    data += ipaddr[3]
    data += ipaddr[2]
    data += ipaddr[1]
    data += ipaddr[0]
    data += x16(port)

    self.setitem(0x00C5, data)

  # wait for and decode a UDP packet of I/Q samples.
  # returns a buffer with interleaved I and Q.
  # return an array of complex (real=I, imag=Q).
  def readiq(self):
    if self.iqin != None:
      samples = numpy.fromfile(self.iqin, dtype=numpy.float64, count=512)
      ii1 = samples[0::2]
      qq1 = samples[1::2]
      cc1 = ii1 + 1j*qq1
      return cc1

    # read from the pipe; a 4-byte length, then the packet.
    #buf = self.ds.recv(4096)
    x4 = os.read(self.pipe, 4)
    if len(x4) != 4:
        sys.stderr.write("sdrip read from child failed\n")
        os._exit(1)
    [plen] = struct.unpack("I", x4)
    assert plen > 0 and plen < 65536
    buf = ""
    while len(buf) < plen:
        x = os.read(self.pipe, plen - len(buf))
        buf = buf + x

    # parse SDR-IP header into length, msg type
    lx = ord(buf[0])
    lx |= (ord(buf[1]) << 8)
    mtype = (lx >> 13) & 0x7 # 0x4 is data
    lx &= 0x1fff # should == len(buf)
    # print "%d %d %x" % (len(buf), lx, mtype)

    # packet sequence number (wraps to 1, not 0)
    seq = ord(buf[2]) | (ord(buf[3]) << 8)
    gap = 0
    if seq != self.nextseq and (seq != 1 or self.nextseq != 65536):
      sys.stderr.write("seq oops got=%d wanted=%d\n" % (seq, self.nextseq))
      if seq > self.nextseq:
        gap = seq - self.nextseq
    self.nextseq = seq + 1

    if self.samplebits == 16:
      samples = numpy.fromstring(buf[4:], dtype=numpy.int16)
    else:
      s8 = numpy.fromstring(buf[4:], dtype=numpy.uint8)
      x0 = s8[0::3]
      x1 = s8[1::3]
      x2 = s8[2::3]
      # top 8 bits, sign-extended from x2
      high = numpy.greater(x2, 127)
      x3 = numpy.where(high,
                       numpy.repeat(255, len(x2)),
                       numpy.repeat(0, len(x2)))

      z = numpy.empty([len(x0)*4], dtype=numpy.uint8)
      z[0::4] = x0
      z[1::4] = x1
      z[2::4] = x2
      z[3::4] = x3
      zz = z.tostring()
      #s32 = numpy.fromstring(zz, dtype=numpy.int32)
      #samples = s32.astype(numpy.int16)
      samples = numpy.fromstring(zz, dtype=numpy.int32)

    samples = samples.astype(numpy.float64)

    if gap > 0:
      samples = numpy.append(numpy.zeros(len(samples)*gap,
                                         dtype=numpy.float64),
                             samples)

    if self.iqout != None:
      samples.tofile(self.iqout)

    ii1 = samples[0::2]
    qq1 = samples[1::2]
    cc1 = ii1 + 1j*qq1
    return cc1

  #
  # read from SDR-IP, demodulate as USB.
  #
  def readusb(self):
    iq = self.readiq()
    usb = iq2usb(iq)
    return usb
