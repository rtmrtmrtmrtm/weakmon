#
# Control an RFSpace SDR-IP, NetSDR, or CloudIQ.
#
# Example:
#   sdr = sdrip.open("192.168.3.125")
#   sdr.setrate(32000)
#   sdr.setgain(-10)
#   sdr.setrun()
#   while True:
#     buf = sdr.readiq()
#     OR buf = sdr.readusb()
#
# Robert Morris, AB1HL
#

import socket
import sys
import os
import numpy
import scipy
import scipy.signal
import threading
import time
import struct
import weakutil

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

def y16(s):
    x = (ord(s[0]) + 
         (ord(s[1]) << 8))
    return x

def y32(s):
    x = (ord(s[0]) + 
         (ord(s[1]) << 8) +
         (ord(s[2]) << 16) +
         (ord(s[3]) << 24))
    return x

# turn 5 bytes from NetSDR into a 40-bit number.
# LSB first.
def y40(s):
    hz = (ord(s[0]) + 
          (ord(s[1]) << 8) +
          (ord(s[2]) << 16) +
          (ord(s[3]) << 24) +
          (ord(s[4]) << 32))
    return hz

# turn a string into hex digits
def hx(s):
  buf = ""
  for i in range(0, len(s)):
    buf += "%02x " % (ord(s[i]))
  return buf

mu = threading.Lock()

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
  
  def __init__(self, ipaddr):
    # ipaddr is SDR-IP's IP address e.g. "192.168.3.123"
    self.mode = "usb"
    self.ipaddr = ipaddr
    self.mu = threading.Lock()

    self.rate = None
    self.frequency = None
    self.running = False

    # 16 or 24
    # only 24 seems useful
    self.samplebits = 24

    # iq? i think only True works.
    self.iq = True

    self.nextseq = 0
    self.reader_pid = None

    self.connect()

  # "usb" or "fm"
  # maybe only here to be ready by weakaudio.py/SDRIP.
  def set_mode(self, mode):
      self.mode = mode

  def connect(self):
    # allocate a UDP socket and port for incoming data from the SDR-IP.
    self.ds = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.ds.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
    self.ds.bind(('', 0)) # ask kernel to choose a free port
    hostport = self.ds.getsockname() # hostport[1] is port number

    # fork() a sub-process to read and buffer the data UDP socket,
    # since the Python thread scheduler doesn't run us often enough if
    # WSPR is compute-bound in numpy for tens of seconds.
    r, w = os.pipe()
    self.reader_pid = os.fork()
    if self.reader_pid == 0:
        os.close(r)
        self.reader(w)
        os._exit(0)
    else:
        self.pipe = r
        os.close(w)
        self.ds.close()

    # commands over TCP to port 50000
    self.cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.cs.connect((self.ipaddr, 50000))

    # tell the SDR-IP where to send UDP packets
    self.setudp(hostport[1])

    # boilerplate
    self.setad()
    self.setfilter()
    self.setgain(0)
    #self.setgain(-20)

    # keep reading the control TCP socket, to drain any
    # errors, so NetSDR doesn't get upset.
    th = threading.Thread(target=lambda : self.drain_ctl())
    th.daemon = True
    th.start()

    # "SDR-IP"
    #print "name: %s" % (self.getitem(0x0001))

    # option 0x02 means reflock board is installed
    #oo = self.getitem(0x000A) # Options
    #oo0 = ord(oo[0])
    #print "options: %02x" % (oo0)

    if False:
        # set calibration.
        # 192.168.3.130 wants + 506
        # 192.168.3.131 wants + 525
        # (these are with the 10 mhz reflock ocxo, but not locked)
        data = ""
        data += x8(0) # ignored
        if self.ipaddr == "192.168.3.130":
            data += x32(80000000 + 506)
        elif self.ipaddr == "192.168.3.131":
            data += x32(80000000 + 525)
        else:
            print "sdrip.py: unknown IP address %s for calibration" % (self.ipaddr)
            # data += x32(80000000 + 0)
            data = None
        if data != None:
            self.setitem(0x00B0, data)

    # A/D Input Sample Rate Calibration
    # factory set to 80000000
    x = self.getitem(0x00B0)
    cal = y32(x[1:5])
    print "sdrip %s cal: %s" % (self.ipaddr, cal)

  # read the UDP socket from the SDR-IP.
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
      self.packets_mu = threading.Lock()

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
              # we expect 100 pkts/second
              # but OSX seems to limit a process to 150 wakeups/second!
              # time.sleep(0.005)
              time.sleep(0.01)

          for pkt in ppp:
              try:
                  ww.write(struct.pack('I', len(pkt)))
                  ww.write(pkt)
                  ww.flush()
              except:
                  #sys.stderr.write("sdrip: pipe write failed\n")
                  os._exit(1)

  # consume unsolicited messages from the NetSDR,
  # and notice if it goes away.
  def drain_ctl(self):
      try:
          while True:
              self.getitem(0x0001) # name
              time.sleep(1)
      except:
          pass
      sys.stderr.write("sdrip: control connection died\n")
      os.kill(self.reader_pid, 9)

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
    if len == 2:
        # NAK -- but for what?
        sys.stderr.write("sdrip: NAK\n")
        return None
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
      if reply == None:
          # NAK
          return None
      if reply[0] == 0 and reply[1] == item:
        return reply[2]
      # sys.stderr.write("sdrip: unexpected mtype=%02x item=%04x datalen=%d\n" % (reply[0], reply[1], len(reply[2])))
      if reply[0] == 1 and reply[1] == 5:
        # A/D overload
        # sys.stderr.write("sdrip: unsolicited A/D overload\n")
        continue
      if reply[0] != 0:
        sys.stderr.write("sdrip: readreply oops1 %d %d\n" % (mtype, reply[0]))
      if reply[1] != item:
        sys.stderr.write("sdrip: readreply oops2 wanted=%04x got=%04x\n" % (item,
                                                                   reply[1]))
      # print "reply: %04x %s" % (reply[1], hx(reply[2]))
    sys.exit(1)

  # send a Request Control Item, wait for and return the result
  def getitem(self, item, extra=None):
    self.mu.acquire()
    mtype = 1 # type=request control item
    buf = ""
    buf += x8(4) # overall length, lsb
    buf += x8((mtype << 5) | 0) # 0 is len msb
    buf += x16(item)
    if extra != None:
        buf += extra
    self.cs.send(buf)
    ret = self.readreply(mtype, item)
    self.mu.release()
    return ret

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

  def print_setup(self):
      print(("freq 0: %d" % (self.getfreq(0)))) # 32770 if down-converting
      print(("name: %s" % (self.getname())))
      print(("serial: %s" % (self.getserial())))
      print(("interface: %d" % (self.getinterface())))
      # print "boot version: %s" % (self.getversion(0))
      # print "application firmware version: %s" % (self.getversion(1))
      # print "hardware version: %s" % (self.getversion(2))
      # print "FPGA config: %s" % (self.getversion(3))
      print(("rate: %d" % (self.getrate())))
      print(("freq 0: %d" % (self.getfreq(0)))) # 32770 if down-converting
      print(("A/D mode: %s" % (self.getad(0))))
      print(("filter: %d" % (self.getfilter(0))))
      print(("gain: %d" % (self.getgain(0))))
      print(("fpga: %s" % (self.getfpga())))
      print(("scale: %s" % (self.getscale(0))))
      # print "downgain: %s" % (self.getdowngain())

  # set Frequency
  def setfreq1(self, chan, hz):
    hz = int(hz)
    data = ""
    data += chr(chan) # 1=display, 0=actual receiver DDC
    data += x40(hz)
    self.setitem(0x0020, data)

  def setfreq(self, hz):
    self.setfreq1(0, hz) # DDC
    self.setfreq1(1, hz) # display

    # a sleep seems to be needed for the case in which
    # a NetSDR is switching on the down-converter.
    if hz > 30000000 and (self.frequency == None or self.frequency < 30000000):
      time.sleep(0.5)

    self.frequency = hz

  def getfreq(self, chan):
      x = self.getitem(0x0020, x8(chan))
      hz = y40(x[1:6])
      return hz

  # set Receiver State to Run
  # only I/Q seems to work, not real.
  def setrun(self):
    self.running = True
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
    # self.print_setup()

  # stop receiver
  def stop(self):
    self.running = False
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
    data = ""
    data += x8(0) # ignored
    data += x32(rate)
    self.setitem(0x00B8, data)

  def getrate(self):
      x = self.getitem(0x00B8, x8(0))
      rate = y32(x[1:5])
      return rate

  # A/D Modes
  # set dither and A/D gain
  def setad(self):
    data = ""
    data += x8(0) # ignored
    # bit zero is dither, bit 1 is A/D gain 1.5
    #data += x8(0x3)
    data += x8(0x1)
    self.setitem(0x008A, data)

  # [ dither, A/D gain ]
  def getad(self, chan):
      x = self.getitem(0x008A, x8(0))
      dither = (ord(x[1]) & 1) != 0
      gain = (ord(x[1]) & 2) != 0
      return [ dither, gain ]

  # RF Filter Select
  # always sets automatic
  # 0=automatic
  # 11=bypass
  def setfilter(self):
    data = ""
    data += x8(0) # ignored
    data += x8(0) # automatic
    self.setitem(0x0044, data)

  def getfilter(self, chan):
      x = self.getitem(0x0044, x8(chan))
      return ord(x[1])

  # RF Gain
  # gain is 0, -10, -20 -30 dB
  def setgain(self, gain):
    data = ""
    data += x8(0) # channel 1
    data += x8(gain)
    self.setitem(0x0038, data)

  def getgain(self, chan):
    x = self.getitem(0x0038, x8(chan))
    return ord(x[1])

  # e.g. "NetSDR"
  def getname(self):
      x = self.getitem(0x0001)
      return x

  # e.g. "PS000553"
  def getserial(self):
      x = self.getitem(0x0002)
      return x

  # 123 means version 1.23
  # returns 10 for my NetSDR
  def getinterface(self):
      x = self.getitem(0x0003)
      return y16(x[0:2])
  
  # ID=0 boot code
  # ID=1 application firmware
  # ID=2 hardware
  # ID=3 FPGA configuration
  # XXX seems to cause protocol problems, NetSDR sends NAKs or something.
  def getversion(self, id):
      x = self.getitem(0x0004, x8(id))
      if x == None:
          # NAK
          return None
      if id == 3:
          return [ ord(x[1]), ord(x[2]) ] # ID, version
      else:
          return y16(x[1:3]) # version * 100

  # [ FPGA config number, FPGA config ID, FPGA revision, descr string ]
  # e.g. [1, 1, 7, 'Std FPGA Config \x00']
  def getfpga(self):
    x = self.getitem(0x000C)
    return [ ord(x[0]),
             ord(x[1]),
             ord(x[2]),
             x[3:] ]

  # Receiver A/D Amplitude Scale
  def getscale(self, chan):
    x = self.getitem(0x0023, x8(chan))
    return y16(x[1:3])

  # VHF/UHF Down Converter Gain
  # XXX seems to yield a NAK
  def getdowngain(self):
    x = self.getitem(0x003A)
    auto = ord(x[0])
    lna = ord(x[1])
    mixer = ord(x[2])
    ifout = ord(x[3])
    return [ auto, lna, mixer, ifout ]
                                            
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
  # returns a buffer with interleaved I and Q float64.
  # return an array of complex (real=I, imag=Q).
  def readiq(self):
    # read from the pipe; a 4-byte length, then the packet.
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
      # one or more packets were lost.
      # we'll fill the gap with zeros.
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
      pad = numpy.zeros(len(samples)*gap, dtype=numpy.float64),
      samples = numpy.append(pad, samples)

    ii1 = samples[0::2]
    qq1 = samples[1::2]
    cc1 = ii1 + 1j*qq1
    return cc1

  #
  # read from SDR-IP, demodulate as USB.
  #
  def readusb(self):
    iq = self.readiq()
    usb = weakutil.iq2usb(iq)
    return usb
