
#
# control an RFSpace SDR-IQ receiver.
#
# uses the kernel USB FTDI driver, which shows up as a serial device.
# may need SDR-IQ firmware >= 1.07.
#
# works on Linux and Macs.
#
# Example:
#   sdr = sdriq.open("/dev/cu.usbserial-142")
#   sdr.setrate(8138)
#   sdr.setgain(-10)
#   sdr.setifgain(12)
#   sdr.setrun(True)
#   while True:
#     buf = sdr.readiq()
#
# Robert Morris, AB1HL
#

import sys
import os
import thread
import serial
import time
import threading
import numpy
import select

# the SDR-IQ needs to be told how fast its sampling clock
# actually runs, so it can set its down-sampling frequency
# accurately. if frequencies seem a bit high, that means
# the SDR-IQ is sampling to slow, so ppm here should be set
# to a negative values (it's -23 for my SDR-IQ). if signals
# are appearing at too-low frequencies, the SDR-IQ is sampling
# too fast, and ppm should be set to a positive value.
# ppm is parts per million, so if a 10 mhz signal shows up
# at 10.000010, the SDR-IQ is sampling 1 PPM too slow
# and ppm should be set to -1.
ppm = -23

# correspondence between rate codes and I/Q sample rate
rates = [
    [ 0x00001FCA, 8138 ],
    [ 0x00003F94, 16276 ],
    [ 0x000093A1, 37793 ],
    [ 0x0000D904, 55556 ],
    [ 0x0001B207, 111111 ],
    [ 0x00026C0A, 158730 ],
    [ 0x0002FDEE, 196078 ],
]

def x16(x):
  return [ x & 0xff, (x >> 8) & 0xff ]

def x32(x):
  data = [ ]
  data += [ x % 256 ]
  x /= 256
  data += [ x % 256 ]
  x /= 256
  data += [ x % 256 ]
  x /= 256
  data += [ x % 256 ]
  x /= 256
  assert x == 0
  return data

def hx(s):
  buf = ""
  for i in range(0, len(s)):
    buf += "%02x " % (s[i])
  return buf

#
# if already connected, return existing SDRIQ,
# otherwise a new one.
#
sdriqs = { }
mu = thread.allocate_lock()
def open(dev):
    global sdriqs, mu
    mu.acquire()
    if not (dev in sdriqs):
        sdriqs[dev] = SDRIQ(dev)
    sdr = sdriqs[dev]
    mu.release()
    return sdr

class SDRIQ:
    def __init__(self, devname):
        # only one request/response at a time.
        self.w_mu = thread.allocate_lock()

        # protect self.ctl[].
        self.ctl_mu = thread.allocate_lock()

        # protect self.data[].
        self.data_mu = thread.allocate_lock()
        
        self.port = serial.Serial(devname,
                                  #timeout=2,
                                  baudrate=38400,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)

        # waiting type=4 data packets.
        # each entry is a 8192-entry array of ints.
        self.data = [ ]

        # waiting type=0 control replies.
        # each entry is [ mtype, mitem, data ].
        self.ctl = [ ]

        # used only by reader() thread.
        self.reader_buf = [ ]

        # fork() a sub-process to read and buffer USB input,
        # since the Python thread scheduler doesn't run us often enough if
        # WSPR is compute-bound in numpy for tens of seconds.
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            self.child(w)
            os._exit(0)
        else:
            self.pipe = r
            os.close(w)

        # only one thread reads the serial port, appending
        # arriving control and data messages to ctl[] and data[].
        self.th = threading.Thread(target=lambda : self.reader())
        self.th.daemon = True
        self.th.start()

        self.port.write("\x04\x20\x01\x00") # get target name, basically a no-op

        # tell the SDR-IQ what its actual sampling frequency is.
        # this does not change the sampling frequency.
        nclock = int(66666667 * (1.0 + ppm/1000000.0))
        self.setinputrate(nclock)
        sys.stderr.write("sdriq: DDC clock set to %d\n" % (self.getinputrate()))

    def child1(self):
        while True:
            # do our own read and buffer to avoid serial's internal
            # buffering and desire to return exactly the asked-for
            # number of bytes.
            timeout = 10
            select.select([self.port.fileno()], [], [], timeout)
            buf = os.read(self.port.fileno(), 8194)
            if len(buf) > 0:
                self.child_bufs_mu.acquire()
                self.child_bufs.append(buf)
                self.child_bufs_mu.release()

    # read USB data in a separate process, buffer it,
    # and send it on the pipe w.
    def child(self, w):
        ww = os.fdopen(w, 'wb')

        # spawn a thread that just keeps reading from USB
        # and appending buffers of bytes to child_bufs[].

        self.child_bufs = [ ]
        self.child_bufs_mu = thread.allocate_lock()

        th = threading.Thread(target=lambda : self.child1())
        th.daemon = True
        th.start()

        # copy buffers from child_bufs[] to the UNIX pipe.
        # the pipe write() calls may block, but it's OK because
        # the child1() thread keeps draining the UDP socket.
        while True:
            self.child_bufs_mu.acquire()
            bufs = self.child_bufs
            self.child_bufs = [ ]
            self.child_bufs_mu.release()

            if len(bufs) < 1:
                time.sleep(0.1)

            for buf in bufs:
                try:
                    ww.write(buf)
                except:
                    os._exit(1)
            ww.flush()

    def reader(self):
        while True:
            [ mtype, mitem, data ] = self.readmsg()
            if mtype == 4:
                # data -- I/Q samples
                if len(data) != 8192:
                    sys.stderr.write("sdriq: len(data) %d != 8192\n" % (len(data)))
                self.data_mu.acquire()
                self.data.append(data)
                self.data_mu.release()
            elif mtype > 4:
                sys.stderr.write("sdriq: unexpected type %d len=%d\n" % (mtype,
                                                                         len(data)))
            elif mtype == 0:
                # reply to a set/get control request.
                self.ctl_mu.acquire()
                self.ctl.append([ mtype, mitem, data ])
                self.ctl_mu.release()
            elif mtype == 1 and mitem == 5:
                # probably A/D overload.
                pass
            else:
                sys.stderr.write("sdriq: unexpected type=%d item=%d len=%d\n" % (mtype,
                                                                                 mitem,
                                                                                 len(data)))
                pass

    # read data from the pipe from the child process.
    # absorb it into self.reader_buf[].
    # only called by the reader thread.
    def rawread(self):
        buf = os.read(self.pipe, 10240)
        if len(buf) > 0:
            buf = [ ord(x) for x in buf ]
            self.reader_buf = self.reader_buf + buf

    # return exactly n bytes
    # only called by the reader thread.
    def readn(self, n):
        while len(self.reader_buf) < n:
            self.rawread()
        a = self.reader_buf[0:n]
        self.reader_buf = self.reader_buf[n:]
        return a

    # only called by the reader thread.
    def readmsg(self):
        x = self.readn(2)

        # 16-bit header
        mtype = (x[1] >> 5) & 7
        mlen = ((x[1] & 31) << 8) | x[0]

        if mtype >= 4:
            # data
            if mlen == 0:
                mlen = 8192 + 2
            data = self.readn(mlen - 2)
            return [ mtype, -1, data ]
        else:
            # response to control, or unsolicited control
            y = self.readn(2)
            mitem = y[0] | (y[1] << 8)
            data = self.readn(mlen - 4)
            return [ mtype, mitem, data ]

    def rawwrite(self, a):
        b = ""
        for aa in a:
            b = b + chr(aa)
        self.port.write(b)

    # look for the most recent reply to a control request
    # for item=mitem.
    def readreply(self, mitem):
        t0 = time.time()
        while True:
            if time.time() - t0 > 5:
                sys.stderr.write("sdriq: timed out waiting for reply to item=%x\n" % (mitem))
                sys.exit(1)
            got = None
            self.ctl_mu.acquire()
            for m in self.ctl:
                if m[0] == 0 and m[1] == mitem:
                    got = m
            self.ctl_mu.release()
            if got != None:
                return got[2]
            time.sleep(0.1)

    def clear_ctl(self):
        self.ctl_mu.acquire()
        self.ctl = [ ]
        self.ctl_mu.release()

    def getitem(self, mitem, param=None):
        buf = [ ]
        buf += [ 4 ] # overall length, lsb
        buf += [ 1 << 5 ] # mtype 1
        buf += x16(mitem)
        if param != None:
            buf += param
        # print "getitem writing %s" % (hx(buf))

        self.w_mu.acquire()
        self.clear_ctl()
        self.rawwrite(buf)
        ret = self.readreply(mitem)
        self.w_mu.release()

        return ret

    def setitem(self, mitem, data):
        lx = 4 + len(data)
        buf = [ ]
        buf += [ lx ] # overall length, lsb
        buf += [ 0 << 5 ] # mtype 0
        buf += x16(mitem)
        buf += data
        # print "setitem writing %s" % (hx(buf))

        self.w_mu.acquire()
        self.clear_ctl()
        self.rawwrite(buf)
        ret = self.readreply(mitem)
        self.w_mu.release()

        return ret

    def getfreq(self):
        x = self.getitem(0x0020, [0x00])
        x = x[1:]
        # four bytes of frequency in hz, LSB first
        hz = x[0]
        hz += x[1]*256
        hz += x[2]*256*256
        hz += x[3]*256*256*256
        return hz

    def setfreq(self, hz):
        data = [ ]
        data += [ 0 ]
        data += x32(hz)
        data += [ 0 ]
        self.setitem(0x0020, data)

    # id=0 --> PIC boot code version
    # id=1 --> PIC firmware version
    # returns version * 100
    # but does not work as documented on my SDR-IQ v1.05.
    def getversion(self, id):
        x = self.getitem(0x0004, [ id ])
        assert x[0] == id
        x = x[1:]
        return x[0] + 256*x[1]

    # RF gain, in dB.
    # this controls a combination of pre-amp and attenuator.
    def getgain(self):
        x = self.getitem(0x0038, [ 0 ] )
        db = x[1]
        if db == -10 & 0xff:
            return -10
        if db == -20 & 0xff:
            return -20
        if db == -30 & 0xff:
            return -30
        return db

    # 0, -10, -20, -30
    # -10 is probably good.
    def setgain(self, db):
        self.setitem(0x0038, [ 0, db & 0xff ])

    # the IF gain controls which of the 20 A/D bits are
    # returned in the 16-bit I/Q stream.
    # only 0, 6, 12, 18, and 24 dB.
    def getifgain(self):
        x = self.getitem(0x0040, [ 0 ] )
        return x[1]

    # 12 is probably good.
    def setifgain(self, db):
        self.setitem(0x0040, [ 0, db ])

    # I/Q data output sample rate
    def getrate(self):
        x = self.getitem(0x00B8, [ 0 ] )
        x = x[1:]
        z = x[0]
        z += x[1]*256
        z += x[2]*256*256
        z += x[3]*256*256*256
        for r in rates:
            if z == r[0]:
                return r[1]
        return 0

    def setrate(self, rate):
        code = None
        for r in rates:
            if rate == r[1]:
                code = r[0]
        if code == None:
            sys.stderr.write("sdriq: unknown I/Q rate %d\n" % (rate))
            sys.exit(1)
        data = [ 0 ]
        data += x32(code)
        self.setitem(0x00B8, data)

    # always more or less 66666667 Hz.
    def getinputrate(self):
        x = self.getitem(0x00B0, [ 0 ] )
        x = x[1:]
        z = x[0]
        z += x[1]*256
        z += x[2]*256*256
        z += x[3]*256*256*256
        return z

    # set to something very near 66666667 Hz to correct small
    # inaccuracies in the SDR-IQ's clock.
    def setinputrate(self, hz):
        data = [ 0 ]
        data += x32(hz)
        self.setitem(0x00B0, data)
    
    # ask the SDR-IQ to start generating samples.
    def setrun(self, run):
        data = [ ]
        data += [ 0x81 ]
        if run:
            data += [ 0x02 ] # 1=idle, 2=run
        else:
            data += [ 0x01 ] # 1=idle, 2=run
        data += [ 0 ] # 0=contiguous, 2=one shot, 3=trigger
        data += [ 1 ] # number of one-shot blocks
        self.setitem(0x0018, data)

    # return a numpy array of complex floats, real=I, imag=Q.
    def readiq(self):
        buf = None
        while buf == None:
            self.data_mu.acquire()
            if len(self.data) > 0:
                buf = self.data[0]
                self.data = self.data[1:]
            self.data_mu.release()
            if buf == None:
                time.sleep(0.1)

        # buf[] is 8192 long.
        # I1 lsb, I1 msb, Q1 lsb, Q1 msb, I2 lsb, I2 msb, &c.

        #print buf[1024:1024+10]

        buf = numpy.array(buf)

        ilsb = buf[0::4]
        imsb = buf[1::4]
        ii = numpy.add(ilsb, imsb * 256.0)

        qlsb = buf[2::4]
        qmsb = buf[3::4]
        qq = numpy.add(qlsb, qmsb * 256.0)

        ret = ii + 1j*qq

        return ret
