#
# Control a Rohde & Schwarz EB200 receiver through its ethernet interface.
#
# Robert Morris, AB1HL
#

import socket
import sys
import thread
import threading
import time
import struct
import numpy
import weakutil

#
# if already connected, return existing EB200,
# otherwise a new one.
#
eb200s = { }
mu = thread.allocate_lock()
def open(dev):
    global eb200s, mu
    mu.acquire()
    if not (dev in eb200s):
        eb200s[dev] = EB200(dev)
    eb = eb200s[dev]
    mu.release()
    return eb

class EB200:

    def __init__(self, ipaddr, port=5555):
        self.ipaddr = ipaddr
        self.port = port

        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.connect((ipaddr, port))
        myhost = self.tcp.getsockname()[0]

        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
        self.udp.bind(('', 0)) # ask kernel for a port
        myport = self.udp.getsockname()[1]

        # set audio format to 8000/sec, 16 bits/sample, one channel.
        self.rate = 8000
        self.tcp.send("SYSTem:AUDio:REMote:MODe 10\n")

        # set up a UdpPath pointing at our UDP port.
        # use TRACE:UDP? to read this back.
        cmd = "TRACE:UDP:DEFAULT:TAG:ON \"%s\", %d, AUDIO" % (myhost,
                                                              myport)
        self.tcp.send("%s\n" % (cmd))

        # input buffer of buffers filled by reader() thread.
        self.bufs = [ ]
        self.bufs_mu = thread.allocate_lock()

        # last sequence number seen from EB200 in a UDP packet.
        self.seq = None

        # separate reader thread to keep reading the UDP
        # socket so it doesn't overflow.
        self.th = threading.Thread(target=lambda : self.reader())
        self.th.daemon = True
        self.th.start()

    def parse(self, buf):
        if len(buf) < 16+4+8:
            sys.stderr.write("eb200: too short\n")
            return

        # EB200 header.
        h = struct.unpack(">IHHH", buf[0:10])
        if h[0] != 0x000EB200:
            sys.stderr.write("eb200: bad magic\n")
            return
        seq = h[3]

        if self.seq != None:
            if seq != self.seq + 1 and seq > self.seq:
                sys.stderr.write("eb200: missed %d packets (%d %d)\n" % (seq - self.seq,
                     self.seq, seq))
                # XXX should fake the right number of samples.
        self.seq = seq

        # GenericAttribute
        ga = struct.unpack(">HH", buf[16:20])
        if ga[0] != 401:
            sys.stderr.write("eb200: not an AUDIO tag\n")
            return
        galen = ga[1] # the whole rest of the packet

        # TraceAttribute == AudioAttribute
        ta = struct.unpack(">HBBI", buf[20:28])
        
        # ta[0] is the number of 16-bit samples.
        # ta[3] is selectorFlags, always 0x40000.
        if ta[2] != 0:
            sys.stderr.write("eb200: unexpected optional header\n")
            return
        if ta[3] != 0x40000:
            sys.stderr.write("eb200: unexpected flag %x\n" % (ta[3]))
            return

        samples = buf[28:]

        # samples is now a string, containing big-endian shorts.
        # convert to a numpy array.
        # this code assumes the EB200 sends signed samples.
        # samples = numpy.fromstring(samples, dtype=numpy.int16)
        samples = numpy.fromstring(samples, dtype='>i2')

        self.bufs_mu.acquire()
        self.bufs.append(samples)
        self.bufs_mu.release()
        

    def reader(self):
        while True:
            buf = self.udp.recv(8192)
            self.parse(buf)

    # blocks until some samples are available.
    def readaudio(self):
        while True:
            self.bufs_mu.acquire()
            bufs = self.bufs
            self.bufs = [ ]
            self.bufs_mu.release()

            if len(bufs) > 0:
                buf = numpy.concatenate(bufs)
                return buf

            time.sleep(0.2)

    def getrate(self):
        return self.rate

    def setfreq(self, hz):
        self.tcp.send("freq %d\n" % (hz))

    def set_usb_data(self):
        self.tcp.send(":freq:afc 0\n")
        self.tcp.send(":output:squelch 0\n")
        self.tcp.send(":input:att:auto 1\n")
        # set BW first, since otherwise can't change to USB.
        self.tcp.send("band 2400\n")
        self.tcp.send("demodulation usb\n")

    def set_fm_data(self):
        self.tcp.send(":freq:afc 0\n")
        self.tcp.send(":output:squelch 0\n")
        self.tcp.send(":input:att:auto 1\n")
        self.tcp.send("band 15000\n")
        self.tcp.send("demodulation fm\n")
        self.tcp.send("band 15000\n")
