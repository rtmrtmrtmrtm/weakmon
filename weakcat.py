#!/usr/local/bin/python

#
# control various radios via rs232.
#

# pySerial
# https://pyserial.readthedocs.org/en/latest/
# port install py27-serial
# python -m serial.tools.list_ports
# python -m serial.tools.miniterm /dev/XXX

import serial
import serial.tools
import serial.tools.list_ports

import sys
import re
import time

import sdrip
import sdriq
import eb200
import sdrplay

def open(desc):
    [type, dev] = desc
    ret = None
    if type == "k3":
        ret = K3(dev)
    if type == "rx340":
        ret = RX340(dev)
    if type == "8711":
        ret = WJ8711(dev)
    if type == "r75":
        ret = R75(dev, 0x5A)
    if type == "r8500":
        ret = R75(dev, 0x4A)
    if type == "r8600":
        ret = R75(dev, 0x96)
    if type == "f8101":
        ret = F8101(dev, 0x8A)
    if type == "ar5000":
        ret = AR5000(dev)
    if type == "sdrip":
        ret = SDRIP(dev)
    if type == "sdriq":
        ret = SDRIQ(dev)
    if type == "eb200":
        ret = EB200(dev)
    if type == "sdrplay":
        ret = SDRplay(dev)
    if type == "prc138":
        ret = PRC138(dev)

    if ret == None:
        sys.stderr.write("weakcat: unknown radio type %s\n" % (type))
        sys.exit(1)

    ret.type = type
    ret.dev = dev
    return ret

# return a list of serial port device names.
def comports():
    a = serial.tools.list_ports.comports()
    if len(a) == 0:
        return [ ]
    if type(a[0]) == tuple:
        b = [ ]
        for aa in a:
            b.append(aa[0])
        return b
    else:
        b = [ ]
        for aa in a:
            b.append(aa.device)
        return b

# print serial ports and radio types.
def usage():
    sys.stderr.write("serial devices for -cat:\n")
    coms = comports()
    for com in coms:
        sys.stderr.write("  %s\n" % (com))
    sys.stderr.write("radio types for -cat: ")
    for ty in [ "k3", "rx340", "8711", "sdrip", "sdriq", "r75", "r8500", "r8600", "f8101", "ar5000", "eb200", "sdrplay", "prc138" ]:
        sys.stderr.write("%s " % (ty))
    sys.stderr.write("\n")

class K3(object):
    def __init__(self, devname):
        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=38400,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)
        
        # input buffer (from radio).
        self.buf = ""
  
    # read input from the radio through the next semi-colon.
    # don't return the semicolon.
    def readsemi(self):
        while True:
          i = self.buf.find(';')
          if i != -1:
              r = self.buf[0:i]
              self.buf = self.buf[i+1:]
              return r
          z = self.port.read()
          if z == "":
              print("k3 read timeout")
          self.buf += z
  
    # we're expecting some information from the radio in
    # response to a command, starting with the word prefix; get it
    # and return the response.
    def readrsp(self, prefix):
        while True:
            x = self.readsemi()
            if x[0:len(prefix)] == prefix.upper():
                return x

    def cmd(self, cmd):
        time.sleep(0.01) # KX2 needs this
        self.port.write(cmd + ";")
  
    # send a no-op command and wait for the response.
    def sync(self):
        self.cmd("AN")
        r = self.readrsp("AN")
        if r[0:2] != "AN":
            sys.stderr.write("k3.sync: got weird %s\n" % (r))

    # get the frequeny in Hz from vfo=0 (A) or vfo=1 (B / sub-receiver).
    def getf(self, vfo):
        if vfo == 0:
            cmd = "fa"
        else:
            cmd = "fb"
        self.cmd(cmd)
        r = self.readrsp(cmd)
        assert r[0:2] == cmd.upper()
        return int(r[2:])

    # put the radio into a mode and bandwidth suitable for data.
    # DATA. DATA A. USB. 2.8 khz.
    # need to do it for each ham band.
    def set_usb_data(self):
        for mhz in [ 1.838, 3.576, 5.357, 7.076, 10.138,
                     14.076, 18.102, 21.076, 24.917, 28.076 ]:
            self.setf(0, int(mhz * 1000000))
            self.cmd("MD6") # DATA
            self.cmd("DT0") # DATA A
            self.cmd("BW0280") # filter bandwidth in 10-hz units

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        if vfo == 0:
            cmd = "fa"
        else:
            cmd = "fb"
        cmd += "%011d" % (int(fr))
        self.cmd(cmd)

    def setpower(self, watts):
        self.cmd("PC%03d" % watts)

    def tx(self):
        pass

    def rx(self):
        pass

# Ten-Tec RX-340
class RX340(object):
    def __init__(self, devname):
        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=19200,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)
        
    def cmd(self, s):
        self.port.write("\r$0%s\r" % (s))
        time.sleep(0.05)
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.cmd("F%.6f" % (fr / 1000000.0))

    def set_usb_data(self):
        self.cmd("D7") # USB
        self.cmd("I2.0") # b/w 2 khz
        self.cmd("K1") # preamp off, att off
        self.cmd("M1") # fast AGC

# Watkins Johnson WJ-8711, HF-1000, etc.
class WJ8711(object):
    def __init__(self, devname):
        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=9600,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)
        
    def cmd(self, s):
        self.port.write("\r%s\r" % (s))
        time.sleep(0.05)
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.cmd("FRQ %.6f" % (fr / 1000000.0))

    def set_usb_data(self):
        self.cmd("DET 4") # USB
        self.cmd("BWC 2800") # b/w
        self.cmd("AGC 2") # fast AGC

class SDRIP(object):
    def __init__(self, devname):
        self.sdr = sdrip.open(devname) # devname is the IP address
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.sdr.setfreq(fr)
        if fr < 8000000:
            self.sdr.setgain(-10)
        else:
            self.sdr.setgain(0)

    def set_usb_data(self):
        self.sdr.set_mode("usb")

    def set_fm_data(self):
        self.sdr.set_mode("fm")

class SDRIQ(object):
    def __init__(self, devname):
        self.sdr = sdriq.open(devname) # devname is /dev/SERIALPORT
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.sdr.setfreq(fr)

    def set_usb_data(self):
        pass

# Icom IC-R75
# Icom IC-R8500
# Icom IC-R8600
class R75(object):
    def __init__(self, devname, civ):
        # ic-r75 CI-V address defaults to 0x5A
        # ic-r8500 CI-V address defaults to 0x4A
        # ic-r8600 CI-V address defaults to 0x96

        self.civ = civ

        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=9600,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)
        
    def cmd(self, cmd, subcmd, data):
        # python3's serial module wants bytes, not str.
        s = b""
        s += b"\xfe\xfe%c\xe0" % (self.civ)
        s += bytearray([cmd]) # chr(cmd)
        if subcmd != None:
            s += bytearray([subcmd]) # chr(subcmd)
        s += data
        s += b"\xfd"
        self.port.write(s)
        time.sleep(0.01)
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # encode a frequency in hz in BCD.
    def bcd(self, hz):
        # 10 hz first -- no single hz.
        hz = int(hz)
        s = b""
        for i in range(0, 5):
            d0 = hz % 10
            hz //= 10
            d1 = hz % 10
            hz //= 10
            s += bytearray([d1*16+d0]) # chr(d1*16 + d0)
        return s

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.cmd(0x05, None, self.bcd(fr))

    def set_usb_data(self):
        self.cmd(0x06, 0x01, "") # USB

    def set_fm_data(self):
        self.cmd(0x06, 0x05, "") # FM

# Icom F8101
# CI-V commands are not compatible with R75.
# on Linux, /dev/ttyUSB1
class F8101(object):
    def __init__(self, devname, civ):
        # ic-f8101 CI-V address defaults to 0x8A
        self.civ = civ

        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=38400,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)

        self.rx()

        self.printinfo()

    # fetch various parameters from the radio and print them.
    # for many of them you have to be in MGR mode.
    def printinfo(self):
        a = self.cmd([ 0x03 ], b"")
        hz = self.parse_freq(a)
        print("freq %d" % (hz))

        a = self.cmd([ 0x1A, 0x34 ], b"")
        mode = a[1] + 256*a[0]
        print("mode %04x" % (mode)) 

        a = self.cmd([ 0x1A, 0x37 ], b"")
        x = a[1] + 256*a[0]
        print("TX status %04x" % (x)) 

        a = self.cmd([ 0x1C, 0x00 ], b"")
        print("ptt output %02x" % (a[0])) # 00 RX, 01 PTT TX

        if False:
            a = self.cmd([ 0x1C, 0x00 ], b"\x01") # transmit!
            time.sleep(3)
            a = self.cmd([ 0x1C, 0x00 ], b"\x00") # receive

        a = self.cmd([ 0x1C, 0x01 ], b"")
        print("tuner %02x" % (a[0]))  # 00 on, 01 through

        cc = [
            [ "preamp", 0x03, 0x05 ],
            [ "tuner", 0x03, 0x14 ],
            [ "ptt tune", 0x03, 0x015 ],
            [ "power", 0x03, 0x07 ],
            [ "NB", 0x03, 0x01 ],
            [ "agc", 0x03, 0x06 ],
            [ "fan", 0x03, 0x08 ],
            [ "usb enabled", 0x08, 0x00 ],
            [ "usb filter", 0x08, 0x02 ],
            [ "usb source", 0x08, 0x03 ],
            [ "home", 0x19, 0x04 ],
        ]

        for c in cc:
            a = self.cmd([ 0x1A, 0x05, c[1], c[2] ], b"")
            if len(a) == 2:
                x = a[1] + 256*a[0]
                print("%s %04x" % (c[0], x))
            else:
                print("%s ???" % (c[0]))

        #self.cmd([ 0x07, 0x00 ], b"") # Set VFO A -- works
        #self.cmd([ 0x03 ], b"") # read operating freq -- works
        #self.cmd([ 0x1A, 0x35 ], b"\x80\x67\x45\x23") # set freq -- works

    # given a bytearray from self.cmd() of the
    # 03 of 1A 35 command, return Hz.
    def parse_freq(self, a):
        hz = 0
        hz += 1 * ((a[0] >> 0) & 0xf)
        hz += 10 * ((a[0] >> 4) & 0xf)
        hz += 100 * ((a[1] >> 0) & 0xf)
        hz += 1000 * ((a[1] >> 4) & 0xf)
        hz += 10000 * ((a[2] >> 0) & 0xf)
        hz += 100000 * ((a[2] >> 4) & 0xf)
        hz += 1000000 * ((a[3] >> 0) & 0xf)
        hz += 10000000 * ((a[3] >> 4) & 0xf)
        hz += 100000000 * ((a[4] >> 0) & 0xf)
        hz += 1000000000 * ((a[4] >> 4) & 0xf)
        return hz

    # drain and discard any leftover input.
    def drain(self):
        n = self.port.in_waiting
        if n != 0:
            print("draining %d" % (n))
        self.port.read(size=n)
        
    # send a command, wait for the reply,
    # return any replied data as a bytearray,
    # or True for a simple OK reply, or False for
    # an NG error.
    # cmd is an array of integers, e.g. [ cmd, subcmd, category, item ].
    def cmd(self, cmd, data):
        self.drain()

        # python3's serial module wants bytes, not str.
        s = b""
        s += b"\xfe\xfe%c\xe0" % (self.civ)
        s += bytearray(cmd)
        if data != None:
            s += data
        s += b"\xfd"
        self.port.write(s)
        time.sleep(0.01)

        # wait for the reply
        # OK: fe fe e0 8a fb fd
        # error: fe fe e0 8a fa fd
        # reply to 0x03: fe fe e0 8a 03 80 67 45 23 00 fd

        buf = b""
        while True:
            x = self.port.read(size=1)
            if len(x) == 0:
                # timeout
                sys.stderr.write("F8101 reply read timeout\n")
                return False
            buf += x
            fefe = buf.rfind(b"\xfe\xfe")
            if fefe != -1 and buf.endswith(b"\xfd"):
                # avoid python2/python3 ord() difficulty
                a = bytearray(buf[fefe:])
                if a[2] == 0xe0 and a[3] == self.civ:
                    if False:
                        for x in a:
                            sys.stderr.write("%02x " % (x))
                        sys.stderr.write("\n")
                    # from radio to us
                    if len(a) == 6 and a[4] == 0xFB:
                        return True
                    if len(a) == 6 and a[4] == 0xFA:
                        sys.stderr.write("F8101 NG %s %s\n" % (cmd, data))
                        return False
                    if a[4:4+len(cmd)] == bytearray(cmd):
                        i0 = 4 + len(cmd)
                        return a[i0:len(a)-1]
  
    # send a no-op command and wait for the response.
    # no need for this since all commands are synchronous.
    def sync(self):
        pass

    def tx(self):
        # seems to both switch audio input to USB sound card,
        # and to activate VOX, so PTT isn't needed.
        self.cmd([ 0x1A, 0x37 ], b"\x00\x02") # TX by ACC PTT

        # simulate PTT
        # not needed when PTT ACC set.
        #self.cmd([ 0x1C, 0x00 ], b"\x01")

        if False:
            a = self.cmd([ 0x1A, 0x37 ], b"")
            x = a[1] + 256*a[0]
            print("TX status %04x" % (x)) 


    def rx(self):
        # release PTT -- return to receive mode
        self.cmd([ 0x1C, 0x00 ], b"\x00")
        self.cmd([ 0x1A, 0x37 ], b"\x00\x00") # TX by PTT

    # encode a frequency in hz in BCD.
    def bcd(self, hz):
        hz = int(hz)
        s = b""
        for i in range(0, 5):
            d0 = hz % 10
            hz //= 10
            d1 = hz % 10
            hz //= 10
            s += bytearray([d1*16+d0]) # chr(d1*16 + d0)
        return s

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        data = self.bcd(fr)
        self.cmd( [ 0x1A, 0x35 ], data )

    def set_usb_data(self):
        xxx

# AOR AR-5000
class AR5000(object):
    def __init__(self, devname):
        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=9600,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)

        # BW1 -- 3 khz
        # AU0 -- auto mode off
        # MD3 -- USB
        # RF0014076000 -- set frequency in Hz
        # AC0 -- fast AGC
        
    def cmd(self, cmd):
        self.port.write(cmd + "\r")
        time.sleep(0.1)
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.cmd("RF%010d" % (int(fr)))

    def set_usb_data(self):
        self.cmd("MD3") # USB
        self.cmd("BW1") # 3 khz
        self.cmd("AC0") # fast AGC

#
# The PRC-138 knob must be set to RMT.
#
class PRC138(object):
    def __init__(self, devname):
        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=9600,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)

        # input buffer
        self.buf = ""

        self.port.write("\r")
        self.prompt()
        self.port.write("PORT_R EC OF\r") # don't echo commands
        self.prompt()
    
    def readline(self, retprompt=False):
        while True:
            if retprompt and "SSB>" in self.buf:
                self.buf = ""
                return "SSB>"
            i = self.buf.find('\r')
            if i == -1:
                i = self.buf.find('\n')
            if i != -1:
                r = self.buf[0:i]
                self.buf = self.buf[i+1:]
                return r
            z = self.port.read()
            self.buf += z

    # wait for a prompt
    def prompt(self):
        while True:
            x = self.readline(True)
            if x == "SSB>":
                return True

    # we're expecting some information from the radio in
    # response to a command, starting with the word prefix; get it
    # and return the info.
    def scan(self, prefix):
        while True:
            x = self.readline()
            while len(x) > 0 and (x[-1] == '\r' or x[-1] == '\n'):
                x = x[0:-1]
            #print "readline: %s" % (x)
            a = x.split(" ")
            if len(a) >= 2:
                if a[0] == prefix:
                    return a[1]
        
    def cmd(self, cmd):
        self.port.write(cmd + "\r")
        time.sleep(0.1)

    # send a no-op command wait for SSB> to make sure
    # the radio is really there and that we eat all
    # pending input.
    def sync(self):
        self.cmd("TIME")
        self.scan("DATE")
        self.prompt()
        [ rx, tx ] = self.get_fr()
        mode = self.get_mode()
        bw = self.get_bw()
        print("prc138 sync %d %d %s %.1f" % (rx, tx, mode, bw))

    def get_fr(self):
        self.port.write("FR\r")
        rx = self.scan("RxFr")
        tx = self.scan("TxFr")
        self.prompt()
        return [ int(rx), int(tx) ]

    def get_mode(self):
        self.port.write("MODE\r")
        x = self.scan("MODE")
        self.prompt()
        return x

    def get_bw(self):
        self.port.write("BAND\r")
        x = self.scan("BAND")
        self.prompt()
        return float(x)

    # USB, LSB, AME, CW, FM
    def set_mode(self, mode):
        self.port.write("MODE %s\r" % (mode))
        self.prompt()

    # OFF, SLOW, MED, FAST, DATA
    def set_agc(self, agc):
        self.port.write("AGC %s\r" % (agc))
        self.prompt()

    # LOW, MED, HIGH
    def set_pow(self, pow):
        self.port.write("POW %s\r" % (pow))
        self.prompt()

    # kHz (2.4, 2.7, 3.0 for ssb)
    def set_bw(self, bw):
        self.port.write("BA %.3f\r" % (bw))
        self.prompt()

    # set both rx and tx, in hz.
    # prc-138 demands exactly 8 digits,
    # with leading zeroes if needed.
    def setf(self, vfo, fr):
        assert vfo == 0
        cmd = "FR %08d" % (int(fr))
        self.port.write(cmd + "\r")
        self.prompt()

    # ask coupler to re-tune.
    def retune(self):
        self.port.write("RETUNE\r")
        self.prompt()

    def set_usb_data(self):
        self.set_mode("USB")
        self.set_agc("DATA")
        # self.set_pow("HIGH")
        self.set_bw(2.7)

    def tx(self):
        pass

    def rx(self):
        pass

class EB200(object):
    def __init__(self, devname):
        self.eb200 = eb200.open(devname) # devname is the IP address
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.eb200.setfreq(fr)

    def set_usb_data(self):
        self.eb200.set_usb_data()

    def set_fm_data(self):
        self.eb200.set_fm_data()

class SDRplay(object):
    def __init__(self, devname):
        self.sdr = sdrplay.open(devname)
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    def setf(self, vfo, fr):
        self.sdr.setfreq(fr)

    def set_usb_data(self):
        pass
