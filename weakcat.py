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
    if type == "k3":
        return K3(dev)
    if type == "rx340":
        return RX340(dev)
    if type == "8711":
        return WJ8711(dev)
    if type == "r75":
        return R75(dev, 0x5A)
    if type == "r8500":
        return R75(dev, 0x4A)
    if type == "ar5000":
        return AR5000(dev)
    if type == "sdrip":
        return SDRIP(dev)
    if type == "sdriq":
        return SDRIQ(dev)
    if type == "eb200":
        return EB200(dev)
    if type == "sdrplay":
        return SDRplay(dev)
    if type == "prc138":
        return PRC138(dev)

    sys.stderr.write("weakcat: unknown radio type %s\n" % (type))
    sys.exit(1)

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
    sys.stderr.write("serial devices:\n")
    coms = comports()
    for com in coms:
        sys.stderr.write("  %s\n" % (com))
    sys.stderr.write("radio types: ")
    for ty in [ "k3", "rx340", "8711", "sdrip", "sdriq", "r75", "r8500", "ar5000", "eb200", "sdrplay", "prc138" ]:
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
              print "k3 read timeout"
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
        self.cmd("K22")
        self.cmd("K2")
        r = self.readrsp("K2")
        if r != "K22":
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
class R75(object):
    def __init__(self, devname, civ):
        # ic-r75 CI-V address defaults to 0x5A
        # ic-r8500 CI-V address defaults to 0x4A

        self.civ = civ

        self.port = serial.Serial(devname,
                                  timeout=2,
                                  baudrate=9600,
                                  parity=serial.PARITY_NONE,
                                  bytesize=serial.EIGHTBITS)
        
    def cmd(self, cmd, subcmd, data):
        self.port.write("\xfe\xfe%c\xe0" % (self.civ))
        self.port.write(chr(cmd))
        if subcmd != None:
            self.port.write(chr(subcmd))
        self.port.write(data)
        self.port.write("\xfd")
        time.sleep(0.01)
  
    # send a no-op command and wait for the response.
    def sync(self):
        pass

    # encode a frequency in hz in BCD.
    def bcd(self, hz):
        # 10 hz first -- no single hz.
        hz = int(hz)
        s = ""
        for i in range(0, 5):
            d0 = hz % 10
            hz /= 10
            d1 = hz % 10
            hz /= 10
            s += chr(d1*16 + d0)
        return s

    # set the frequeny in Hz for vfo=0 (A) or vfo=1 (B / sub-receiver).
    # does not wait.
    def setf(self, vfo, fr):
        self.cmd(0x05, None, self.bcd(fr))

    def set_usb_data(self):
        self.cmd(0x06, 0x01, "") # USB

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
        print "prc138 sync %d %d %s %.1f" % (rx, tx, mode, bw)

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
        self.set_pow("HIGH")
        self.set_bw(2.7)

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
