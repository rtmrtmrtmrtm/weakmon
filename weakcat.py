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

def open(desc):
    [type, dev] = desc
    if type == "k3":
        return K3(dev)
    if type == "rx340":
        return RX340(dev)
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
    for ty in [ "k3", "rx340", "sdrip", "sdriq", "r75", "r8500", "ar5000", "eb200" ]:
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
        for mhz in [ 1.8, 3.5, 7.0, 10.1, 14.0, 18.068, 21.0, 24.890, 28.0 ]:
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
        pass

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
