#
# send a report to pskreporter.info.
#
# https://pskreporter.info/pskdev.html
#

import struct
import time
import socket
import sys

# turn an array of 8-bit numbers into a string.
def hx(a):
    s = ""
    for x in a:
        s = s + chr(x)
    return s

# pack a string, preceded by length.
def pstr(s):
    return chr(len(s)) + s

# pack a 32-bit int.
def p32(i):
    z = struct.pack(">I", i)
    assert len(z) == 4
    return z

# pack a 16-bit int.
def p16(i):
    z = struct.pack(">H", i)
    assert len(z) == 2
    return z

# pad to a multiple of four byte.
def pad(s):
    while (len(s) % 4) != 0:
        s += chr(0)
    return s

#
# format and send reports.
#
class T:

    def __init__(self, mycall, mygrid, mysw, testing=False):
        self.testing = testing
        self.seq = 1
        self.sessionId = int(time.time())
        self.mycall = mycall
        self.mygrid = mygrid
        self.mysw = mysw

        host = "pskreporter.info"
        if self.testing:
            port = 14739 # test server
            # test view: https://pskreporter.info/cgi-bin/psk-analysis.pl
        else:
            port = 4739 # production server

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.connect((host, port))

        # accumulate a list, send only every 5 minutes.
        self.pending = [ ]
        self.last_send = 0

    # seq should increment once per packet.
    # sessionId should stay the same.
    # mysw is the name of the software.
    # each senders element is [ call, freq, snr, grid, time ]
    # e.g. [ "KB1MBX", 14070987, "PSK31", "FN42", 1200960104 ]
    # modes: JT65, PSK31
    def fmt(self, senders):
    
        # receiver record format descriptor.
        # callsign, locator, s/w.
        rrf = hx([0x00, 0x03, 0x00, 0x24, 0x99, 0x92, 0x00, 0x03, 0x00, 0x00,
                  0x80, 0x02, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                  0x80, 0x04, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                  0x80, 0x08, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                  0x00, 0x00,])
    
        # sender record format descriptor.
        if False:
            # senderCallsign, frequency, sNR (1 byte), iMD (1 byte), mode (1 byte), informationSource, flowStartSeconds.
            srf = hx([ 0x00, 0x02, 0x00, 0x3C, 0x99, 0x93, 0x00, 0x07,
                       0x80, 0x01, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x05, 0x00, 0x04, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x07, 0x00, 0x01, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x0A, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x0B, 0x00, 0x01, 0x00, 0x00, 0x76, 0x8F,
                       0x00, 0x96, 0x00, 0x04,])
    
        if True:
            # senderCallsign, frequency, mode, informationSource=1, senderLocator, flowStartSeconds
            srf = hx([ 0x00, 0x02, 0x00, 0x34, 0x99, 0x93, 0x00, 0x06,
                       0x80, 0x01, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x05, 0x00, 0x04, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x0A, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x0B, 0x00, 0x01, 0x00, 0x00, 0x76, 0x8F,
                       0x80, 0x03, 0xFF, 0xFF, 0x00, 0x00, 0x76, 0x8F,
                       0x00, 0x96, 0x00, 0x04,])
    
        # receiver record.
        # first cook up the data part of the record, since length comes first.
        rr = ""
        rr += pstr(self.mycall)
        rr += pstr(self.mygrid)
        rr += pstr(self.mysw)
        rr = pad(rr)
        # prepend rr's header.
        rr = hx([0x99, 0x92]) + p16(len(rr) + 4) + rr
    
        # sender records.
        # first the array of per-sender records, so we can find the length.
        sr = ""
        for snd in senders:
            # snd = [ "KB1MBX", 14070987, "PSK", "FN42", 1200960104 ]
            sr += pstr(snd[0]) # call sign
            sr += p32(snd[1])  # frequency
            sr += pstr(snd[2]) # "JT65"
            sr += chr(1) # informationSource
            sr += pstr(snd[3]) # grid
            sr += p32(int(snd[4]))
        sr = pad(sr)
        # prepend the sender records header, with length.
        sr = hx([0x99, 0x93]) + p16(len(sr) + 4) + sr
    
        # now the overall header (16 bytes long).
        h = ""
        h += hx([ 0x00, 0x0a ])
        h += p16(len(rrf) + len(srf) + len(rr) + len(sr) + 16)
        h += p32(int(time.time()))
        h += p32(self.seq)
        self.seq += 1
        h += p32(self.sessionId)
    
        pkt = h + rrf + srf + rr + sr
    
        return pkt

    def dump(self, pkt):
        for i in range(0, 20):
            sys.stdout.write("%02x " % ord(pkt[i]))
        sys.stdout.write("\n")

    def send(self, pkt):
        self.s.send(pkt)

    # caller received something. buffer it until 5 minutes
    # since last send.
    # XXX what if packet would be > MTU but not yet 5 minutes?
    def got(self, call, hz, mode, grid, tm):
        info = [ call, int(hz), mode, grid, int(tm) ]
        self.pending.append(info)
        if time.time() - self.last_send >= 5*60:
            pkt = self.fmt(self.pending)
            self.send(pkt)
            self.pending = [ ]
            self.last_send = time.time()
    
if __name__ == '__main__':
    pskr = T("AB1HL", "FN42", "weakmon 0.1", True)

    #pkt = pskr.fmt([ [ "W1WW", 10138000, "CW", "EM11", time.time() ],
    #                 [ "K1D", 14076000, "JT65", "DD14", time.time() ] ])
    #pskr.dump(pkt)
    #pskr.send(pkt)

    pskr.got("W1WW", 10138000, "CW", "EM11", time.time())
    pskr.got("K1D", 14076000, "JT65", "DD14", time.time())
