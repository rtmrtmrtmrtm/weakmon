#!/usr/local/bin/python2.7

#
# decode APRS packets.
#
# needs a sound card connected to a radio
# tuned to 144.390 in FM.
#

import aprsrecv
import sys

def cb(start, n, fate, msg):
    # fate=0 -- unlikely to be correct.
    # fate=1 -- CRC failed but syntax look OK.
    # fate=2 -- CRC is correct.
    if fate >= 1:
        print msg

def main():
    ar = aprsrecv.APRSRecv(44100)
    ar.callback = cb
    ar.opencard("2:0")
    ar.gocard()

main()
