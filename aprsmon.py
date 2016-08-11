#!/usr/local/bin/python

#
# decode APRS packets.
#
# needs a sound card connected to a radio
# tuned to 144.390 in FM.
#

import aprsrecv
import weakargs
import weakaudio
import weakcat
import sys

def cb(start, n, fate, msg):
    # fate=0 -- unlikely to be correct.
    # fate=1 -- CRC failed but syntax look OK.
    # fate=2 -- CRC is correct.
    if fate >= 1:
        print msg

def main():
    parser = weakargs.stdparse('Decode APRS.')
    args = weakargs.parse_args(parser)
    
    if args.cat != None:
        cat = weakcat.open(args.cat)
        # really should be a way to set FM.
        cat.setf(0, 144390000)

    if args.card == None:
        parser.error("aprsmon requires -card")

    ar = aprsrecv.APRSRecv(44100)
    ar.callback = cb
    ar.opencard(args.card)
    ar.gocard()

    sys.exit(0)

main()
