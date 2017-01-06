#!/usr/local/bin/python

#
# decode APRS packets.
#
# needs a sound card connected to a radio
# tuned to 144.390 in FM.
#

import na
import weakargs
import weakaudio
import weakcat
import sys

def cb(fate, msg, start, space_to_mark):
    # fate=0 -- unlikely to be correct.
    # fate=1 -- CRC failed but syntax look OK.
    # fate=2 -- CRC is correct.
    if fate >= 2:
        print "%s" % (msg)

def main():
    parser = weakargs.stdparse('Decode APRS.')
    args = weakargs.parse_args(parser)
    
    if args.cat != None:
        cat = weakcat.open(args.cat)
        cat.set_fm_data()
        cat.setf(0, 144390000)
        cat.sdr.setgain(0)

    if args.card == None:
        parser.error("aprsmon requires -card")

    ar = na.APRSRecv(11025)
    ar.callback = cb
    ar.opencard(args.card)
    ar.gocard()

    sys.exit(0)

main()
