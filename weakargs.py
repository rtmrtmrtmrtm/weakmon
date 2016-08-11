#
# standard argument parsing
#

import argparse
import sys
import weakaudio
import time

#
# set up for argument parsing, with standard arguments.
# caller can then optionally call parser.add_argument()
# for non-standard arguments, then weakargs.parse_args().
#
def stdparse(description):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("-card", nargs=2, metavar=('CARD', 'CHAN'))
  parser.add_argument("-cat", nargs=2, metavar=('TYPE', 'DEV'))
  parser.add_argument("-levels", action='store_true')
  parser.add_argument("-v", action='store_true')

  def myerror(message):
    parser.print_usage(sys.stderr)
    weakaudio.usage()
    parser.exit(2, ('%s: error: %s\n') % (parser.prog, message))

  parser.error = myerror

  return parser

#
# parse, and standard post-parsing actions.
#
def parse_args(parser):
    args = parser.parse_args()

    # don't require -cat if the "card" is really a controllable
    # radio itself.
    if args.card != None and args.card[0] in [ "sdrip", "sdriq", "eb200", "sdrplay" ] and args.cat == None:
        args.cat = args.card

    if args.levels == True:
        weakaudio.levels(args.card)

    return args
