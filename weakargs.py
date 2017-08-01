#
# standard argument parsing
#

import argparse
import sys
import weakaudio
import weakcat
import time

class Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        # print '{n} {v} {o}'.format(n = namespace, v = values, o = option_string)
        if getattr(namespace, self.dest) is not None:
            msg = '{o} can only be specified once'.format(o = option_string)
            raise argparse.ArgumentError(None, msg)
        setattr(namespace, self.dest, values)
#
# set up for argument parsing, with standard arguments.
# caller can then optionally call parser.add_argument()
# for non-standard arguments, then weakargs.parse_args().
#
def stdparse(description):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("-card", nargs=2, metavar=('CARD', 'CHAN'), action=Once)
  parser.add_argument("-cat", nargs=2, metavar=('TYPE', 'DEV'), action=Once)
  parser.add_argument("-card2", nargs=2, metavar=('CARD', 'CHAN'))
  parser.add_argument("-card3", nargs=2, metavar=('CARD', 'CHAN'))
  parser.add_argument("-card4", nargs=2, metavar=('CARD', 'CHAN'))
  parser.add_argument("-cat2", nargs=2, metavar=('TYPE', 'DEV'))
  parser.add_argument("-cat3", nargs=2, metavar=('TYPE', 'DEV'))
  parser.add_argument("-cat4", nargs=2, metavar=('TYPE', 'DEV'))
  parser.add_argument("-levels", action='store_true')
  parser.add_argument("-v", action='store_true')

  def myerror(message):
    parser.print_usage(sys.stderr)
    weakaudio.usage()
    weakcat.usage()
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
    if args.card2 != None and args.card2[0] in [ "sdrip", "sdriq", "eb200", "sdrplay" ] and args.cat2 == None:
        args.cat2 = args.card2
    if args.card3 != None and args.card3[0] in [ "sdrip", "sdriq", "eb200", "sdrplay" ] and args.cat3 == None:
        args.cat3 = args.card3
    if args.card4 != None and args.card4[0] in [ "sdrip", "sdriq", "eb200", "sdrplay" ] and args.cat4 == None:
        args.cat4 = args.card4

    if args.levels == True:
        weakaudio.levels(args.card)

    return args
