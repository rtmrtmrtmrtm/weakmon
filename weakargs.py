#
# standard argument parsing
#

import argparse
import sys
import weakaudio

# set up for argument parsing, with standard arguments.
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
