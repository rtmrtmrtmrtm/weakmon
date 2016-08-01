# weakmon
Monitor WSPR and JT65A.

These are command-line Python programs to monitor WSPR and JT65A. For
a few radios the software understands, it will switch bands
automatically. The software works on Macs, Linux, and FreeBSD.

I've borrowed code and ideas from Joe Taylor, Phil Karn, and others
identified in comments in the code.

The software depends on the following:
  python 2.7
  numpy
  scipy
  pyaudio

Here's how to install the dependencies on Ubuntu Linux:

If you have a Mac with macports:
  sudo port install python27
  sudo port install py27-numpy
  sudo port install py27-scipy
  sudo port install py27-pyaudio

Now compile Phil Karn's Reed-Solomon and convolutional decoders:

At this point you should be able to run wsprmon.py and jt65mon.py with
no arguments, and see lists of available sound cards and serial ports.

You'll need to ensure your computer's clock is correct, perhaps with ntp.

This code includes receive software for wwvb's new phase-shift
modulation, and for APRS.
