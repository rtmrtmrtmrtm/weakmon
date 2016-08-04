# weakmon

These are command-line Python programs to monitor WSPR and JT65A,
printing receptions and reporting them to wsprnet and pskreporter. For
the radios the software understands, it will switch bands
automatically. The software works on Macs, Linux, and FreeBSD.

I've borrowed code and ideas from Joe Taylor, Phil Karn, and others
identified in comments in the code.

The software depends on a few packages. Here's how to install them
on Ubuntu Linux:
```
  sudo apt-get install python2.7
  sudo apt-get install gcc-5
  sudo apt-get install python-numpy
  sudo apt-get install python-scipy
  sudo apt-get install python-pyaudio
  sudo apt-get install python-serial
```

If you have a Mac with macports:
```
  sudo port install python27
  sudo port install py27-numpy
  sudo port install py27-scipy
  sudo port install py27-pyaudio
  sudo port install py27-serial
```

Now compile Phil Karn's Reed-Solomon and convolutional decoders:
```
  (cd libfano ; make)
  (cd librs ; make)
```

At this point you should be able to run wsprmon.py and jt65mon.py with
no arguments, and see lists of available sound cards and serial ports,
like this:

```
  % python2.7 jt65mon.py
  Usage: jt65mon.py -in CARD:CHAN -cat type dev [-band BAND] [-levels]
  sound card numbers:
    0: Built-in Microph, channels=2
    1: Built-in Output, channels=0
    2: iMic USB audio system, channels=2
    or sdrip:IPADDR
    or sdriq:/dev/SERIALPORT
    or eb200:IPADDR
  serial devices:
    /dev/cu.Bluetooth-Incoming-Port
    /dev/cu.usbserial-FTXVKSG8A
    /dev/cu.usbserial-FTXVKSG8D
    /dev/cu.usbserial-FTXVKSG8B
    /dev/cu.usbserial-FTXVKSG8C
  radio types: k3 rx340 sdrip sdriq r75 r8500 ar5000 eb200
```

If I hook my radio's audio output up to my iMic USB sound card input,
and set the radio frequency to 14.076, I can monitor jt65 like this:

```
  % python2.7 jt65mon.py -in 2:0 -band 20
```

The "-in 2:0" means the left (0) channel of sound card number 2. After
a few minutes I might see output like this:

```
  01/08/16 21:46:50 20 rcv  0 19 -1 -16  780.6 CQ OE5CCN JN67
  01/08/16 21:46:49 20 rcv  0  4 -1 -15 1859.8 TA2NC   W0JMP EN34
```

The 3rd column is the band (20 meters), the 8th is the SNR, and the
9th is the offset in Hz.

If your radio is a K3, Icom R75 or R8500, Ten-Tex RX-340, or AOR AR-5000,
and it's connected by a serial connection, you can monitor JT65 while
switching bands periodically, e.g:

```
  % python2.7 jt65mon.py -in 2:0 -cat k3 /dev/cu.usbserial-FTXVKSG8A
```

You may need to take steps to give yourself permission to use the
serial device (change its mode or put yourself in the appropriate
group).

For an RFSpace SDR-IP or NetSDR at IP address 10.0.0.2:

```
  % python2.7 jt65mon.py -in sdrip:10.0.0.2 -cat sdrip 10.0.0.2
```

Use the -levels flag to help you adjust the audio level from the
radio. Peaks of a few thousand are good.

You must set your call sign and grid in order for the software to
report to wsprnet and pskreporter. Do this by copying weak.cfg.example
to weak.cfg, un-commenting the mycall and mygrid lines, and changing
them to your callsign and grid.

Your computer's clock must be correct to within a second for WSPR and
JT65 to be received; try ntp.

I've included receive software for WWVB's phase-shift modulation, and
for APRS.

This software surely contains errors, particularly since I'm no expert
at signal processing. I'd appreciate fixes for any bugs you discover.

Robert Morris, AB1HL
