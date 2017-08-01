# weakmon

This software implements terminal-window programs for HF JT65
(jt65i.py) and FT8 (ft8i.py). With an Elecraft K3, both can listen for CQs on
multiple receivers simultaneously, and switch the receiver(s) among
bands each minute.

There are also demodulators for WSPR, APRS, WWV, and WWVB. The
software works on Macs, Linux, and FreeBSD.

While these programs don't use Joe Taylor's WSJT software, they do
incorporate ideas and protocol details derived from that software.
These programs use Phil Karn's Reed-Solomon and convolutional
decoders.

The software depends on a few packages. Here's how to install them
on Ubuntu Linux:
```
  sudo apt-get install python2.7
  sudo apt-get install python-six
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

Now compile the LDPC decoder, and Phil Karn's Reed-Solomon and convolutional decoders:
```
  (cd libldpc ; make)
  (cd libfano ; make)
  (cd librs ; make)
```

At this point you should be able to run ft8i.py, jt65i.py, wsprmon.py, etc. with
no arguments, and see lists of available sound cards and serial ports,
like this:

```
% python2.7 ft8i.py
usage: ft8i.py [-h] [-card CARD CHAN] [-cat TYPE DEV] [-levels] [-v]
                [-band BAND] [-bands BANDS] [-card2 CARD CHAN]
                [-card3 CARD CHAN] [-card4 CARD CHAN] [-out CARD] [-test]
sound card numbers for -card and -out:
  0: Built-in Output, channels=0
  1: USB Audio CODEC , channels=0
  2: USB Audio CODEC , channels=2 11025 12000 16000 22050 44100 48000
  or -card sdrip IPADDR
  or -card sdriq /dev/SERIALPORT
  or -card eb200 IPADDR
  or -card sdrplay sdrplay
serial devices for -cat:
  /dev/cu.usbserial-A503XT23
  /dev/cu.Bluetooth-Incoming-Port
radio types for -cat: k3 rx340 8711 sdrip sdriq r75 r8500 ar5000 eb200 sdrplay prc138 
```

If you've hooked up a transceiver with VOX to your computer's sound
card, and set it to 14.074 and USB, you can use ft8i.py like
this:

```
  % python2.7 ft8i.py -card 2 0 -out 1 -band 20

```

The "-card 2 0" means the left (0) channel of sound card number 2 (as
listed in ft8i.py's "usage" output). The "-out 1" means sound card 1.

ft8i.py will display decoded messages, and mark each received CQ
with a letter; type the letter to respond to the CQ.

ft8i.py can automatically switch among a set of bands, once per
minute. For example, this command will tell a K3 to scan 30, 20, and
17 meters for JT65.

```
  % python2.7 ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -bands "30 20 17"
```

ft8i.py can listen to multiple receivers at the same time, so that
you can look for CQs on more than one band simultaneously. For
example, for a K3 with a sub-receiver:

```
  % python2.7 ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -cat2 k3 - -bands "40 30 20"
```

For a K3 (without sub-receiver) and an RFSpace NetSDR/CloudIQ/SDR-IP:

```
  % python2.7 ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 sdrip 192.168.3.130 -bands "40 30 20"
```

For a K3 with sub-receiver and an RFSpace NetSDR/CloudIQ/SDR-IP (i.e. three receivers):

```
  % python2.7 ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -cat2 k3 - -card3 sdrip 192.168.3.130 -bands "40 30 20"
```

You may need to take steps to give yourself permission to use the
serial device (change its mode or put yourself in the appropriate
group).

The aprsmon.py, jt65mon.py, wsprmon.py, wwvbmon.py, and wwvmon.py
programs each decode and display receptions for the respective format.
They use argument conventions similar to those of ft8i.py.

These programs can switch among bands on a number of radio types:
Elecraft K3, Ten-Tec RX-340, Watkins Johnson WJ-8711, RFSpace SDR-IP,
RFSpace CloudIQ, RFSpace NetSDR, RFSpace SDR-IQ, Icom R75, Icom R8500,
AOR AR5000, Rohde and Schwarz EB200, SDRplay, and Harris PRC-138.

Use the -levels flag to help adjust the audio level from the radio.
Peaks of a few thousand are good.

You must set your call sign and grid to send with jt65i.py or ft8i.py,
or to report to wsprnet and pskreporter. Do this by copying
weak.cfg.example to weak.cfg, un-commenting the mycall and mygrid
lines, and changing them to your callsign and grid.

Your computer's clock must be correct to within about second for WSPR,
JT65, and FT8; try ntp.

This software surely contains errors, particularly since I'm no expert
at signal processing. I'd appreciate fixes for any bugs you discover.

Robert Morris, AB1HL
