#!/usr/local/bin/python

#
# interactive JT65.
# runs in a Linux or Mac terminal window.
# user can respond to CQs, but not send CQ.
# optional automatic band switching when not in QSO.
#
# I use jt65i.py with a K3S via USB, automatically switching
# among bands:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23
#
# To switch among just a few bands:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -bands "30 20 17"
#
# To use on a single band without CAT (radio must be set up
# correctly already, and have VOX if you want to transmit):
# ./jt65i.py -card 2 0 -out 1 -band 30
#
# Select a CQ to reply to by typing the letter displayed
# next to the CQ. jt65i.py automates the rest of the exchange.
#
# jt65i.py can use multiple receivers, listening to a different band on
# each; when you reply to a CQ it automatically sets the transmitter to
# the correct band. This works for a K3 with a sub-receiver, or a K3
# with SDRs that jt65i.py knows how to control.
#
# For a K3 with a sub-receiver:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -cat2 k3 - -bands "40 30 20"
#
# For a K3 without sub-receiver, and an RFSpace NetSDR/CloudIQ/SDR-IP:
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 sdrip 192.168.3.130 -bands "40 30 20"
#
# For a K3 with sub-receiver and an RFSpace NetSDR/CloudIQ/SDR-IP (i.e. three receivers):
# ./jt65i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -cat2 k3 - -card3 sdrip 192.168.3.130 -bands "40 30 20"
#
# Robert Morris, AB1HL
#

#import fake65 as jt65
import jt65
import weakdriver

# automatically switch only among these bands.
# auto_bands = [ "160", "80", "60", "40", "30", "20", "17", "15", "12", "10" ]
auto_bands = [ "40", "30", "20", "17" ]

frequencies = { "160" : 1.838, "80" : 3.576, "60" : 5.357, "40" : 7.076,
                "30" : 10.138, "20" : 14.076,
                "17" : 18.102, "15" : 21.076, "12" : 24.917,
                "10" : 28.076, "6" : 50.276 }
def main():
    weakdriver.driver_main("jt65i", jt65.JT65, jt65.JT65Send, frequencies,
                           auto_bands, "jt65", "FT")

main()
