#!/usr/local/bin/python

#
# interactive FT8.
# runs in a Linux or Mac terminal window.
# user can respond to CQs, though not send CQ.
# optional automatic band switching when not in QSO.
#
# I use ft8i.py with a K3S via USB, automatically switching
# among bands:
# ./ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23
#
# To switch among just a few bands:
# ./ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -bands "30 20 17"
#
# To use on a single band without CAT (radio must be set up
# correctly already, and have VOX if you want to transmit):
# ./ft8i.py -card 2 0 -out 1 -band 30
#
# Select a CQ to reply to by typing the letter displayed
# next to the CQ. ft8i.py automates the rest of the exchange.
#
# ft8i.py can use multiple receivers, listening to a different band on
# each; when you reply to a CQ it automatically sets the transmitter to
# the correct band. This works for a K3 with a sub-receiver, or a K3
# with SDRs that ft8i.py knows how to control.
#
# For a K3 with a sub-receiver:
# ./ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -cat2 k3 - -bands "40 30 20"
#
# For a K3 without sub-receiver, and an RFSpace NetSDR/CloudIQ/SDR-IP:
# ./ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 sdrip 192.168.3.130 -bands "40 30 20"
#
# For a K3 with sub-receiver and an RFSpace NetSDR/CloudIQ/SDR-IP (i.e. three receivers):
# ./ft8i.py -card 2 0 -out 1 -cat k3 /dev/cu.usbserial-A503XT23 -card2 2 1 -cat2 k3 - -card3 sdrip 192.168.3.130 -bands "40 30 20"
#
# Robert Morris, AB1HL
#

#import fakeft8 as ft8
import ft8
import weakdriver

# automatically switch only among these bands.
# auto_bands = [ "160", "80", "60", "40", "30", "20", "17", "15", "12", "10" ]
auto_bands = [ "40", "30", "20", "17", "15", "12", "10" ]

frequencies = { "160" : 1.840, "80" : 3.573, "40" : 7.074,
                "30" : 10.136, "20" : 14.074,
                "17" : 18.100, "15" : 21.074, "12" : 24.915,
                "10" : 28.074, "6" : 50.313 }

def main():
    weakdriver.driver_main("ft8i", ft8.FT8, ft8.FT8Send, frequencies,
                           auto_bands, "ft8", "FT")

main()
