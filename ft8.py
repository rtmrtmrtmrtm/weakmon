#!/usr/local/bin/python

#
# encode and decode FT8.
#
# for new 77-bit FT8.
# LDPC tables and pack/unpack details from wsjt-x 2.0.
#
# Robert Morris, AB1HL
#

import numpy
import wave
import scipy
import scipy.signal
import sys
import os
import math
import time
import multiprocessing
import threading
import re
import random
import ctypes
import weakaudio
import weakutil

#
# tuning parameters.
#
budget = 2.2 # max seconds of time for decoding.
pass0_fstep = 2 # coarse search granularity, per FFT bin
pass0_tstep = 2 # coarse search granularity, per symbol time
passN_fstep = 2 # coarse search granularity, per FFT bin
passN_tstep = 4 # coarse search granularity, per symbol time
pass0_tminus = 2.2 # start search this many seconds before 0.5
pass0_tplus = 2.7 # end search this many seconds after 0.5
passN_tminus = 1.5
passN_tplus = 1.8
coarse_no    = 1 # number of best offsets to use per hz
fine_no    = 1 # number of best fine offsets to look at
fine_fstep = 4 # fine-tuning steps per coarse_fstep
fine_tstep = 8 # fine-tuning steps per coarse_tstep
start_adj = 0.1 # signals seem on avg to start this many seconds late.
ldpc_iters = 30 # how hard LDPC should work on pass 0
softboost = 1.0 # log(prob) if #2 symbol has same bit value
do_subtract = 1 # 0 none, 1 once per unique decode, 2 three per unique, 3 once per decode
subgap = 0.8  # extra subtract()s this many hz on either side of main bin
substeps = 16 # subtract phase steps, in 2pi
subpasses = 2 # 0 means no subtraction, 1 means subtract, 2 means another subtraction pass
pass0_frac = 1.0
pass0_hints = True # hints in pass 0 (as well as later passes)?
contrast_weight = 0.5
noise_factor = 0.8 # for strength()
top_high_order = 0 # 0 for cheby, 19 for butter
high_cutoff = 1.05
low_pass_order = 0 # 15
top_down = True
bottom_slow = True
osd_crc = False # True means OSD only accepts if CRC is correct
osd0_crc = True # True means OSD accepts if depth=0 CRC is correct
osd_depth = 6
osd_thresh = -500
already_o = 1
already_f = 1
down200 = False # process1() that down-converts to 200 hz / 32 samples/symbol
use_apriori = True
nchildren = 4
child_overlap = 30
osd_no_snr = False
padfactor = 0.1 # quiet-ish before/after padding
osd_hints = False # use OSD on hints BUT causes lots of false pseudo-juicy CQs!
down_cutoff = 0.45 # low-pass filter cutoff before down-sampling
cheb_cut1 = 0.48
cheb_cut2 = 0.61
cheb_ripple_pass = 0.5
cheb_atten_stop = 50
cheb_high_minus = 40
cheb_high_plus = 60
hint_tol = 9 # look for CQ XXX hints in this +/- hz range of where heard
crc_and_83 = True # True means require both CRC and LDPC
ldpc_thresh = 83 # 83 means all LDPC check-bits must be correct
snr_overlap = 3 # -1 means don't convert to snr, 0 means each sym time by itself
snr_wintype = "blackman"
real_min_hz = 150
real_max_hz = 2900
sub_amp_win = 2
adjust_hz_for_sub = True
adjust_off_for_sub = True
yes_mul = 1.0
yes_add = 0.0
no_mul = 1.0
no_add = 0.0
soft1 = 7
soft2 = 8
soft3 = 4
soft4 = 6
guard200 = 10
order200 = 5
strength_div = 4.0
decimate_order = 8

# FT8 modulation and protocol definitions.
# 1920-point FFT at 12000 samples/second
#   yields 6.25 Hz spacing, 0.16 seconds/symbol
# encode chain:
#   77 bits
#   append 14 bits CRC (for 91 bits)
#   LDPC(174,91) yields 174 bits
#   that's 58 3-bit FSK-8 symbols
#   gray code each 3 bits
#   insert three 7-symbol Costas sync arrays
#     at symbol #s 0, 36, 72 of final signal
#   thus: 79 FSK-8 symbols
# total transmission time is 12.64 seconds

costas_symbols = [ 3, 1, 4, 0, 6, 5, 2 ] # new FT8

# gray map for encoding 3-bit chunks of the 174 bits,
# after LDPC and before generating FSK-8 tones.
graymap = [ 0, 1, 3, 2, 5, 6, 4, 7 ]

# the CRC-14 polynomial, from wsjt-x's 0x2757,
# with leading 1 bit.
crc14poly = [ 1,   1, 0,   0, 1, 1, 1,   0, 1, 0, 1,   0, 1, 1, 1 ]

def crc_c(msg):
    msgtype = ctypes.c_int * len(msg)
    outtype = ctypes.c_int * 14

    msg1 = msgtype()
    for i in range(0, len(msg)):
        msg1[i] = msg[i]

    out1 = outtype()

    libldpc.ft8_crc(msg1, len(msg), out1)

    out = numpy.zeros(14, dtype=numpy.int32)
    for i in range(0, 14):
        out[i] = out1[i]

    return out

#
# thank you, evan sneath.
# https://gist.github.com/evansneath/4650991
#
# generate with x^3 + x + 1:
#   >>> xc.crc([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1])
#   array([1, 0, 0])
# check:
#   >>> xc.crc([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0])
#   array([0, 0, 0])
#
# 0xc06 is really 0x1c06 or [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 ]
#
def crc_python(msg, div, code=None):
    """Cyclic Redundancy Check
    Generates an error detecting code based on an inputted message
    and divisor in the form of a polynomial representation.
    Arguments:
        msg: The input message of which to generate the output code.
        div: The divisor in polynomial form. For example, if the polynomial
            of x^3 + x + 1 is given, this should be represented as '1011' in
            the div argument.
        code: This is an option argument where a previously generated code may
            be passed in. This can be used to check validity. If the inputted
            code produces an outputted code of all zeros, then the message has
            no errors.
    Returns:
        An error-detecting code generated by the message and the given divisor.
    """

    # Append the code to the message. If no code is given, default to '000'
    if code is None:
        code = numpy.zeros(len(div)-1, dtype=numpy.int32)
    assert len(code) == len(div) - 1
    msg = numpy.append(msg, code)

    div = numpy.array(div, dtype=numpy.int32)
    divlen = len(div)

    # Loop over every message bit (minus the appended code)
    for i in range(len(msg)-len(code)):
        # If that messsage bit is 1, perform modulo 2 multiplication
        if msg[i] == 1:
            #for j in range(len(div)):
            #    # Perform modulo 2 multiplication on each index of the divisor
            #    msg[i+j] = (msg[i+j] + div[j]) % 2
            msg[i:i+divlen] = numpy.mod(msg[i:i+divlen] + div, 2)

    # Output the last error-checking code portion of the message generated
    return msg[-len(code):]

def crc(msg, div):
    if True:
        return crc_c(msg)
    else:
        return crc_python(msg, div)

def check_crc(a91):
    padded = numpy.append(a91[0:77], numpy.zeros(5, dtype=numpy.int32))
    cksum = crc(padded, crc14poly)
    if numpy.array_equal(cksum, a91[-14:]) == False:
        # CRC failed.
        return False
    return True

# this is the LDPC(174,91) parity check matrix.
# each row describes one parity check.
# each number is an index into the codeword (1-origin).
# the codeword bits mentioned in each row must xor to zero.
# From WSJT-X's ldpc_174_91_c_reordered_parity.f90
Nm = [
  [   4,  31,  59,  91,  92,  96, 153 ],
  [   5,  32,  60,  93, 115, 146,   0 ],
  [   6,  24,  61,  94, 122, 151,   0 ],
  [   7,  33,  62,  95,  96, 143,   0 ],
  [   8,  25,  63,  83,  93,  96, 148 ],
  [   6,  32,  64,  97, 126, 138,   0 ],
  [   5,  34,  65,  78,  98, 107, 154 ],
  [   9,  35,  66,  99, 139, 146,   0 ],
  [  10,  36,  67, 100, 107, 126,   0 ],
  [  11,  37,  67,  87, 101, 139, 158 ],
  [  12,  38,  68, 102, 105, 155,   0 ],
  [  13,  39,  69, 103, 149, 162,   0 ],
  [   8,  40,  70,  82, 104, 114, 145 ],
  [  14,  41,  71,  88, 102, 123, 156 ],
  [  15,  42,  59, 106, 123, 159,   0 ],
  [   1,  33,  72, 106, 107, 157,   0 ],
  [  16,  43,  73, 108, 141, 160,   0 ],
  [  17,  37,  74,  81, 109, 131, 154 ],
  [  11,  44,  75, 110, 121, 166,   0 ],
  [  45,  55,  64, 111, 130, 161, 173 ],
  [   8,  46,  71, 112, 119, 166,   0 ],
  [  18,  36,  76,  89, 113, 114, 143 ],
  [  19,  38,  77, 104, 116, 163,   0 ],
  [  20,  47,  70,  92, 138, 165,   0 ],
  [   2,  48,  74, 113, 128, 160,   0 ],
  [  21,  45,  78,  83, 117, 121, 151 ],
  [  22,  47,  58, 118, 127, 164,   0 ],
  [  16,  39,  62, 112, 134, 158,   0 ],
  [  23,  43,  79, 120, 131, 145,   0 ],
  [  19,  35,  59,  73, 110, 125, 161 ],
  [  20,  36,  63,  94, 136, 161,   0 ],
  [  14,  31,  79,  98, 132, 164,   0 ],
  [   3,  44,  80, 124, 127, 169,   0 ],
  [  19,  46,  81, 117, 135, 167,   0 ],
  [   7,  49,  58,  90, 100, 105, 168 ],
  [  12,  50,  61, 118, 119, 144,   0 ],
  [  13,  51,  64, 114, 118, 157,   0 ],
  [  24,  52,  76, 129, 148, 149,   0 ],
  [  25,  53,  69,  90, 101, 130, 156 ],
  [  20,  46,  65,  80, 120, 140, 170 ],
  [  21,  54,  77, 100, 140, 171,   0 ],
  [  35,  82, 133, 142, 171, 174,   0 ],
  [  14,  30,  83, 113, 125, 170,   0 ],
  [   4,  29,  68, 120, 134, 173,   0 ],
  [   1,   4,  52,  57,  86, 136, 152 ],
  [  26,  51,  56,  91, 122, 137, 168 ],
  [  52,  84, 110, 115, 145, 168,   0 ],
  [   7,  50,  81,  99, 132, 173,   0 ],
  [  23,  55,  67,  95, 172, 174,   0 ],
  [  26,  41,  77, 109, 141, 148,   0 ],
  [   2,  27,  41,  61,  62, 115, 133 ],
  [  27,  40,  56, 124, 125, 126,   0 ],
  [  18,  49,  55, 124, 141, 167,   0 ],
  [   6,  33,  85, 108, 116, 156,   0 ],
  [  28,  48,  70,  85, 105, 129, 158 ],
  [   9,  54,  63, 131, 147, 155,   0 ],
  [  22,  53,  68, 109, 121, 174,   0 ],
  [   3,  13,  48,  78,  95, 123,   0 ],
  [  31,  69, 133, 150, 155, 169,   0 ],
  [  12,  43,  66,  89,  97, 135, 159 ],
  [   5,  39,  75, 102, 136, 167,   0 ],
  [   2,  54,  86, 101, 135, 164,   0 ],
  [  15,  56,  87, 108, 119, 171,   0 ],
  [  10,  44,  82,  91, 111, 144, 149 ],
  [  23,  34,  71,  94, 127, 153,   0 ],
  [  11,  49,  88,  92, 142, 157,   0 ],
  [  29,  34,  87,  97, 147, 162,   0 ],
  [  30,  50,  60,  86, 137, 142, 162 ],
  [  10,  53,  66,  84, 112, 128, 165 ],
  [  22,  57,  85,  93, 140, 159,   0 ],
  [  28,  32,  72, 103, 132, 166,   0 ],
  [  28,  29,  84,  88, 117, 143, 150 ],
  [   1,  26,  45,  80, 128, 147,   0 ],
  [  17,  27,  89, 103, 116, 153,   0 ],
  [  51,  57,  98, 163, 165, 172,   0 ],
  [  21,  37,  73, 138, 152, 169,   0 ],
  [  16,  47,  76, 130, 137, 154,   0 ],
  [   3,  24,  30,  72, 104, 139,   0 ],
  [   9,  40,  90, 106, 134, 151,   0 ],
  [  15,  58,  60,  74, 111, 150, 163 ],
  [  18,  42,  79, 144, 146, 152,   0 ],
  [  25,  38,  65,  99, 122, 160,   0 ],
  [  17,  42,  75, 129, 170, 172,   0 ],
]

# Mn from WSJT-X's ldpc_174_91_c_reordered_parity.f90
# each of the 174 rows corresponds to a codeword bit.
# the numbers indicate which three parity
# checks (rows in Nm) refer to the codeword bit.
# 1-origin.
Mn = [
  [  16,  45,  73 ],
  [  25,  51,  62 ],
  [  33,  58,  78 ],
  [   1,  44,  45 ],
  [   2,   7,  61 ],
  [   3,   6,  54 ],
  [   4,  35,  48 ],
  [   5,  13,  21 ],
  [   8,  56,  79 ],
  [   9,  64,  69 ],
  [  10,  19,  66 ],
  [  11,  36,  60 ],
  [  12,  37,  58 ],
  [  14,  32,  43 ],
  [  15,  63,  80 ],
  [  17,  28,  77 ],
  [  18,  74,  83 ],
  [  22,  53,  81 ],
  [  23,  30,  34 ],
  [  24,  31,  40 ],
  [  26,  41,  76 ],
  [  27,  57,  70 ],
  [  29,  49,  65 ],
  [   3,  38,  78 ],
  [   5,  39,  82 ],
  [  46,  50,  73 ],
  [  51,  52,  74 ],
  [  55,  71,  72 ],
  [  44,  67,  72 ],
  [  43,  68,  78 ],
  [   1,  32,  59 ],
  [   2,   6,  71 ],
  [   4,  16,  54 ],
  [   7,  65,  67 ],
  [   8,  30,  42 ],
  [   9,  22,  31 ],
  [  10,  18,  76 ],
  [  11,  23,  82 ],
  [  12,  28,  61 ],
  [  13,  52,  79 ],
  [  14,  50,  51 ],
  [  15,  81,  83 ],
  [  17,  29,  60 ],
  [  19,  33,  64 ],
  [  20,  26,  73 ],
  [  21,  34,  40 ],
  [  24,  27,  77 ],
  [  25,  55,  58 ],
  [  35,  53,  66 ],
  [  36,  48,  68 ],
  [  37,  46,  75 ],
  [  38,  45,  47 ],
  [  39,  57,  69 ],
  [  41,  56,  62 ],
  [  20,  49,  53 ],
  [  46,  52,  63 ],
  [  45,  70,  75 ],
  [  27,  35,  80 ],
  [   1,  15,  30 ],
  [   2,  68,  80 ],
  [   3,  36,  51 ],
  [   4,  28,  51 ],
  [   5,  31,  56 ],
  [   6,  20,  37 ],
  [   7,  40,  82 ],
  [   8,  60,  69 ],
  [   9,  10,  49 ],
  [  11,  44,  57 ],
  [  12,  39,  59 ],
  [  13,  24,  55 ],
  [  14,  21,  65 ],
  [  16,  71,  78 ],
  [  17,  30,  76 ],
  [  18,  25,  80 ],
  [  19,  61,  83 ],
  [  22,  38,  77 ],
  [  23,  41,  50 ],
  [   7,  26,  58 ],
  [  29,  32,  81 ],
  [  33,  40,  73 ],
  [  18,  34,  48 ],
  [  13,  42,  64 ],
  [   5,  26,  43 ],
  [  47,  69,  72 ],
  [  54,  55,  70 ],
  [  45,  62,  68 ],
  [  10,  63,  67 ],
  [  14,  66,  72 ],
  [  22,  60,  74 ],
  [  35,  39,  79 ],
  [   1,  46,  64 ],
  [   1,  24,  66 ],
  [   2,   5,  70 ],
  [   3,  31,  65 ],
  [   4,  49,  58 ],
  [   1,   4,   5 ],
  [   6,  60,  67 ],
  [   7,  32,  75 ],
  [   8,  48,  82 ],
  [   9,  35,  41 ],
  [  10,  39,  62 ],
  [  11,  14,  61 ],
  [  12,  71,  74 ],
  [  13,  23,  78 ],
  [  11,  35,  55 ],
  [  15,  16,  79 ],
  [   7,   9,  16 ],
  [  17,  54,  63 ],
  [  18,  50,  57 ],
  [  19,  30,  47 ],
  [  20,  64,  80 ],
  [  21,  28,  69 ],
  [  22,  25,  43 ],
  [  13,  22,  37 ],
  [   2,  47,  51 ],
  [  23,  54,  74 ],
  [  26,  34,  72 ],
  [  27,  36,  37 ],
  [  21,  36,  63 ],
  [  29,  40,  44 ],
  [  19,  26,  57 ],
  [   3,  46,  82 ],
  [  14,  15,  58 ],
  [  33,  52,  53 ],
  [  30,  43,  52 ],
  [   6,   9,  52 ],
  [  27,  33,  65 ],
  [  25,  69,  73 ],
  [  38,  55,  83 ],
  [  20,  39,  77 ],
  [  18,  29,  56 ],
  [  32,  48,  71 ],
  [  42,  51,  59 ],
  [  28,  44,  79 ],
  [  34,  60,  62 ],
  [  31,  45,  61 ],
  [  46,  68,  77 ],
  [   6,  24,  76 ],
  [   8,  10,  78 ],
  [  40,  41,  70 ],
  [  17,  50,  53 ],
  [  42,  66,  68 ],
  [   4,  22,  72 ],
  [  36,  64,  81 ],
  [  13,  29,  47 ],
  [   2,   8,  81 ],
  [  56,  67,  73 ],
  [   5,  38,  50 ],
  [  12,  38,  64 ],
  [  59,  72,  80 ],
  [   3,  26,  79 ],
  [  45,  76,  81 ],
  [   1,  65,  74 ],
  [   7,  18,  77 ],
  [  11,  56,  59 ],
  [  14,  39,  54 ],
  [  16,  37,  66 ],
  [  10,  28,  55 ],
  [  15,  60,  70 ],
  [  17,  25,  82 ],
  [  20,  30,  31 ],
  [  12,  67,  68 ],
  [  23,  75,  80 ],
  [  27,  32,  62 ],
  [  24,  69,  75 ],
  [  19,  21,  71 ],
  [  34,  53,  61 ],
  [  35,  46,  47 ],
  [  33,  59,  76 ],
  [  40,  43,  83 ],
  [  41,  42,  63 ],
  [  49,  75,  83 ],
  [  20,  44,  48 ],
  [  42,  49,  57 ],
]

#
# LDPC generator matrix from WSJT-X's ldpc_174_91_c_generator.f90.
# 83 rows, since LDPC(174,91) needs 83 parity bits.
# each row has 23 hex digits, to be turned into 91 bits,
# to be xor'd with the 91 data bits.
#
rawg = [
  "8329ce11bf31eaf509f27fc", 
  "761c264e25c259335493132", 
  "dc265902fb277c6410a1bdc", 
  "1b3f417858cd2dd33ec7f62", 
  "09fda4fee04195fd034783a", 
  "077cccc11b8873ed5c3d48a", 
  "29b62afe3ca036f4fe1a9da", 
  "6054faf5f35d96d3b0c8c3e", 
  "e20798e4310eed27884ae90", 
  "775c9c08e80e26ddae56318", 
  "b0b811028c2bf997213487c", 
  "18a0c9231fc60adf5c5ea32", 
  "76471e8302a0721e01b12b8", 
  "ffbccb80ca8341fafb47b2e", 
  "66a72a158f9325a2bf67170", 
  "c4243689fe85b1c51363a18", 
  "0dff739414d1a1b34b1c270", 
  "15b48830636c8b99894972e", 
  "29a89c0d3de81d665489b0e", 
  "4f126f37fa51cbe61bd6b94", 
  "99c47239d0d97d3c84e0940", 
  "1919b75119765621bb4f1e8", 
  "09db12d731faee0b86df6b8", 
  "488fc33df43fbdeea4eafb4", 
  "827423ee40b675f756eb5fe", 
  "abe197c484cb74757144a9a", 
  "2b500e4bc0ec5a6d2bdbdd0", 
  "c474aa53d70218761669360", 
  "8eba1a13db3390bd6718cec", 
  "753844673a27782cc42012e", 
  "06ff83a145c37035a5c1268", 
  "3b37417858cc2dd33ec3f62", 
  "9a4a5a28ee17ca9c324842c", 
  "bc29f465309c977e89610a4", 
  "2663ae6ddf8b5ce2bb29488", 
  "46f231efe457034c1814418", 
  "3fb2ce85abe9b0c72e06fbe", 
  "de87481f282c153971a0a2e", 
  "fcd7ccf23c69fa99bba1412", 
  "f0261447e9490ca8e474cec", 
  "4410115818196f95cdd7012", 
  "088fc31df4bfbde2a4eafb4", 
  "b8fef1b6307729fb0a078c0", 
  "5afea7acccb77bbc9d99a90", 
  "49a7016ac653f65ecdc9076", 
  "1944d085be4e7da8d6cc7d0", 
  "251f62adc4032f0ee714002", 
  "56471f8702a0721e00b12b8", 
  "2b8e4923f2dd51e2d537fa0", 
  "6b550a40a66f4755de95c26", 
  "a18ad28d4e27fe92a4f6c84", 
  "10c2e586388cb82a3d80758", 
  "ef34a41817ee02133db2eb0", 
  "7e9c0c54325a9c15836e000", 
  "3693e572d1fde4cdf079e86", 
  "bfb2cec5abe1b0c72e07fbe", 
  "7ee18230c583cccc57d4b08", 
  "a066cb2fedafc9f52664126", 
  "bb23725abc47cc5f4cc4cd2", 
  "ded9dba3bee40c59b5609b4", 
  "d9a7016ac653e6decdc9036", 
  "9ad46aed5f707f280ab5fc4", 
  "e5921c77822587316d7d3c2", 
  "4f14da8242a8b86dca73352", 
  "8b8b507ad467d4441df770e", 
  "22831c9cf1169467ad04b68", 
  "213b838fe2ae54c38ee7180", 
  "5d926b6dd71f085181a4e12", 
  "66ab79d4b29ee6e69509e56", 
  "958148682d748a38dd68baa", 
  "b8ce020cf069c32a723ab14", 
  "f4331d6d461607e95752746", 
  "6da23ba424b9596133cf9c8", 
  "a636bcbc7b30c5fbeae67fe", 
  "5cb0d86a07df654a9089a20", 
  "f11f106848780fc9ecdd80a", 
  "1fbb5364fb8d2c9d730d5ba", 
  "fcb86bc70a50c9d02a5d034", 
  "a534433029eac15f322e34c", 
  "c989d9c7c3d3b8c55d75130", 
  "7bb38b2f0186d46643ae962", 
  "2644ebadeb44b9467d1f42c", 
  "608cc857594bfbb55d69600"
]

# gen[row][col], derived from rawg, has one row per
# parity bit, to be xor'd with the 91 data bits.
# thus gen[83][91].
# as in encode174_91.f90
gen = [ ]

# turn rawg into gen.
def make_gen():
    global gen

    # hex digit to number
    hex2 = { }
    for i in range(0, 16):
        hex2[hex(i)[2]] = i

    assert len(rawg) == 83

    for e in rawg:
        row = numpy.zeros(91, dtype=numpy.int32)
        for i,c in enumerate(e):
            x = hex2[c]
            for j in range(0, 4):
                ind = i*4 + (3-j)
                if ind >= 0 and ind < 91:
                    if (x & (1 << j)) != 0:
                        row[ind] = 1
                    else:
                        row[ind] = 0
        gen.append(row)

make_gen()

# turn gen[] into a systematic array by prepending
# a 91x91 identity matrix.
gen_sys = numpy.zeros((174, 91), dtype=numpy.int32)
gen_sys[91:,:] = gen
gen_sys[0:91,:] = numpy.eye(91, dtype=numpy.int32)

# plain is 91 bits of plain-text.
# returns a 174-bit codeword.
# mimics wsjt-x's encode174_91.f90.
def ldpc_encode(plain):
    assert len(plain) == 91

    ncw = numpy.zeros(174, dtype=numpy.int32)
    numpy.dot(gen_sys[91:,:], plain, out=ncw[91:])
    numpy.mod(ncw[91:], 2, out=ncw[91:])
    ncw[0:91] = plain

    return ncw

# given a 174-bit codeword as an array of log-likelihood of zero,
# return a 91-bit plain text, or zero-length array.
# this is an implementation of the sum-product algorithm
# from Sarah Johnson's Iterative Error Correction book.
# codeword[i] = log ( P(x=0) / P(x=1) )
# returns [ nok, plain ], where nok is the number of parity
# checks that worked out, should be 83=174-91.
def ldpc_decode_python(codeword, ldpc_iters):
    # 174 codeword bits:
    #   91 systematic data bits
    #   83 parity checks

    mnx = numpy.array(Mn, dtype=numpy.int32)
    nmx = numpy.array(Nm, dtype=numpy.int32)

    # Mji
    # each codeword bit i tells each parity check j
    # what the bit's log-likelihood of being 0 is
    # based on information *other* than from that
    # parity check.
    m = numpy.zeros((83, 174))

    for i in range(0, 174):
        for j in range(0, 83):
            m[j][i] = codeword[i]

    for iter in range(0, ldpc_iters):
        # Eji
        # each check j tells each codeword bit i the
        # log likelihood of the bit being zero based
        # on the *other* bits in that check.
        e = numpy.zeros((83, 174))

        # messages from checks to bits.
        # for each parity check
        #for j in range(0, 83):
        #    # for each bit mentioned in this parity check
        #    for i in Nm[j]:
        #        if i <= 0:
        #            continue
        #        a = 1
        #        # for each other bit mentioned in this parity check
        #        for ii in Nm[j]:
        #            if ii != i:
        #                a *= math.tanh(m[j][ii-1] / 2.0)
        #        e[j][i-1] = math.log((1 + a) / (1 - a))
        for i in range(0, 7):
            a = numpy.ones(83)
            for ii in range(0, 7):
                if ii != i:
                    x1 = numpy.tanh(m[range(0, 83), nmx[:,ii]-1] / 2.0)
                    x2 = numpy.where(numpy.greater(nmx[:,ii], 0.0), x1, 1.0)
                    a = a * x2
            # avoid divide by zero, i.e. a[i]==1.0
            # XXX why is a[i] sometimes 1.0?
            b = numpy.where(numpy.less(a, 0.99999), a, 0.99)
            c = numpy.log((b + 1.0) / (1.0 - b))
            # have assign be no-op when nmx[a,b] == 0
            d = numpy.where(numpy.equal(nmx[:,i], 0),
                            e[range(0,83), nmx[:,i]-1],
                            c)
            e[range(0,83), nmx[:,i]-1] = d

        # decide if we are done -- compute the corrected codeword,
        # see if the parity check succeeds.
        # sum the three log likelihoods contributing to each codeword bit.
        e0 = e[mnx[:,0]-1, range(0,174)]
        e1 = e[mnx[:,1]-1, range(0,174)]
        e2 = e[mnx[:,2]-1, range(0,174)]
        ll = codeword + e0 + e1 + e2
        # log likelihood > 0 => bit=0.
        cw = numpy.select( [ ll < 0 ], [ numpy.ones(174, dtype=numpy.int32) ])
        if ldpc_check(cw):
            # success!
            # it's a systematic code, though the plain-text bits are scattered.
            # collect them.
            decoded = cw[0:91]
            return [ 91, decoded ]

        # messages from bits to checks.
        for j in range(0, 3):
            # for each column in Mn.
            ll = codeword
            if j != 0:
                e0 = e[mnx[:,0]-1, range(0,174)]
                ll = ll + e0
            if j != 1:
                e1 = e[mnx[:,1]-1, range(0,174)]
                ll = ll + e1
            if j != 2:
                e2 = e[mnx[:,2]-1, range(0,174)]
                ll = ll + e2
            m[mnx[:,j]-1, range(0,174)] = ll


    # could not decode.
    return [ 0, numpy.array([]) ]

# turn log-likelihood bits into hard bits.
# codeword[i] = log ( P(x=0) / P(x=1) )
# so > 0 means bit=0, < 0 means bit=1.
def soft2hard(codeword):
    hard = numpy.less(codeword, 0.0)
    hard = numpy.array(hard, dtype=numpy.int32) # T/F -> 1/0
    two = numpy.array([0, 1], dtype=numpy.int32)
    hardword = two[hard]
    return hardword

# given a 174-bit codeword as an array of log-likelihood of zero,
# return a 91-bit plain text, or zero-length array.
# this is an implementation of the bit-flipping algorithm
# from Sarah Johnson's Iterative Error Correction book.
# codeword[i] = log ( P(x=0) / P(x=1) )
# returns [ nok, plain ], where nok is the number of parity
# checks that worked out, should be 83=174-91.
def ldpc_decode_flipping(codeword):
    cw = soft2hard(codeword)

    for iter in range(0,100):
        # for each codeword bit,
        # count of votes for 0 and 1.
        votes = numpy.zeros((len(codeword), 2))

        # for each parity check equation.
        for e in Nm:
            # for each codeword bit mentioned in e.
            for bi in e:
                if bi == 0:
                    continue
                # value for bi implied by remaining bits.
                x = 0
                for i in e:
                    if i != bi:
                        x ^= cw[i-1]
                # the other bits in the equation suggest that
                # bi must have value x.
                votes[(bi-1),x] += 1

        for i in range(0, len(cw)):
            if cw[i] == 0 and votes[i][1] > votes[i][0]:
                cw[i] = 1
            elif cw[i] == 1 and votes[i][0] > votes[i][1]:
                cw[i] = 0

        if ldpc_check(cw):
            # success!
            # it's a systematic code; data is first 91 bits.
            return [ 91, cw[0:91] ]

    return [ 0, numpy.array([]) ]

# does a 174-bit codeword pass the LDPC parity checks?
def ldpc_check(codeword):
    for e in Nm:
        x = 0
        for i in e:
            if i != 0:
                x ^= codeword[i-1]
        if x != 0:
            return False
    return True

libldpc = None
try:
    libldpc = ctypes.cdll.LoadLibrary("libldpc/libldpc.so")
except:
    libldpc = None
    sys.stderr.write("ft8: using the Python LDPC decoder, not the C decoder.\n")

if False:
    # test CRC
    # nov 30 2018: crc12 works
    # dec  3 2018: crc14 works
    msg = numpy.zeros(82, dtype=numpy.int32)
    msg[3] = 1
    msg[7] = 1
    msg[44] = 1
    msg[45] = 1
    msg[46] = 1
    msg[51] = 1
    msg[61] = 1
    msg[71] = 1

    expected = [ 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, ]

    cksum = crc_python(msg, crc14poly)
    eq = numpy.equal(cksum, numpy.array(expected, dtype=numpy.int32))
    assert numpy.all(eq)

    cksum = crc_c(msg)
    eq = numpy.equal(cksum, numpy.array(expected, dtype=numpy.int32))
    assert numpy.all(eq)

    sys.exit(1)

def ldpc_test(ldpci):
    tt = 0.0
    niters = 5000
    ok = 0
    for iter in range(0, niters):
        # ldpc_encode() takes 91 bits.
        a91 = numpy.random.randint(0, 2, 91, dtype=numpy.int32)
        a174 = ldpc_encode(a91)

        if True:
            # check that ldpc_encode() generated the right parity bits.
            assert ldpc_check(a174)

        # turn hard bits into 0.99 vs 0.01 log-likelihood,
        # log( P(0) / P(1) )
        # log base e.
        two = numpy.array([ 4.6, -4.6 ])
        ll174 = two[a174]

        if True:
            # check decode is perfect before wrecking bits.
            [ nn, d91 ] = ldpc_decode(ll174, ldpci)
            assert numpy.array_equal(a91, d91)
            assert nn == 83

        # wreck some bits
        #for junk in range(0, 70):
        #    ll174[random.randint(0, len(ll174)-1)] = (random.random() - 0.5) * 4

        perm = numpy.random.permutation(len(ll174))
        perm = perm[0:70]
        for i in perm:
            p = random.random()
            bit = a174[i]
            if random.random() > p:
                # flip the bit
                bit = 1 - bit
            if bit == 0:
                p = 0.5 + (p / 2)
            else:
                p = 0.5 - (p / 2)
            ll = math.log(p / (1.0 - p))
            ll174[i] = ll

        t0 = time.time()

        # decode LDPC(174,91)
        [ _, d91 ] = ldpc_decode(ll174, ldpci)

        t1 = time.time()
        tt += t1 - t0

        if numpy.array_equal(a91, d91):
            ok += 1

    print("ldpc_iters %d, success %.2f, %.6f sec/call" % (ldpci,
                                                          ok / float(niters),
                                                          tt / niters))

    # success 0.88
    # 0.019423 per call

    # but Dec 28 2017
    # ldpc_iters 20, success 0.64, 0.000592 sec/call
    # ldpc_iters 33, success 0.68, 0.000749 sec/call
    # ldpc_iters 37, success 0.68, 0.000806 sec/call
    # ldpc_iters 50, success 0.69, 0.000943 sec/call
    # ldpc_iters 100, success 0.71, 0.001515 sec/call
    # fast_tanh is a bit faster, but has same success as tanh()
    # ldpc_decode_python() has about the same success rate

    # nov 30 2018, old FT8
    # ldpc_iters 15, success 0.60, 0.000383 sec/call

    # dec 3 2018, new FT8
    # ldpc_iters 15, success 0.43, 0.000341 sec/call
    # ldpc_iters 15, success 0.41, 0.014654 sec/call


# codeword is 174 log-likelihoods.
# return is  [ ok, 83 bits ].
# ok is 83 if all ldpc parity checks worked, < 83 otherwise.
# result is usually garbage if ok < 83.
def ldpc_decode_c(codeword, ldpc_iters):
    double174 = ctypes.c_double * 174
    int174 = ctypes.c_int * 174

    c174 = double174()
    for i in range(0, 174):
        c174[i] = codeword[i]

    out174 = int174()
    for i in range(0, 174):
        out174[i] = -1;

    ok = ctypes.c_int()
    ok.value = -1

    libldpc.ldpc_decode(c174, ldpc_iters, out174, ctypes.byref(ok))

    plain174 = numpy.zeros(174, dtype=numpy.int32);
    for i in range(0, 174):
        plain174[i] = out174[i];

    plain91 = plain174[0:91]
    return [ ok.value, plain91 ]

# returns [ nok, plain ], where nok is the number of parity
# checks that worked out, should be 83=174-91.
def ldpc_decode(codeword, ldpc_iters):
    if libldpc != None:
        return ldpc_decode_c(codeword, ldpc_iters)
    else:
        return ldpc_decode_python(codeword, ldpc_iters)

if False:
    ldpc_test(1*17)
    ldpc_test(2*17)
    ldpc_test(4*17)
    ldpc_test(8*17)
    sys.exit(1)
    # nov 30 2018:
    # C:      ldpc_iters 15, success 0.59, 0.000383 sec/call
    # python: ldpc_iters 15, success 0.58, 0.013829 sec/call
    # dec 3 2018:
    # ldpc_iters 15, success 0.45, 0.000339 sec/call
    #   XXX why worse now?
    # apr 26 2019:
    # ldpc_iters 17, success 0.46, 0.000139 sec/call
    # ldpc_iters 34, success 0.51, 0.000183 sec/call
    # ldpc_iters 68, success 0.51, 0.000272 sec/call
    # ldpc_iters 136, success 0.52, 0.000436 sec/call

# gauss-jordan elimination of rows.
# m[row][col]
# inverts the square top of the matrix, swapping
# with lower rows as needed.
# returns the inverse of the top of the matrix.
# cooks up the identity matrix (the right-hand half)
# as needed.
# mod 2, so elements should be 0 or 1.
# keeps track of swaps in which[] -- every time it swaps two
# rows, it swaps the same two rows in which[].
# which[] could start out as range(0, n_rows).
def python_gauss_jordan(m, which):
    rows = m.shape[1] # rows to invert = columns
    # assert numpy.all(numpy.greater_equal(m, 0))
    # assert numpy.all(numpy.less_equal(m, 1))
    b = numpy.zeros((m.shape[0], rows * 2), dtype=m.dtype)
    b[:,0:rows] = m

    for row in range(0, rows):
        if b[row,row] != 1:
            # oops, find a row that has a 1 in row,row,
            # and swap.
            for row1 in range(row+1,m.shape[0]):
                if b[row1,row] == 1:
                    tmp = numpy.copy(b[row])
                    b[row] = b[row1]
                    b[row1] = tmp
                    tmp = which[row]
                    which[row] = which[row1]
                    which[row1] = tmp
                    break
        if b[row,row] != 1:
            sys.stderr.write("not reducible\n")
            print(b)
            return numpy.array([])
        # lazy creation of identity matrix in right half
        b[row,rows+row] = (b[row,rows+row] + 1) % 2 
        # now eliminate
        for row1 in range(0, m.shape[0]):
            if row1 == row:
                continue
            if b[row1,row] != 0:
                b[row1] = numpy.mod(b[row]+b[row1], 2)
    
    # assert numpy.array_equal(b[0:rows,0:rows],
    #                          numpy.eye(rows, dtype=numpy.int32))

    c = b[0:rows,rows:2*rows]
    return c

# m[174,91]
def gauss_jordan(m, which):
    if m.dtype != numpy.int32:
        m = m.astype(numpy.int32)

    rows = m.shape[1]
    int174 = ctypes.c_int * (m.shape[0])

    # m.shape = (174, 91)
    # len(which) = 174

    b = numpy.zeros((m.shape[0], rows*2), dtype=numpy.int32)
    b[:,0:rows] = m
    xb = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    xwhich = int174()
    for i in range(0, len(which)):
        xwhich[i] = which[i]

    ok = ctypes.c_int()
    ok.value = -1

    cols = m.shape[0]
    libldpc.gauss_jordan(rows, cols, xb, xwhich, ctypes.byref(ok))

    if ok.value != 1:
        sys.stderr.write("C gauss-jordan: not reducible %d\n" % (ok.value))
        return numpy.array([])

    for i in range(0, len(which)):
        which[i] = xwhich[i]

    c = b[0:rows,rows:2*rows]
    return c

def test_gauss_jordan():
    rows = 14
    cols = 8
    a = numpy.random.randint(0, 2, (rows, cols))

    which = range(0, a.shape[0])
    which = numpy.array(which, dtype=numpy.int32)
    pwhich = numpy.copy(which)

    pb = python_gauss_jordan(numpy.copy(a), pwhich)

    cb = gauss_jordan(numpy.copy(a), which)

    if not numpy.array_equal(cb, pb):
        print("not the same")
        sys.exit(1)

    if len(cb) > 0:
        aa = a[which][0:cols,:]
        prod = numpy.mod(numpy.dot(aa, cb), 2)
        assert numpy.array_equal(prod, numpy.eye(cols, dtype=numpy.int32))

if False:
    test_gauss_jordan()
    sys.exit(1)

# turn an array of bits into an integer.
# for unpacking FT8 messages.
# MSB, most-significant bit first.
# works with bignums.
def un(a):
    if sys.version_info.major >= 3:
        x = 0
        for i, b in enumerate(reversed(a)):
            # b is numpy.int32, which is not
            # a bignum, so must convert to int().
            x = x + (int(b) << i)
    else:
        x = long(0)
        for i, b in enumerate(reversed(a)):
            x = x + (long(b) << i)
    return x

# turn an integer into an array of bits.
# for packing FT8 messages.
# MSB.
# works with bignums.
def bv(x, n):
    a = numpy.zeros(n, dtype=numpy.int32)
    for i in range(0, n):
        a[i] = (x >> (n - i - 1)) & 1
    return a

if False:
    # check that un() and bv() can handle numbers
    # larger than 2^32, by using Python's
    # bignum ("long") feature.
    bbb = numpy.ones(71, dtype=numpy.int32)
    bbb[7] = 0
    bbb[55] = 0
    x = un(bbb)
    ccc = bv(x, len(bbb))
    print(bbb)
    print(ccc)
    assert numpy.array_equal(bbb, ccc)
    sys.exit(0)

# does this OSD decode look plausible?
def osd_check(plain):
    if numpy.all(plain==0) == True:
        # all zeros
        return False
    if un(plain[74:74+3]) != 1:
        # i3 != 1
        return False
    if osd_crc and check_crc(plain) == False:
        return False
    return True

# xplain is 91 bits.
# codeword is the received 174 log-likelihoods.
# return a score, smaller is better.
def osd_score(xplain, codeword):
    two = numpy.array([ 4.6, -4.6 ]) # for hard->soft conversion
    xcode = ldpc_encode(xplain)
    xsoftcode = two[xcode]
    xscore = 0.0 - numpy.sum(xsoftcode * codeword)
    return xscore

def junkdec(d91):
    crcok = check_crc(d91)
    r = FT8()
    msg = None
    dec = r.unpack(d91)
    if dec != None:
        msg = dec.msg
    x = "%s" % (msg)
    if crcok:
        x += " YYY"
    return x

# ordered statistics decoder for LDPC and new FT8.
# idea from wsjt-x.
# codeword[i] = log ( P(x=0) / P(x=1) )
# codeword has 174 bits.
# returns [ 91 bits, score ]
def osd_decode(codeword, depth):

    # first 91 bits are plaintext, remaining 83 are parity.

    if True:
        # force i3=1, with high strength.
        maxll = numpy.max(abs(codeword))
        codeword = numpy.copy(codeword)
        codeword[74] = maxll  # 0
        codeword[75] = maxll  # 0
        codeword[76] = -maxll # 1

    # we're going to use the strongest 91 bits of codeword.
    strength = abs(codeword)
    which = numpy.argsort(-strength)

    # gen_sys[174 rows][91 cols] has a row per each of the 174 codeword bits,
    # indicating how to generate it by xor with each of the 91 plain bits.

    # generator matrix, reordered strongest codeword bit first.
    gen1 = gen_sys[which]

    # gen1[row,col]
    # gen1[i,:] produced y[i]

    gen1_inv = gauss_jordan(gen1, which)

    # which[i] = j means y1[i] is a174[j]
    # and gen1[i] is gen_sys[j]
    # gauss_jordan usually changes which[].

    # y1 is the received bits, same order as gen1_inv,
    # more or less strongest-first.
    #y1 = numpy.copy(codeword)
    #y1 = y1[which]
    #y1 = y1[0:91]
    y1 = codeword[which[0:91]]
    y1 = soft2hard(y1)

    # we expect this to yield the original plain-text.
    # z = gen1_inv * y
    # z = gen1_inv * (gen1 * plain)
    # z = plain
    # junk = numpy.matmul(gen1_inv, y1)
    # junk = numpy.mod(junk, 2)
    # assert numpy.array_equal(junk, plain)

    # next: 
    # code to reconstruct the whole codeword
    # flip one or more bits in y1
    # see which version of y1 reconstructs to codeword closest to received

    best_plain = numpy.array([])
    best_score = None # lower is better

    # don't flip any bits

    xplain = numpy.mod(numpy.matmul(gen1_inv, y1), 2)

    xscore = osd_score(xplain, codeword)
    if xscore < osd_thresh and osd_check(xplain):
        if osd0_crc and check_crc(xplain):
            return [ xplain, 0 ]
        # print("depth=X score=%d %s" % (xscore, junkdec(xplain)))
        best_plain = xplain
        best_score = xscore # lower is better

    # flip a few single bits, weakest first.
    # this is not entirely justified since it's probably
    # entire symbols that are bad. on the other hand,
    # the log-likelyhood and gray code machinery will
    # probably give the low bit of each symbol the
    # lowest strength.

    for ii in range(0, depth):
        i = len(y1) - 1 - ii
        y1[i] = (y1[i] + 1) % 2
        xplain = numpy.mod(numpy.matmul(gen1_inv, y1), 2)
        y1[i] = (y1[i] + 1) % 2
        xscore = osd_score(xplain, codeword)
        if xscore < osd_thresh and osd_check(xplain) and (best_score == None or xscore < best_score):
            # print("ii=%d score %s->%d %s" % (ii, best_score, xscore, junkdec(xplain)))
            best_plain = xplain
            best_score = xscore
            
    return [ best_plain, best_score ]


def osd_test_gen(nbad):
    a91 = numpy.random.randint(0, 2, 91, dtype=numpy.int32)
    a91[74:74+3] = bv(1, 3) # i3 == 1, the most common case, for osd_check()
    cksum = crc(numpy.append(a91[0:77], numpy.zeros(5, dtype=numpy.int32)),
                crc14poly)
    a91[-14:] = cksum
    a174 = ldpc_encode(a91)

    # turn hard bits into 0.99 vs 0.01 log-likelihood,
    # log( P(0) / P(1) )
    # log base e.
    ll174 = 1.0 - (2.0 * a174)
    ll174 *= 4.5
    #ll174 *= numpy.random.random(len(ll174))

    good174 = numpy.copy(ll174)

    # ll174[i] is -4.5 or 4.5, for bit value of 1 or 0

    # disturb nbad distinct bits
    bb = numpy.array(range(0, 174), dtype=numpy.int32)
    numpy.random.shuffle(bb)
    for i in range(0, nbad):
        if True:
            # flip the bit
            ll174[bb[i]] *= -1
            ll174[bb[i]] *= 0.5
            ll174[bb[i]] *= random.random()
        else:
            # randomize the bit
            ll174[bb[i]] = random.random() - 0.5
            ll174[bb[i]] /= 5.0

    return ( a91, ll174, good174 )

def osd_test_1(nbad, depth):
    global ldpc_iters

    iters = 500

    plains = [ ]
    codewords = [ ] # 174 log-likelihoods with nbad errors
    goodwords = [ ] # 174 log-likelihoods with no errors
    for iter in range(0, iters):
        ( p, c, g )  = osd_test_gen(nbad)
        plains.append(p)
        codewords.append(c)
        goodwords.append(g)

    if True:
        # check that osd_test_gen() flipped exactly nbad bits.
        for iter in range(0, iters):
            xbad = 0
            xgood = 0
            for i in range(0, 174):
                if ((goodwords[iter][i] < 0 and codewords[iter][i] < 0) or
                    (goodwords[iter][i] > 0 and codewords[iter][i] > 0)):
                    xgood += 1
                    assert abs(codewords[iter][i]) > 4.0
                else:
                    assert abs(codewords[iter][i]) < 4.0
                    xbad += 1
            # print("xbad=%d xgood=%d" % (xbad, xgood))
            assert xbad == nbad and (xbad + xgood) == 174

    if False:
        # for comparison
        # for ldpc_iters in [ oli // 2, oli, oli * 2, oli * 4 ]:
        for xldpc_iters in [ 15 ]:
            nok = 0
            t0 = time.time()
            for iter in range(0, iters):
                [ _, d91 ] = ldpc_decode(numpy.copy(codewords[iter]), xldpc_iters)
                ok = numpy.array_equal(plains[iter], d91)
                if ok:
                    nok += 1
            t1 = time.time()
            print("ldpc_decode iters=%d ok=%.2f, %.3f sec" % (xldpc_iters,
                                                              nok / float(iters),
                                                              (t1-t0)/iters))

    nok = 0
    t0 = time.time()
    for iter in range(0, iters):
        [ d91, sc ] = osd_decode(numpy.copy(codewords[iter]), depth)
        ok = numpy.array_equal(plains[iter], d91)
        if ok:
            nok += 1
    t1 = time.time()
    print("n=%d depth=%d ok=%.02f, %.3f sec" % (nbad,
                                                depth,
                                                nok/float(iters),
                                                (t1-t0)/iters))

def osd_test():
    for nbad in range(78, 84):
        osd_test_1(nbad, 0)
        osd_test_1(nbad, 1)

if False:
    osd_test()
    sys.exit(1)

if False:
    profiling = True
    pfile = "cprof.out"
    sys.stderr.write("ft8: cProfile -> %s\n" % (pfile))
    import cProfile
    import pstats
    cProfile.run('osd_test()', pfile)
    p = pstats.Stats(pfile)
    p.strip_dirs().sort_stats('time')
    # p.print_stats(10)
    p.print_callers()
    sys.exit(1)

# a-priori probability of each of the 174 LDPC codeword
# bits being one. measured from reconstructed correct
# codewords, into ft8bits, then python bprob.py.
# from ft8-n4
apriori174 = numpy.array([
0.47, 0.32, 0.29, 0.37, 0.52, 0.36, 0.40, 0.42, 0.42, 0.53, 0.44,
0.44, 0.39, 0.46, 0.39, 0.38, 0.42, 0.43, 0.45, 0.51, 0.42, 0.48,
0.31, 0.45, 0.47, 0.53, 0.59, 0.41, 0.03, 0.50, 0.30, 0.26, 0.40,
0.65, 0.34, 0.49, 0.46, 0.49, 0.69, 0.40, 0.45, 0.45, 0.60, 0.46,
0.43, 0.49, 0.56, 0.45, 0.55, 0.51, 0.46, 0.37, 0.55, 0.52, 0.56,
0.55, 0.50, 0.01, 0.19, 0.70, 0.88, 0.75, 0.75, 0.74, 0.73, 0.18,
0.71, 0.35, 0.60, 0.58, 0.36, 0.60, 0.38, 0.50, 0.02, 0.01, 0.98,
0.48, 0.49, 0.54, 0.50, 0.49, 0.53, 0.50, 0.49, 0.49, 0.51, 0.51,
0.51, 0.47, 0.50, 0.53, 0.51, 0.46, 0.51, 0.51, 0.48, 0.51, 0.52,
0.50, 0.52, 0.51, 0.50, 0.49, 0.53, 0.52, 0.50, 0.46, 0.47, 0.48,
0.52, 0.50, 0.49, 0.51, 0.49, 0.49, 0.50, 0.50, 0.50, 0.50, 0.51,
0.50, 0.49, 0.49, 0.55, 0.49, 0.51, 0.48, 0.55, 0.49, 0.48, 0.50,
0.51, 0.50, 0.51, 0.50, 0.51, 0.53, 0.49, 0.54, 0.50, 0.48, 0.49,
0.46, 0.51, 0.51, 0.52, 0.49, 0.51, 0.49, 0.51, 0.50, 0.49, 0.50,
0.50, 0.47, 0.49, 0.52, 0.49, 0.51, 0.49, 0.48, 0.52, 0.48, 0.49,
0.47, 0.50, 0.48, 0.50, 0.49, 0.51, 0.51, 0.51, 0.49,
])
null_apriori174 = numpy.repeat(0.5, 174)

# take 174 bits just after LDPC decode, un-gray-code,
# and return resulting 174 bits.
def un_gray_code(a174):
    revmap = numpy.zeros(8, dtype=numpy.int32)
    for i in range(0, 8):
        revmap[graymap[i]] = i
    return gray_common(a174, revmap)

def gray_code(b174):
    mm = numpy.array(graymap, dtype=numpy.int32)
    return gray_common(b174, mm)

def gray_common(b174, mm):
    # create 58 3-bit numbers
    x = 4*b174[0::3] + 2*b174[1::3] + 1*b174[2::3]
    y = mm[x]
    a174 = numpy.zeros(len(b174), dtype=numpy.int32)
    a174[0::3] = numpy.bitwise_and(numpy.right_shift(y, 2), 1)
    a174[1::3] = numpy.bitwise_and(numpy.right_shift(y, 1), 1)
    a174[2::3] = numpy.bitwise_and(numpy.right_shift(y, 0), 1)
    return a174

if False:
    print("testing gray_code")
    b174 = numpy.random.randint(0, 2, 174)
    a174 = gray_code(b174)
    bb174 = un_gray_code(a174)
    assert not numpy.array_equal(a174, b174)
    assert numpy.array_equal(b174, bb174)
    sys.exit(0)

# hash a string call into m bits (10, 12, or 22),
# for FT8 message types that include a hash of
# a previously-seen call sign that doesn't fit
# in the ordinary 28 bit encoding.
# e.g. ihashcall("SX60RAAG",22) == 3214310
# copied from packjt77.f90
def ihashcall(call, m):
    chars = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"

    while len(call) < 11:
        call = call + " "

    if sys.version_info.major >= 3:
        x = 0
        for c in call[0:11]:
            j = chars.find(c)
            x = 38*x + j
            x = x & ((int(1) << 64) - 1)
        x = x & ((1 << 64) - 1)
        x = x * 47055833459
        x = x & ((1 << 64) - 1)
        x = x >> (64 - m)
    else:
        x = long(0)
        for c in call[0:11]:
            j = chars.find(c)
            x = 38*x + j
            x = x & ((long(1) << 64) - 1)
        x = x & ((long(1) << 64) - 1)
        x = x * long(47055833459)
        x = x & ((long(1) << 64) - 1)
        x = x >> (64 - m)
    return x

# ARRL RTTY Round-Up states/provinces
ru_states = [
       "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
       "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
       "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
       "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
       "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
       "NB","NS","QC","ON","MB","SK","AB","BC","NWT","NF",
       "LB","NU","YT","PEI","DC" ]


# information about one decoded signal.
class Decode:
    def __init__(self, msg):
        self.hza = None
        self.msg = msg
        self.snr = None
        self.bits77 = None
        self.decode_time = None # unix time of decode
        self.minute = None # cycle number
        self.start = None # sample number
        self.dt = None # dt in seconds
        self.hint = None
        self.hashcalls = [ ] # calls (as strings) to add to hashes22 and hashes12 and hashes10
        self.m58 = numpy.zeros([0,8])

    def hz(self):
        return numpy.mean(self.hza)

class Hint:
    def __init__(self, call1=None, call2=None, hz=None):
        # call1 and call2 are string call-signs, e.g. "JA2XX".
        self.i3 = 1
        self.n3 = None
        self.call1 = call1 # could be "CQ"
        self.packed1 = numpy.array([])
        self.call2 = call2
        self.packed2 = numpy.array([])
        self.hz = hz # only look near this Hz

    def __str__(self):
        if self.hz == None:
            hz = "-"
        else:
            hz = "%.0f" % (self.hz)
        return "%s.%s.%s" % (self.call1, self.call2, hz)

normal_table_stds = 10 # +/- this many std dev
normal_table_gran = 40 # this many points per std dev
normal_table = None
normal_table_log = None

# Normal function integrated from -Inf to x. Range: 0-1.
# x in units of std dev.
# mean is zero.
# the same as scipy.stats.norm.cdf([x])
def real_normal(x):
    y = 0.5 + 0.5*math.erf(x / 1.414213)
    return y

def real_normal_log(x):
    y = scipy.stats.norm.logcdf(x)
    return y

def make_normal_table():
    global normal_table, normal_table_log
    tt = [ ]
    tt_log = [ ]
    x = 0 - normal_table_stds
    while x < normal_table_stds:
        tt.append(real_normal(x))
        tt_log.append(real_normal_log(x))
        x += 1.0 / normal_table_gran
    normal_table = numpy.array(tt)
    normal_table_log = numpy.array(tt_log)

# x is distance from mean, in units of std-dev.
# uses pre-computed table.
def vnormal(x):
    x *= normal_table_gran
    x += (normal_table_stds * normal_table_gran)
    x = numpy.rint(x)
    x = x.astype(numpy.int32)
    x = numpy.maximum(x, 0)
    x = numpy.minimum(x, len(normal_table)-1)
    return normal_table[x]

def vnormal_log(x):
    x *= normal_table_gran
    x += (normal_table_stds * normal_table_gran)
    x = numpy.rint(x)
    x = x.astype(numpy.int32)
    x = numpy.maximum(x, 0)
    x = numpy.minimum(x, len(normal_table_log)-1)
    return normal_table_log[x]

# vectorized normal()
def real_vnormal(v):
    return scipy.stats.norm.cdf(v)

# how much of the distribution is < x?
def problt(x, mean, std):
    if std != 0.0:
        y = normal((x - mean) / std)
    else:
        y = 0.5
    return y

def vproblt(x, mean, std):
    y = vnormal((x - mean) / std)
    return y

def vproblt_log(x, mean, std):
    y = vnormal_log((x - mean) / std)
    return y

# how much of the distribution is > x?
def probgt(x, mean, std):
    if std != 0.0:
        y = 1.0 - normal((x - mean) / std)
    else:
        y = 0.5
    return y

def vprobgt(x, mean, std):
    y = 1.0 - vnormal((x - mean) / std)
    return y

def bit_reverse(x, width):
    y = 0
    for i in range(0, width):
        z = (x >> i) & 1
        y <<= 1
        y |= z
    return y

# turn an array of bits into a number.
# most significant bit first.
def bits2num(bits):
    assert len(bits) < 32
    n = 0
    for i in range(0, len(bits)):
        n *= 2
        n += bits[i]
    return n

# return the best FFT bin number for a frequency.
# assumes 6.25-Hz bins, for FT8.
# the critical thing is that each bin is *centered* at
# a multiple of 6.25 Hz (it does not *start* at a multiple).
def bin_of(hz):
    bin = int((hz + 3.125) / 6.25)
    return bin

# gadget that returns FFT buckets of a fixed set of
# original samples, with required (inter-bucket)
# frequency, drift, and offset.
class FFTCache:
    def __init__(self, samples, jrate, jblock, tstep, fstep):
        self.jrate = jrate
        self.jblock = jblock
        self.samples = samples
        self.tstep = tstep # was coarse_tstep
        self.fstep = fstep # was coarse_fstep

        fg = self.fstep
        if down200 == False and fine_fstep > 0:
            fg *= fine_fstep
        self.bin_granules = int(round(fg))

        bg = self.tstep
        if down200 == False and fine_tstep > 0:
            bg *= fine_tstep
        self.block_granules = int(round(bg))

        # compute self.all[][]
        self.makeall()

    # do all the FFTs, for all granules.
    def makeall(self):
        bin_hz = self.jrate / float(self.jblock)

        # all[79*granules][fftsize*granules]
        # all[block*granules+blockoff][bin*granules+binoff]
        nblocks = len(self.samples) // self.jblock
        nxall = self.block_granules * nblocks
        nbins = (self.jblock // 2) + 1
        nyall = self.bin_granules * nbins
        self.all = numpy.zeros((nxall, nyall), dtype=numpy.complex128)

        # block granule size, in samples.
        gsize = self.jblock / float(self.block_granules)

        hilb = weakutil.pre_freq_shift(self.samples)

        for i in range(0, self.bin_granules):
            ss = self.samples
            if i != 0:
                freq_off = i * (bin_hz / self.bin_granules)
                ss = weakutil.freq_shift(ss, -freq_off, 1.0/self.jrate, hilb)
            for gi in range(0, self.block_granules):
                if False:
                    for bi in range(0, nblocks):
                        off = (bi * self.jblock) + int(round(gi * gsize))
                        if off + self.jblock > len(ss):
                            break
                        block = ss[off:off+self.jblock]
                        a = weakutil.rfft(block)
                        alli = bi * self.block_granules + gi
                        self.all[alli][i::self.bin_granules] = a
                else:
                    # use rfftn()
                    i0 = int(round(gi * gsize))
                    nb = len(ss[i0:]) // self.jblock
                    bb = numpy.reshape(ss[i0:i0+nb*self.jblock], (nb, self.jblock))
                    assert nb >= 79
                    mm = numpy.zeros((nb, nbins), dtype=numpy.complex128)
                    # mm[0:79,:] = numpy.fft.rfftn(bb[0:79,:], axes=[1])
                    bbb = numpy.copy(bb[0:79,:]) # fftw rfftn needs this
                    mm[0:79,:] = weakutil.rfftn(bbb, axes=[1])
                    for k in range(79, nb):
                        mm[k,:] = weakutil.rfft(bb[k])
                    self.all[gi:gi+nb*self.block_granules:self.block_granules,i::self.bin_granules] = mm

    # return bins[symbol][bin] -- i.e. a mini-FFT per symbol.
    def get_complex(self, hz, start):
        [ hz0, offset0, m ] = self.getall_complex(hz, start)
        bin = bin_of(hz - hz0)
        subm = m[start // self.jblock : , bin : bin+8]
        return [ hz0, offset0, subm ]

    def get(self, hz, start):
        [ hz0, offset0, m1 ] = self.get_complex(hz, start)
        m1 = abs(m1)
        return [ hz0, offset0, m1 ]

    # not used yet, but does hz drift.
    def new_get(self, hza, start):
        bin_hz = self.jrate / float(self.jblock)

        nx = len(self.samples) // self.jblock
        m = numpy.zeros((nx, 8))

        for x in range(0, nx):
            offset = start + x*self.jblock
            xi = offset // (self.jblock // self.block_granules)
            hz = hza[0] + (hza[1] - hza[0]) * (x / float(nx))
            hzi = int(round(hz / (bin_hz / self.bin_granules)))
            if xi < self.all.shape[0]:
                m[x] = self.all[xi][hzi:hzi+8*self.bin_granules:self.bin_granules]

        return m

    # return the complete set of FFTs, m[symbol][bin]
    # hz and start just cause a sub-bin and sub-symbol shift.
    # that is, the returned array starts near hz=0
    # and offset=0.
    # returns complex FFT results.
    # returns [ hz0, offset0, ffts ]
    def getall_complex(self, hz, start):
        bin_hz = self.jrate / float(self.jblock)
        bingran = bin_hz / self.bin_granules

        # which quarter-bin?
        #bin = int(hz / bin_hz)
        #binfrac = (hz / bin_hz) - bin
        #binkey = int(binfrac / (1.0 / self.bin_granules))

        bgi = int((hz + (bingran/2)) / bingran)
        binkey = bgi % self.bin_granules

        # which eighth-block?
        blockoff = start % self.jblock
        gsize = self.jblock / float(self.block_granules)
        blockkey = int(round(blockoff / gsize))

        # binkey is 0..self.bin_granules
        # blockkey is 0..self.block_granules
        m = self.all[blockkey::self.block_granules,binkey::self.bin_granules]

        return [ binkey*bingran,
                 blockkey*(self.jblock // self.block_granules),
                 m ]

    def getall(self, hz, start):
        [ hz0, offset0, m ] = self.getall_complex(hz, start)
        m = abs(m)
        return [ hz0, offset0, m ]

    def len(self):
        return len(self.samples)

# offsets for 28-bit call sign encoding.
NTOKENS = 2063592
MAX22 = 4194304

# start of special grid locators for sig strength &c.
NGBASE = 180*180

class FT8:
    debug = False

    offset = 0

    def __init__(self):
        self.msgs_lock = threading.Lock()
        self.msgs = [ ]
        self.verbose = False
        self.enabled = True
        self.extra_budget = 0
        self.band = "-"
        self.carddesc = None
        self.restrict_hz = None
        self.hints = [ Hint("CQ") ]
        self.hashes22 = { } # non-standard calls indexed by 22-bit hash
        self.hashes12 = { } # non-standard calls indexed by 12-bit hash
        self.hashes10 = { } # non-standard calls indexed by 10-bit hash
        self.forked = False
        self.ldpc_calls = 0

        #self.jrate = 12000 // 2 # sample rate for processing (FFT &c)
        #self.jblock = 1920 // 2 # samples per symbol

        weakutil.init_freq_from_fft(32)
        weakutil.init_freq_from_fft(1920)
        weakutil.init_freq_from_fft(1920 // 2)
        weakutil.init_freq_from_fft(1920 // 4)
        weakutil.init_freq_from_fft(1920 // 6)

        weakutil.fft_sizes([1920, 1920 // 2, 1920 // 4, 1920 // 6, 32])
        weakutil.init_fft()

        # set self.start_time to the UNIX time of the start
        # of the last UTC minute.
        now = int(time.time())
        gm = time.gmtime(now)
        self.start_time = now - gm.tm_sec

        make_normal_table()

    def close(self):
        pass

    def junklog(self, samples_time, msg):
        minute = self.minute(samples_time + 1)
        gm = time.gmtime(self.minute2time(minute))
        hms = "%02d:%02d:%02d" % (
            gm.tm_hour,
            gm.tm_min,
            gm.tm_sec)
        msg = "%s %s %s %s\n" % (self.ts(time.time()),
                                 hms,
                                 self.carddesc,
                                 msg)

        #sys.stderr.write(msg)
        f = open("ft8-junk.txt", "a")
        f.write(msg)
        f.close()

    # seconds per cycle
    def cycle_seconds(self):
        return 15

    # return the minute number for t, a UNIX time in seconds.
    # truncates down, so best to pass a time mid-way through a minute.
    # returns only even minutes.
    def minute(self, t):
        dt = t - self.start_time
        return int(dt / 15.0)

    # convert cycle number to UNIX time.
    def minute2time(self, m):
        return (m * 15) + self.start_time

    # seconds since minute(), 0..15
    def second(self, t):
        dt = t - self.start_time
        dt /= 15.0
        return 15.0 * (dt - int(dt))

    def seconds_left(self, t):
        return 15 - self.second(t)

    # printable UTC timestamp, e.g. "07/07/15 16:31:00"
    # dd/mm/yy hh:mm:ss
    # t is unix time.
    def ts(self, t):
        gm = time.gmtime(t)
        return "%02d/%02d/%02d %02d:%02d:%02d" % (gm.tm_mday,
                                                  gm.tm_mon,
                                                  gm.tm_year - 2000,
                                                  gm.tm_hour,
                                                  gm.tm_min,
                                                  gm.tm_sec)

    # UNIX time to HHMMSS
    def hhmmss(self, t):
        gm = time.gmtime(t)
        return "%02d%02d%02d" % (gm.tm_hour, gm.tm_min, gm.tm_sec)

    def openwav(self, filename):
        self.wav = wave.open(filename)
        self.wav_channels = self.wav.getnchannels()
        self.wav_width = self.wav.getsampwidth()
        self.cardrate = self.wav.getframerate()

    def readwav(self, chan):
        z = self.wav.readframes(8192)
        if self.wav_width == 1:
            zz = numpy.fromstring(z, numpy.int8)
        elif self.wav_width == 2:
            if (len(z) % 2) == 1:
                return numpy.array([])
            zz = numpy.fromstring(z, numpy.int16)
        else:
            sys.stderr.write("oops wave_width %d" % (self.wav_width))
            sys.exit(1)
        if self.wav_channels == 1:
            return zz
        elif self.wav_channels == 2:
            return zz[chan::2] # chan 0/1 => left/right
        else:
            sys.stderr.write("oops wav_channels %d" % (self.wav_channels))
            sys.exit(1)

    def gowav(self, filename, chan):
        self.openwav(filename)
        bufbuf = [ ]
        while True:
            buf = self.readwav(chan)
            if buf.size < 1:
                break
            bufbuf.append(buf)
        samples = numpy.concatenate(bufbuf)

        # trim trailing zeroes that wsjt-x adds to .wav files.
        i = len(samples)
        while i > 1000 and numpy.max(samples[i-1:]) == 0.0:
            if numpy.max(samples[i-1000:]) == 0.0:
                i -= 1000
            elif numpy.max(samples[i-100:]) == 0.0:
                i -= 100
            elif numpy.max(samples[i-10:]) == 0.0:
                i -= 10
            else:
                i -= 1
        samples = samples[0:i]

        self.process(samples, 0)

    def opencard(self, desc):
        self.carddesc = desc
        self.cardrate = 6000 # 12000 // 2
        self.audio = weakaudio.new(desc, self.cardrate)

    def gocard(self):
        bufbuf = [ ]
        nsamples = 0
        prev_buf_time = None
        last_tmin = None
        self.audio.read() # get the SDR-IP started
        while True:
            sec0 = self.second(time.time())
            howlong = 13.0 - sec0
            if howlong > 0:
                # sleep for a long time if we're not near the deadline.
                howlong = max(howlong, 0.010)
                time.sleep(howlong)

            [ buf, buf_time ] = self.audio.raw_read()

            # buf_time is the UNIX time of the last sample in buf[].

            if len(buf) > 0:
                # is there a gap in the sample stream?
                if prev_buf_time != None:
                    dt = buf_time - prev_buf_time
                    expected = self.audio.rawrate * dt
                    got = len(buf)
                    if abs(expected-got) > 2:
                        self.junklog(buf_time,
                          "gocard expected %s got %s, prev_buf_time %s buf_time %s" % (expected,
                                                                                       got,
                                                                                       prev_buf_time,
                                                                                       buf_time))

                # doesn't work for raw_read().
                #mx = numpy.max(numpy.abs(buf))
                #if mx > 30000:
                #    sys.stderr.write("!")

                bufbuf.append(buf)
                nsamples += len(buf)
                prev_buf_time = buf_time
            else:
                # self.audio.read() is non-blocking, so sleep a bit.
                time.sleep(0.050)

            # an FT8 frame starts on second 0.5, and takes 12.64 seconds.
            if nsamples >= 13.14*self.audio.rawrate:
                sec = self.second(buf_time)
                if sec >= 13.14:
                    # we have >= 13.14 seconds of samples,
                    # and second of minute is >= 13.14.

                    samples = numpy.concatenate(bufbuf)

                    # device-specific I/Q to SSB, rate conversion
                    # from self.audio.rawrate to self.cardrate.
                    # do this for the full 15 seconds, which
                    # improves the conversion quality.
                    samples = self.audio.postprocess(samples)

                    excess = len(samples) - 15*self.cardrate
                    if excess < -1000 or excess > 1000:
                        self.junklog(buf_time, "gocard excess %d" % (excess))

                    # sample # of start of 15-second interval.
                    i0 = len(samples) - self.cardrate * self.second(buf_time)
                    i0 = int(i0)
                    i0 = max(i0, 0)

                    # UNIX time of samples[i0]
                    samples_time = buf_time - (len(samples)-i0) * (1.0/self.cardrate)

                    tmin = self.minute(samples_time + 1)
                    if last_tmin != None and tmin != last_tmin + 1:
                        self.junklog(samples_time, "gocard jumped minute %d %d" % (last_tmin, tmin))
                    last_tmin = tmin

                    self.process(samples[i0:], samples_time)

                    bufbuf = [ ]
                    nsamples = 0

    # received a message, add it to the list.
    # offset in seconds.
    # drift in hz/minute.
    def got_msg(self, dec):
        self.msgs_lock.acquire()

        # already in msgs with worse nerrs?
        found = False
        for i in range(max(0, len(self.msgs)-40), len(self.msgs)):
            xm = self.msgs[i]
            if xm.minute == dec.minute and abs(xm.hz() - dec.hz()) < 10 and xm.msg == dec.msg:
                # we already have this msg
                found = True
                if dec.snr > xm.snr:
                    self.msgs[i] = dec

        if found == False:
            self.msgs.append(dec)

        self.msgs_lock.release()

    # someone wants a list of all messages received,
    # as array of Decode.
    def get_msgs(self):
        self.msgs_lock.acquire()
        a = self.msgs
        self.msgs = [ ]
        self.msgs_lock.release()
        return a

    # c is a pipe from a sub-process.
    def readchild(self, c, samples_time):
        start_time = time.time()
        n = 0
        while True:
            dec = None
            try:
                # poll so we won't wait forever if the sub-process
                # is wedged.
                howlong = budget + self.extra_budget + 1
                some = c.poll(howlong)
                if some == False:
                    break
                dec = c.recv()
            except:
                break

            self.got_msg(dec)
            n += 1

        dt = time.time() - start_time
        if dt > budget + self.extra_budget + 0.5:
            self.junklog(samples_time, "sub-process did not quit %.1f got %d" % (dt, n))

        c.close()

    # run the FT8 decode in a separate process. this yields
    # much more parallelism for multiple receivers than
    # Python's threads.
    # samples_time is UNIX time that samples[0] was
    # sampled by the sound card.
    def process(self, samples, samples_time):
        min_hz = real_min_hz
        max_hz = real_max_hz

        if self.restrict_hz != None:
            if self.restrict_hz[0] > min_hz:
                min_hz = self.restrict_hz[0]
            if self.restrict_hz[1] < max_hz:
                max_hz = self.restrict_hz[1]

        global very_first_time
        do_fork = not (profiling or very_first_time)
        self.forked = do_fork
        very_first_time = False

        save_extra_budget = self.extra_budget
        if do_fork == False:
            # avoid taking more than 15 seconds.
            self.extra_budget = 0

        ss = self.second(time.time())
        if self.band != "-" and self.enabled and (ss > 13.8 or ss < 13):
            self.junklog(samples_time, "late start %.1f" % (self.second(time.time())))
        sys.stdout.flush()

        procs = [ ]
        readers = [ ]

        for chi in range(0, nchildren):
            # adjust min_hz and max_hz
            hzinc = (max_hz - min_hz) / nchildren
            hz0 = min_hz + chi*hzinc
            hz1 = hz0 + hzinc
            if chi > 0:
                hz0 -= child_overlap
            if chi < nchildren-1:
                hz1 += child_overlap

            rpipe, spipe = multiprocessing.Pipe(False)

            th = threading.Thread(target=lambda c=rpipe: self.readchild(c, samples_time))
            th.start()
            readers.append(th)

            if do_fork:
                px = multiprocessing.Process(target=self.process00,
                                             args=[samples, samples_time, spipe, rpipe,
                                                   hz0, hz1])
                px.start()
                procs.append(px)
                spipe.close()
            else:
                # either profiling, or warming caches for the very first cycle.
                self.process00(samples, samples_time, spipe, None, hz0, hz1)
                # process00() closes spipe when it's done.

        oband = self.band

        # wait for the readchild threads.
        for chi in range(0, len(readers)):
            readers[chi].join(budget+self.extra_budget+1.0)
            if do_fork:
                procs[chi].terminate()
                procs[chi].join(0.5)

        if os.path.isfile("./savewave"):
            filename = "save/%s-%s.wav" % (self.hhmmss(samples_time), oband)
            weakutil.writewav(samples, filename, self.cardrate)

        if self.band != "-" and self.enabled and self.second(time.time()) > 13:
            self.junklog(samples_time, "late end %.1f" % (self.second(time.time())))

        self.extra_budget = save_extra_budget

    def process00(self, samples, samples_time, spipe, rpipe, min_hz, max_hz):
        if rpipe != None:
            rpipe.close()
        thunk = (lambda dec : spipe.send(dec))
        self.process0(samples, samples_time, thunk, min_hz, max_hz)
        spipe.close()

    def process0(self, samples, samples_time, thunk, min_hz, max_hz):

        if self.enabled == False:
            return

        # samples_time is UNIX time that samples[0] was
        # sampled by the sound card.
        samples_minute = self.minute(samples_time + 1)

        t0 = time.time()

        # pre-pack the hints to avoid repeated CPU time later.
        # generates 28-entry arrays of -1.0 / 1.0, for 1 and 0.
        snd = FT8Send()
        two = numpy.array([1.0, -1.0])
        for hint in self.hints:
            assert hint.i3 == 1
            if hint.call1 != None and not ("/" in hint.call1):
                xx = snd.packcall(hint.call1) # an integer, with 28 interesting bits
                if xx != -1:
                    hint.packed1 = two[bv(xx, 28)]
            if hint.call2 != None and (not "/" in hint.call2):
                xx = snd.packcall(hint.call2)
                if xx != -1:
                    hint.packed2 = two[bv(xx, 28)]

        down_hz = 0

        if self.cardrate == 12000:
            cardblock = 1920
        elif self.cardrate == 6000:
            cardblock = 1920 // 2
        else:
            assert False

        # use smallest sample rate that divides 12000 evenly
        # and can represent the Hz range.
        rates = [ 2000, 3000, 6000, 12000 ]
        for jrate in rates:
            if (self.cardrate % jrate) == 0 and jrate > 2 * (max_hz - min_hz + 50):
                break

        # if down-shifting, center in available b/w.
        basement_hz = (jrate/2 - ((max_hz+50)-min_hz)) / 2
        assert basement_hz > 0

        if bottom_slow and max_hz+50 < jrate/2 and (self.cardrate % jrate) == 0:
            down_factor = self.cardrate // jrate
            self.jrate = jrate
            self.jblock = cardblock // down_factor
            samples = scipy.signal.decimate(samples, down_factor,
                                            n=decimate_order, zero_phase=True)
        elif top_down and (max_hz+50+basement_hz) - min_hz < jrate/2 and (self.cardrate % jrate) == 0:
            # high-pass filter
            if min_hz < 210:
                # special case for very low cutoffs.
                # filter = weakutil.cheby_highpass(min_hz-20, self.cardrate)
                filter = weakutil.new_cheby_highpass(min_hz - 100,
                                                     min_hz + 0,
                                                     self.cardrate,
                                                     ripple_pass=cheb_ripple_pass,
                                                     atten_stop=cheb_atten_stop)
                
                # filter = weakutil.butter_highpass(min_hz, self.cardrate,
                #                                   order=11)
            elif top_high_order > 0:
                filter = weakutil.butter_highpass(min_hz * high_cutoff, self.cardrate,
                                                  order=top_high_order)
            else:
                # cut = max(min_hz - 12, 0)
                # filter = weakutil.cheby_highpass(cut, self.cardrate)
                filter = weakutil.new_cheby_highpass(min_hz - cheb_high_minus,
                                                     min_hz + cheb_high_plus,
                                                     self.cardrate,
                                                     ripple_pass=cheb_ripple_pass,
                                                     atten_stop=cheb_atten_stop)

            samples = scipy.signal.filtfilt(filter[0], filter[1], samples)

            # shift down
            down_hz = min_hz - basement_hz
            samples = weakutil.freq_shift(samples, -down_hz, 1.0 / self.cardrate)
            max_hz -= down_hz
            min_hz -= down_hz

            down_factor = self.cardrate // jrate
            self.jrate = jrate
            self.jblock = cardblock // down_factor

            samples = scipy.signal.decimate(samples, down_factor,
                                            n=decimate_order, zero_phase=True)
        elif self.cardrate == 12000:
            if nchildren > 1:
                sys.stderr.write("slow path: min_hz %d max_hz %d diff %d\n" % (min_hz,
                                                                               max_hz,
                                                                               max_hz - min_hz))
            self.jrate = 12000 // 2
            self.jblock = 1920 // 2
            down_factor = 2

            samples = scipy.signal.decimate(samples, down_factor,
                                            n=decimate_order, zero_phase=True)
        elif self.cardrate != self.jrate:
            sys.stderr.write("cannot handle cardrate %d; min_hz %d max_hz %d\n" % (self.cardrate, min_hz, max_hz))
            assert False

        # nominal signal start time is half a second into samples[].
        # prepend and append padding.
        # result: end_pad samples, then start of on-time signals.
        # pad with plausible signal levels.
        start_pad = int(3.0 * self.jrate)
        end_pad = int(2.5 * self.jrate)
        sm = numpy.mean(samples[self.jrate:self.jrate*4])
        sd = numpy.std(samples[self.jrate:self.jrate*4])
        sd *= padfactor
        nbefore = start_pad - self.jrate // 2
        samples = numpy.append(numpy.random.normal(sm, sd, nbefore), samples)

        wanted = 79*self.jblock + start_pad + end_pad - len(samples)
        if wanted > 0:
            samples = numpy.append(samples, numpy.random.normal(sm, sd, wanted))

        bud = budget - (time.time() - t0)
        extra_bud = self.extra_budget

        self.process0a(samples, thunk, min_hz, max_hz, start_pad,
                       bud, extra_bud, down_hz, samples_minute)

        if False:
            hzd = 0.5

            samples1 = weakutil.freq_shift_ramp(samples, [-hzd, hzd], 1.0 / self.jrate)
            self.process0a(samples1, thunk, min_hz, max_hz, start_pad,
                           bud, extra_bud, down_hz, samples_minute)
            
            samples1 = weakutil.freq_shift_ramp(samples, [hzd, -hzd], 1.0 / self.jrate)
            self.process0a(samples1, thunk, min_hz, max_hz, start_pad,
                           bud, extra_bud, down_hz, samples_minute)

    def process0a(self, samples, thunk, min_hz, max_hz, start_pad,
                  bud, extra_bud, down_hz, samples_minute):

        t0 = time.time()

        bin_hz = self.jrate / float(self.jblock)

        # I think signals should start 0.5 seconds into the minute,
        # at offset start_pad. But it seems like they generally start
        # later, captured by start_adj.
        adjusted_start = start_pad + int(self.jrate * start_adj)

        # suppress duplicate message decodes,
        # indexed by message text.
        already_msg = { }

        npasses = subpasses + 1
        for pass_ in range(0, npasses):
            used = time.time() - t0
            if used >= bud + extra_bud:
                break

            pass_got_some = False # decoded anything in this pass?

            if pass_ == 0:
                # non-subtracted
                input_samples = numpy.copy(samples)
                xf = FFTCache(samples, self.jrate, self.jblock,
                              pass0_tstep, pass0_fstep)
                ssamples = numpy.copy(samples) # for subtraction
                ranking = self.coarse(xf, adjusted_start, min_hz, max_hz,
                                      pass0_tminus, pass0_tplus)
            elif pass_ > 0:
                # revisit coarse list, with subtracted ssamples
                input_samples = numpy.copy(ssamples)
                xf = FFTCache(ssamples, self.jrate, self.jblock,
                              passN_tstep, passN_fstep)
                ssamples = numpy.copy(ssamples)
                ranking = self.coarse(xf, adjusted_start, min_hz, max_hz,
                                      passN_tminus, passN_tplus)

            # suppress duplicate attempts to look at
            # the same hz and offset. duplicates can arise
            # due to find_*slop.
            # indexed by "hz-offset".
            already_fine = { }

            # if we successfully decoded, don't bother looking
            # at nearby bins.
            # key is "%d-%d" % (offset/jblock, hz/6.25)
            already_coarse = { }

            for rr in ranking:
                used = time.time() - t0
                if used >= bud + extra_bud:
                    break
                
                # when to switch to next pass?
                deadline = None
                pass0bud = pass0_frac * (bud / float(npasses))
                pass0bud = min(bud, pass0bud)
                if pass_ == 0:
                    deadline = pass0bud
                else:
                    remainder = bud - pass0bud
                    deadline = pass0bud + pass_ * (remainder / (npasses - 1))

                #if pass_+1 < npasses and pass_got_some and used > bud * ((pass_+1.0) / npasses):
                if pass_+1 < npasses and (pass_==0 or pass_got_some) and used > deadline:
                    # switch to [next pass of] subtracted ssamples
                    break

                # rr is [ hz, offset, strength ]
                hz = rr[0]
                offset = rr[1]

                #if (hz+down_hz) > 875 and (hz+down_hz) < 886:
                #    print((hz+down_hz), rr)

                # index into already_coarse.
                acoffset = int(round(offset / float(self.jblock)))
                achz = int(round(hz / bin_hz))
                ackey = "%d-%d" % (acoffset, achz)
                if ackey in already_coarse:
                    continue

                dec = self.process1(xf, hz, offset, already_fine, pass_, down_hz, input_samples)

                if dec != None:
                    for ooo in range(acoffset-already_o, acoffset+already_o+1):
                        for hhh in range(achz-already_f, achz+already_f+1):
                            ackey = "%d-%d" % (ooo, hhh)
                            already_coarse[ackey] = True
                    pass_got_some = True
                    dec.dt = ((dec.start - start_pad) / float(self.jrate))
                    dec.minute = samples_minute
                    dec.hza[0] += down_hz
                    dec.hza[1] += down_hz
                    if do_subtract == 3 and pass_+1 < npasses:
                        ssamples = self.subtract_v6(ssamples, dec, dec.hz() - down_hz)
                    if not dec.msg in already_msg:
                        shz = dec.hz() - down_hz
                        if pass_+1 < npasses and do_subtract in [ 1, 2, 4 ]:
                            if adjust_hz_for_sub:
                                # improved hz for subtraction
                                [ _, _, m ] = xf.get(shz, dec.start)
                                shz = self.known_best_freq(ssamples, dec.symbols, dec.start, shz, m)
                            if adjust_off_for_sub:
                                err = self.known_best_off(ssamples, dec.symbols, dec.start, shz)
                                err = int(err * self.jblock)
                                if dec.start+err >= 0 and dec.start+err+79*self.jblock <= len(ssamples):
                                    dec.start += err
                        if do_subtract == 1 and pass_+1 < npasses:
                            ssamples = self.subtract_v6(ssamples, dec, shz)
                        if do_subtract == 2 and pass_+1 < npasses:
                            # multiple times, vary the hz
                            ssamples = self.subtract_v6(ssamples, dec, shz)
                            ssamples = self.subtract_v6(ssamples, dec, shz - subgap)
                            ssamples = self.subtract_v6(ssamples, dec, shz + subgap)
                        if do_subtract == 4 and pass_+1 < npasses:
                            ssamples = self.subtract_v6(ssamples, dec, shz)
                            ssamples = self.subtract_v6(ssamples, dec, dec.hz() - down_hz)
                        already_msg[dec.msg] = True
                        if self.verbose:
                            print("P%d %s %4.1f %6.1f %5d %.2f %.0f %s" % (pass_,
                                                                          self.band,
                                                                          self.second(dec.decode_time),
                                                                          dec.hz(),
                                                                          dec.start,
                                                                          dec.dt,
                                                                          dec.snr,
                                                                          dec.msg))
                            sys.stdout.flush()
                        thunk(dec)


    # down-convert to 200 samples/second,
    # or 32 samples per symbol. like wsjt-x.
    # moves hz to 25 hz, or bin=4.
    # hz must be on a coarse boundary.
    def downconvert200(self, xf, hz):
        bin_hz = self.jrate / float(self.jblock)

        gran = bin_hz / xf.fstep
        assert int(round(hz / gran)) * gran == hz

        [ hzoff, _, m ] = xf.getall_complex(hz, 0)

        # hz is centered on m[...][bin].
        bin = int(round((hz - hzoff) / bin_hz))

        assert bin >= 4

        if bin-4+17 > m.shape[1]:
            return numpy.array([])

        # move bin to bin=4 == 25 Hz.
        # need 17 bins for inverse FFT to yield 32 samples/symbol.
        m1 = m[:,bin-4:bin-4+17]

        # inverse FFTs, to generate sample rate of 200/second.
        slow = numpy.zeros(32*m1.shape[0])
        for i in range(0, m1.shape[0]):
            zz = m1[i,:]
            z = weakutil.irfft(zz)
            slow[i*32:(i+1)*32] = z

        return slow

    # return a 79x8 complex mini-FFT.
    # s is 200 samples/second, 32 samples per symbol.
    def extract200(self, s, off, hz):
        bin = int(round(hz / 6.25))
        assert bin >= 3 and bin <= 5
        hzshift = hz - 6.25 * bin
        if hzshift != 0.0:
            s = weakutil.freq_shift(s, -hzshift, 1.0/200)

        s = s[off:off+79*32]

        bb = numpy.reshape(s, (79,32))
        bb = numpy.copy(bb) # needed for fftw rfftn
        # mm = numpy.fft.rfftn(bb, axes=[1])
        mm = weakutil.rfftn(bb, axes=[1])

        m = mm[:,bin:bin+8]

        return m

    def process1(self, xf, hz, offset, already_hzo, pass_, down_hz, samples):
        if down200:
            return self.process1_v3(samples, xf, hz, offset, already_hzo, pass_, down_hz)
        else:
            return self.process1_v1(xf, hz, offset, already_hzo, pass_, down_hz)

    # hz and offset are from coarse search.
    def process1_v1(self, xf, hz, offset, already_hzo, pass_, down_hz):
        bin_hz = self.jrate / float(self.jblock)

        hz0 = hz
        offset0 = offset

        if offset < 0 or (offset+79*self.jblock) > xf.len():
            return None

        offstep = self.jblock // xf.tstep
        if fine_tstep > 1:
            offstep //= fine_tstep
        hzstep = bin_hz / xf.fstep
        if fine_fstep > 1:
            hzstep /= fine_fstep

        if fine_tstep > 1:
            # improve the starting offset.
            off0 = offset - (fine_tstep // 2) * offstep
            off1 = offset + (fine_tstep // 2) * offstep
            offs = self.best_offsets(xf, hz, off0, off1, offstep)
            offs = offs[0:fine_no]
        else:
            offs = [ [ offset, 0 ] ]

        for [ offset, _ ] in offs:
            if fine_fstep > 1:
                # improve the starting hz.
                hzoff0 = hz0 - (fine_fstep // 2) * hzstep
                hzoff1 = hz0 + (fine_fstep // 2) * hzstep
                [ hz, strength ] = self.best_freq(xf, offset, hzoff0, hzoff1, hzstep)

            # the fine_ loops overlap at the edges. try each point just once.
            khz = int(round(hz))
            koff = offset // offstep
            key = "%d-%d" % (khz, koff)
            if key in already_hzo:
                continue
            already_hzo[key] = True

            [ _, _, ss ] = xf.get_complex(hz, offset)
            # ss has 79 8-bucket mini-FFTs.

            dec = self.process2(ss[0:79], hz, offset, pass_, down_hz)
                
            if dec != None:
                return dec

        return None

    # hz and offset are from coarse search.
    # for down200
    def process1_v3(self, samples, xf, hz1, offset1, already_fine, pass_, down_hz):
        bin_hz = self.jrate / float(self.jblock)

        if offset1 < 0 or (offset1+79*self.jblock) > xf.len():
            return None
        
        # 200 samples/second, 32 samples/symbol,
        # hz moved to bin 4 (25 Hz).

        if True:
            slow = self.downconvert200(xf, hz1)
            if len(slow) == 0:
                return None
        else:
            filter = weakutil.butter_bandpass(hz1 - guard200, hz1 + 50 + guard200, self.jrate, order200)

            # samples = scipy.signal.lfilter(filter[0], filter[1], samples)
            # filtfilt() doesn't change the offset (phase) of the signal,
            # so it works better here than lfilter(), though it is half
            # as fast (and has twice the effective order).
            samples = scipy.signal.filtfilt(filter[0], filter[1], samples)

            assert hz1 >= 25
            samples = weakutil.freq_shift(samples, -(hz1 - 25), 1.0 / self.jrate)
            slow = samples[::(self.jrate // 200)]

        offset = int((offset1 / float(self.jrate)) * 200)
        hz = 25.0

        if offset < 0 or offset + 79*32 > len(slow):
            return None

        m = self.extract200(slow, offset, hz)
        m = abs(m)

        symbols = numpy.argmax(m, 1)
        costas = numpy.array(costas_symbols, dtype=numpy.int32)
        symbols[0:0+7] = costas
        symbols[36:36+7] = costas
        symbols[72:72+7] = costas

        # hz = self.known_best_freq(slow, symbols, offset, hz, m, 200, 32)
        # err = self.known_best_off(slow, symbols, offset, hz, 200, 32)
        hz = self.blah_freq(slow, symbols, offset, hz, m, 200, 32, pass_)
        err = self.blah_off(slow, symbols, offset, hz, 200, 32, pass_)
        err = int(err * 32)
        offset += err

        offset = max(offset, 0)
        offset = min(offset, len(slow) - 79*32)

        m = self.extract200(slow, offset, hz)
        m = abs(m)

        # reference hz/offset back to self.jrate world.
        offset2 = int((offset / 200.0) * self.jrate)
        hz2 = hz1 + (hz - 25.0)

        dec = self.process2(m[0:79], hz2, offset2, pass_, down_hz)

        if dec != None:
            return dec

        return None

    # use freq_from_fft to find offset from center of bin.
    # this version knows the transmitted signal!
    # for subtraction.
    def known_best_freq(self, samples, symbols, start, hz, m, jrate=None, jblock=None):
        if jrate == None:
            jrate = self.jrate
        if jblock == None:
            jblock = self.jblock

        bin_hz = jrate / float(jblock)
        bin = bin_of(hz)

        hz_sum = 0.0
        weight_sum = 0.0
        
        for i in range(0, len(symbols)):
            ind = start + i*(jblock)
            block = samples[ind:ind+jblock]
            bin1 = bin + symbols[i]
            # bin_from_fft() returns a fractional bin
            bin2 = weakutil.bin_from_fft(block, jrate, bin1)
            if bin2 == None:
                continue
            nhz = bin2 * bin_hz
            nhz -= symbols[i] * bin_hz # reference to base tone
            if abs(nhz - hz) < bin_hz*0.5:
                #weight = 1
                weight = m[i][symbols[i]] / numpy.min(m[i][0:8])
                hz_sum += nhz * weight
                weight_sum += weight
        if weight_sum == 0:
            return hz
        return hz_sum / weight_sum

    def blah_freq(self, samples, symbols, start, hz, m, jrate, jblock, pass_):

        bin_hz = jrate / float(jblock)
        bin = bin_of(hz)

        hz_sum = 0.0
        weight_sum = 0.0
        
        for i in [ 0, 1, 2, 3, 4, 5, 6,
                   36, 37, 38, 39, 40, 41, 42,
                   72, 73, 74, 75, 76, 77, 78 ]:
            ind = start + i*(jblock)
            block = samples[ind:ind+jblock]
            bin1 = bin + symbols[i]
            # bin_from_fft() returns a fractional bin
            bin2 = weakutil.bin_from_fft(block, jrate, bin1)
            if bin2 == None:
                continue
            nhz = bin2 * bin_hz
            nhz -= symbols[i] * bin_hz # reference to base tone
            if abs(nhz - hz) < bin_hz*0.5:
                #weight = 1
                weight = m[i][symbols[i]] / numpy.min(m[i][0:8])
                hz_sum += nhz * weight
                weight_sum += weight
        if weight_sum == 0:
            return hz
        return hz_sum / weight_sum

    # try to find offset error in time domain (not fft).
    # returns time offset error in fractions of a symbol time.
    # return value should be added to offset.
    def known_best_off(self, samples, symbols, start, hz, jrate=None, jblock=None):
        if jrate == None:
            jrate = self.jrate
        if jblock == None:
            jblock = self.jblock

        bin_hz = jrate / float(jblock)

        # pre-compute the eight tones. the phase won't be
        # right, so the final results will be off by
        # around half a cycle, which is too bad, though
        # averaging over many symbols may help.
        tones = [ ]
        for ti in range(0, 8):
            # random phase
            ph = random.random() * math.pi
            tone = weakutil.costone(jrate, hz + bin_hz*ti, jblock, ph)
            tones.append(tone)

        gran = min(pass0_tstep, passN_tstep)
        if down200 == False and fine_tstep > 0:
            gran *= fine_tstep
        max_err = int(round(jblock / float(gran)))

        sum = 0.0
        n = 0

        i = 0
        while i < len(symbols):
            ix = i + 1
            while ix < len(symbols) and symbols[ix] == symbols[i]:
                ix += 1
            nsyms = ix - i

            i0 = start + i*jblock
            i1 = i0 + nsyms*jblock

            if i0 >= max_err and i1+max_err <= len(samples):
                if nsyms > 1:
                    tone = numpy.tile(tones[symbols[i]], nsyms)
                else:
                    tone = tones[symbols[i]]

                block = samples[i0-max_err:i1+max_err]
                cc = numpy.correlate(block, tone)
                mm = numpy.argmax(cc)

                thisoff = mm - max_err
                weight = math.sqrt(max(0, cc[mm])) * nsyms
                sum += thisoff * weight
                n += weight

            i = ix

        err = sum / n
        err = err / float(jblock)
        return err

    def blah_off(self, samples, symbols, start, hz, jrate, jblock, pass_):

        bin_hz = jrate / float(jblock)

        # pre-compute the eight tones. the phase won't be
        # right, so the final results will be off by
        # around half a cycle, which is too bad, though
        # averaging over many symbols may help.
        tones = [ ]
        for ti in range(0, 8):
            # random phase
            ph = random.random() * math.pi
            tone = weakutil.costone(jrate, hz + bin_hz*ti, jblock, ph)
            tones.append(tone)

        if pass_ == 0:
            gran = pass0_tstep
        else:
            gran = passN_tstep
        if down200 == False and fine_tstep > 0:
            gran *= fine_tstep
        max_err = int(round(jblock / float(gran)))

        sum = 0.0
        n = 0

        for i in [ 0, 1, 2, 3, 4, 5, 6,
                   36, 37, 38, 39, 40, 41, 42,
                   72, 73, 74, 75, 76, 77, 78 ]:
            ix = i + 1
            while ix < len(symbols) and symbols[ix] == symbols[i]:
                ix += 1
            nsyms = ix - i

            i0 = start + i*jblock
            i1 = i0 + nsyms*jblock

            if i0 >= max_err and i1+max_err <= len(samples):
                if nsyms > 1:
                    tone = numpy.tile(tones[symbols[i]], nsyms)
                else:
                    tone = tones[symbols[i]]

                block = samples[i0-max_err:i1+max_err]
                cc = numpy.correlate(block, tone)
                mm = numpy.argmax(cc)

                thisoff = mm - max_err
                weight = math.sqrt(max(0, cc[mm])) * nsyms
                sum += thisoff * weight
                n += weight

            #i = ix

        err = sum / n
        err = err / float(jblock)
        return err

    # subtract a decoded signal (hz/start/twelve) from the samples,
    # so that we can then decode weaker signals underneath it.
    # i.e. interference cancellation.
    # generates the right tone for each symbol, finds the best
    # offset w/ correlation, finds the amplitude, subtracts in the time domain.
    def subtract_v5(self, osamples, dec, hz0):
        bin_hz = self.jrate / float(self.jblock)

        # the 79 symbols, each 0..8
        symbols = dec.symbols

        samples = numpy.copy(osamples)

        if dec.start < 0:
            samples = numpy.append([0.0]*(-dec.start), samples)
        else:
            samples = samples[dec.start:]

        # pre-compute the tones, since costone() is somewhat expensive.
        # tone_cache[sym_num][phase_i]
        tone_cache = [ ]
        for sym in range(0, 8):
            ta = [ ]
            for phase_i in range(0, substeps):
                hz = hz0 + sym * bin_hz
                phase = phase_i * ((2*numpy.pi) / substeps)
                tone = weakutil.costone(self.jrate, hz, self.jblock, phase)
                ta.append(tone)
            tone_cache.append(ta)

        # find amplitude and offset (phase) of each symbol.
        amps = [ ]
        offs = [ ]
        tones = [ ]
        i = 0
        while i < len(symbols):
            nb = 1
            #while i+nb < len(symbols) and symbols[i+nb] == symbols[i]:
            #    nb += 1

            # start+end of symbol in samples[]
            i0 = i * self.jblock
            i1 = i0 + nb*self.jblock

            # try a few different phases, remember the best one.
            best_corr = 0.0
            best_tone = numpy.array([])
            for phase_i in range(0, substeps):
                tone = tone_cache[symbols[i]][phase_i]
                corr = numpy.sum(tone * samples[i0:i1])
                if len(best_tone) == 0 or corr > best_corr:
                    best_tone = tone
                    best_corr = corr

            assert len(best_tone) == i1-i0

            # what is the amplitude?
            # if actual signal had a peak of 1.0, then
            # correlation would be sum(tone*tone).
            c1 = numpy.sum(best_tone * best_tone)
            a = best_corr / c1

            amps.append(a)
            offs.append(i0)
            tones.append(best_tone)

            i += nb

        for ai in range(0, len(amps)):
            a = amps[ai]
            off = offs[ai]
            samples[off:off+len(tones[ai])] -= tones[ai] * a

        if dec.start < 0:
            nsamples = samples[(-dec.start):]
        else:
            nsamples = numpy.append(osamples[0:dec.start], samples)

        return nsamples

    # subtract a decoded signal (hz/start/twelve) from the samples,
    # so that we can then decode weaker signals underneath it.
    # i.e. interference cancellation.
    def subtract_v6(self, osamples, dec, hz0):
        bin_hz = self.jrate / float(self.jblock)

        # the 79 symbols, each 0..8
        symbols = dec.symbols

        bin0 = int(hz0 / bin_hz)
        down_hz = hz0 - (bin0 * bin_hz)
        downsamples = weakutil.freq_shift(osamples, -down_hz, 1.0/self.jrate)

        if dec.start < 0:
            samples = numpy.append([0.0]*(-dec.start), downsamples)
        else:
            samples = downsamples[dec.start:]

        ffts = [ ]
        ampls = numpy.zeros(len(symbols))
        i = 0
        while i < len(symbols):
            # start+end of symbol in samples[]
            i0 = i * self.jblock
            i1 = i0 + self.jblock
            a = weakutil.rfft(samples[i0:i1])
            ffts.append(a)
            ampls[i] = abs(a[bin0+symbols[i]])
            i += 1

        # XXX ampls[] includes N as well as S, but nothing
        # obvious works well to fix that.

        medamps = scipy.signal.medfilt(ampls, kernel_size=((sub_amp_win*2)+1))

        i = 0
        while i < len(symbols):
            # start+end of symbol in samples[]
            i0 = i * self.jblock
            i1 = i0 + self.jblock

            ampl = medamps[i]

            a = ffts[i]
            si = bin0 + symbols[i]

            aa = abs(a[si])
            if ampl > aa:
                ampl = aa

            if aa > 0.0:
                a[si] /= aa
                a[si] *= (aa - ampl)

            samples[i0:i1] = weakutil.irfft(a)

            i += 1

        if dec.start < 0:
            nsamples = samples[(-dec.start):]
        else:
            nsamples = numpy.append(downsamples[0:dec.start], samples)

        nsamples = weakutil.freq_shift(nsamples, down_hz, 1.0/self.jrate)

        return nsamples

    # don't freq_shift(); instead, subtract FFT of
    # the tone, which is smeared over many FFT buckets.
    # does not work as well as subtract_v6.
    def subtract_v7(self, osamples, dec, hz0):
        bin_hz = self.jrate / float(self.jblock)

        # the 79 symbols, each 0..8
        symbols = dec.symbols

        if dec.start < 0:
            samples = numpy.append([0.0]*(-dec.start), osamples)
        else:
            samples = numpy.copy(osamples[dec.start:])

        # FFT of a tone.
        tone = weakutil.costone(self.jrate, hz0, self.jblock, 0.0)
        tonefft = weakutil.rfft(tone)
        bin0 = numpy.argmax(abs(tonefft))

        ffts = [ ]
        ampls = numpy.zeros(len(symbols))
        for i in range(0, len(symbols)):
            # start+end of symbol in samples[]
            i0 = i * self.jblock
            i1 = i0 + self.jblock
            a = weakutil.rfft(samples[i0:i1])
            ffts.append(a)
            ampls[i] = abs(a[bin0+symbols[i]])

        medamps = scipy.signal.medfilt(ampls, kernel_size=((sub_amp_win*2)+1))

        for i in range(0, len(symbols)):
            # start+end of symbol in samples[]
            i0 = i * self.jblock
            i1 = i0 + self.jblock

            a = ffts[i]
            si = bin0 + symbols[i]

            # FFT is linear, so subtracting in time domain is the
            # same as subtracting in frequency domain.
            # complex multiplication adds angles and multiplies magnitudes.
            # complex division subtracts angles and divides magnitudes.

            # reference tone is wrong phase; what's the difference?
            # i.e. need to multiply reference tone by phasediff to yield signal.
            phasediff = a[si] / tonefft[bin0]
            phasediff /= abs(phasediff)

            # reference tone is wrong amplitude; what's the ratio?
            # ampdiff = abs(a[si]) / abs(tonefft[bin0])
            ampdiff = medamps[i] / abs(tonefft[bin0])

            #diff = numpy.copy(tonefft)
            #diff *= phasediff
            #diff *= ampdiff
            diff = tonefft * phasediff * ampdiff

            a[symbols[i]:] -= diff[0:len(diff)-symbols[i]]

            samples[i0:i1] = weakutil.irfft(a)

        if dec.start < 0:
            nsamples = samples[(-dec.start):]
        else:
            nsamples = numpy.append(osamples[0:dec.start], samples)

        return nsamples

    def all_ffts(self, samples):
        assert len(samples) >= 79*self.jblock

        bins = numpy.zeros([ 79, self.jblock/2+1 ], dtype=numpy.complex128)
        for si in range(0, 79):
            i0 = si * self.jblock
            i1 = i0 + self.jblock
            a = weakutil.rfft(samples[i0:i1])
            bins[si,:] = a
        return bins

    # subtract a decoded signal (hz/start/twelve) from the samples,
    # so that we can then decode weaker signals underneath it.
    # i.e. interference cancellation.
    def subtract_v8(self, osamples, dec, hz0):
        bin_hz = self.jrate / float(self.jblock)

        # the 79 symbols, each 0..8
        symbols = dec.symbols

        bin0 = int(round(hz0 / bin_hz))
        down_hz = hz0 - (bin0 * bin_hz)

        # center signal in an FFT bin, since we only subtract from
        # a single bin per symbol time.
        downsamples = weakutil.freq_shift(osamples, -down_hz, 1.0/self.jrate)

        if dec.start < 0:
            samples = numpy.append([0.0]*(-dec.start), downsamples)
        else:
            samples = downsamples[dec.start:]

        # re-do an FFT per symbol time.
        ffts = self.all_ffts(samples)

        i = 0
        while i < len(symbols):
            # start+end of symbol in samples[]
            i0 = i * self.jblock
            i1 = i0 + self.jblock

            a = ffts[i,:]
            si = bin0 + symbols[i]

            a[si] = 0.0

            samples[i0:i1] = weakutil.irfft(a)

            i += 1

        if dec.start < 0:
            nsamples = samples[(-dec.start):]
        else:
            nsamples = numpy.append(downsamples[0:dec.start], samples)

        nsamples = weakutil.freq_shift(nsamples, down_hz, 1.0/self.jrate)

        return nsamples

    def strength_v3(self, m79):
        a = m79[0:7] + m79[36:43] + m79[72:79]

        yes = 0.0
        
        for i in range(0,len(costas_symbols)):
            yes += a[i,costas_symbols[i]]

        sum = yes / (numpy.sum(a) - yes)

        return sum

    # contrast: how much the strongest bin
    # in each symbol time is stronger than the second-
    # strongest.
    # sort the 8 bins in each symbol time.
    # (no longer used)
    def contrast(self, m79):
        sss = numpy.sort(m79, 1)
        aaa = sss[:,7]
        bbb = sss[:,6]

        contrasts = numpy.divide(aaa,
                                 bbb,
                                 out=numpy.ones(len(m79)),
                                 where=bbb!=0)

        contrast = numpy.sum(contrasts)
        contrast /= len(m79)
        contrast -= 1.0

        return contrast

    # find hz with best Costas sync at offset=start.
    # look at frequencies midhz +/ slop,
    # at granule hz increments.
    # returns [ hz, strength ]
    def best_freq(self, xf, start, hzoff0, hzoff1, hzstep):
        start = int(start)

        rank = [ ]
        for hz in numpy.arange(hzoff0, hzoff1*1.0001, hzstep):
            [ _, _, m79 ] = xf.get(hz, start)
            if len(m79) < 79:
                continue
            m79 = m79[0:79]
            # m79 has 79 8-bucket mini-FFTs.

            c = self.strength_v3(m79)

            rank.append([hz, c])

        rank = sorted(rank, key = lambda e : - e[1])
        return rank[0]

    # find offset with best Costas sync at hz.
    # looks at offsets at start +/- slop,
    # at granule offset increments.
    # returns [ [ start, strength ], ... ]
    def best_offsets(self, xf, hz, off0, off1, step):
        rank = [ ]
        for xoff in numpy.arange(off0, off1+0.0001, step):
            off = int(xoff)
            if off + 79 * self.jblock > xf.len():
                continue
            [ _, _, m79 ] = xf.get(hz, off)
            if len(m79) < 79:
                continue

            m79 = m79[0:79]
            # m79 has 79 8-bucket mini-FFTs.

            c = self.strength_v3(m79)

            rank.append([off, c])

        rank = sorted(rank, key = lambda e : - e[1])
        return rank

    def coarse1(self, xf, adjusted_start, hzoff, offoff, min_hz, max_hz,
                tminus, tplus):
        # prepare a template for 2d correlation containing
        # the three Costas arrays.
        template = numpy.zeros((79, 8))
        for i0 in [ 0, 36, 72 ]:
            for i1 in range(0, 7):
                template[i0+i1,:] = -1 / 7.0
                template[i0+i1,costas_symbols[i1]] = 1

        # m[symbol][bin]
        [ hz0, offset0, m ] = xf.getall(hzoff, offoff)

        bin_hz = self.jrate / float(self.jblock)
        min_hz_bin = bin_of(min_hz - hz0)
        max_hz_bin = bin_of(max_hz - hz0) + 8

        min_sym = int((adjusted_start - self.jrate*tminus) / self.jblock)
        max_sym = int((adjusted_start + self.jrate*tplus) / self.jblock)

        m = m[min_sym:79+max_sym,min_hz_bin:max_hz_bin]

        if True:
            # instead of matching the full 79-symbol template against
            # the full spectrum matrix m[][], just match the costas part
            # of the template against the costas part of m[][].
            # in practice, this is neither a win nor a lose.
            minussyms = int(round((self.jrate * tminus) / float(self.jblock)))
            plussyms = int(round((self.jrate * tplus) / float(self.jblock)))
            slopsyms = minussyms + plussyms
            m = numpy.concatenate([ m[0:7+slopsyms],
                                    m[36:43+slopsyms],
                                    m[72:79+slopsyms] ])
            template = numpy.concatenate( [ template[0:7+slopsyms],
                                            template[36:43+slopsyms],
                                            template[72:79] ] )

        # for each frequency bin, the total signal level for
        # it and the next eight bins up. we'll divide by this
        # in order to emphasize the correlation, not the
        # signal (or noise) level.
        binsum = numpy.sum(m, axis=0)
        norm = numpy.zeros(len(binsum))
        for i in range(0, 8):
            norm[0:len(norm)-i] += binsum[i:]

        c = scipy.signal.correlate2d(m, template, mode='valid')

        if False:
            h = [ [ (bi+min_hz_bin) * bin_hz + hz0,
                    (si+min_sym) * self.jblock + offoff,
                    c[si,bi] / norm[bi]
                  ]
                  for si in range(0, c.shape[0]) for bi in range(0, c.shape[1]) ]
        else:
            # best few starting symbol indices for each frequency bin.
            # so we only return a few elements per bin, not
            # one element per bin per starting symbol index.
            max_si = numpy.argsort(-c, axis=0)
            h = [ ]
            for mi in range(0, coarse_no):
                h += [ [ (bi+min_hz_bin) * bin_hz + hz0,
                        (max_si[mi][bi]+min_sym) * self.jblock + offoff,
                        c[max_si[mi][bi],bi] / norm[bi]
                      ]
                      for bi in range(0, max_si.shape[1]) ]

        return h

    def coarse(self, xf, adjusted_start, min_hz, max_hz,
               tminus, tplus):
        bin_hz = self.jrate / float(self.jblock)

        h = [ ]
        for hzoff in numpy.arange(0.0, bin_hz, bin_hz / xf.fstep):
            for offoff in range(0, self.jblock, int(self.jblock / xf.tstep)):
                hx = self.coarse1(xf, adjusted_start, hzoff, offoff,
                                  min_hz, max_hz,
                                  tminus, tplus)
                h += hx

        h = sorted(h, key = lambda e : -e[2])

        return h

    # m79 is 79 8-bucket mini FFTs, for 8-FSK demodulation.
    # m79[0..79][0..8]
    def snr(self, m79):
        # estimate SNR.
        # mimics wsjt-x code, though the results are not very close.
        sigi = numpy.argmax(m79, axis=1)
        noisei = numpy.mod(sigi + 4, 8)
        noises = m79[range(0, 79), noisei]
        noise = numpy.mean(noises * noises) # square yields power
        #if noise == 0.0:
        #    # !!!
        #    return 0
        sigs = numpy.amax(m79, axis=1) # guess correct tone
        sig = numpy.mean(sigs * sigs)
        rawsnr = sig / noise
        rawsnr -= 1 # turn (s+n)/n into s/n
        if rawsnr < 0.1:
            rawsnr = 0.1
        rawsnr /= (2500.0 / 2.7) # 2.7 hz noise b/w -> 2500 hz b/w
        snr = 10 * math.log10(rawsnr)
        snr += 3
        return snr

    # three un-gray-coded Costas arrays.
    ci3 = numpy.array([ 2, 5, 6, 0, 4, 1, 3,
                        2, 5, 6, 0, 4, 1, 3,
                        2, 5, 6, 0, 4, 1, 3 ], dtype=numpy.int32)

    # mean and std dev of winning and non-winning
    # FFT bins, to help compute probabilities of
    # symbol values for soft decoding.
    # m79 is 79 8-bucket mini FFTs, for 8-FSK demodulation.
    # m79[0..79][0..8]
    # return [ winmean, winstd, losemean, losestd ],
    def softstats(self, m79, i0=0, i1=79):

        if True:
            # 1210 of 1370
            # 1120 of 1370
            # 1215 of 1370
            # 1212 of 1370
            # 1213 of 1370
            # for some reason it's apparently important that losers
            # include winners!
            winners = numpy.max(m79[i0:i1], 1)
            losers = m79[i0:i1]

        if False:
            # 1184
            x = self.zeroone(m79)
            winners = numpy.max(x[i0:i1], 1)
            losers = numpy.min(x[i0:i1], 1)

        if False:
            xx = numpy.sort(m79[i0:i1], 1)
            winners = xx[:,soft1:soft2]
            losers = xx[:,soft3:soft4]

        if False:
            # 1172 of 1370
            # loglikelihood() compares strongest with second-strongest.
            # no: score 489 (451 of 553)
            xx = numpy.sort(m79[i0:i1], 1)
            winners = xx[:,7]
            losers = xx[:,5:7] # 407
            # losers = xx[:,6:7] # 402

        if False:
            # 1167 of 1370
            # 1173 of 1370
            winners = numpy.max(m79[i0:i1], 1)
            losers = winners

        if False:
            # 1202 of 1370
            # 1205 of 1370
            # 1205 of 1370
            # this works remarkably well!
            winners = m79
            losers = m79

        if False:
            # 1211 of 1370
            # 1206 of 1370
            # 1204 of 1370
            winners = numpy.max(m79[i0:i1], 1)
            losers = m79[i0:i1,0]

        if False:
            # 1136 of 1370
            # 1141 of 1370
            # 1127 of 1370
            xxx = numpy.argmax(m79[i0:i1], 1)
            winners = m79[range(i0,i1),xxx]
            xxx = numpy.mod(xxx + 4, 8)
            losers = m79[range(i0,i1),xxx]

        if False:
            # 1139 of 1370
            # 1143 of 1370
            # 1145 of 1370
            xxx = numpy.argmax(m79[i0:i1], 1)
            winners = m79[range(i0,i1),xxx]
            losers = numpy.concatenate( (
                m79[range(i0,i1),numpy.mod(xxx + 2, 8)],
                m79[range(i0,i1),numpy.mod(xxx + 4, 8)],
                m79[range(i0,i1),numpy.mod(xxx + 6, 8)] ) )

        if False:
            # 1153 of 1370
            # 477 (439 of 553)
            # sort each symbol-time's 8 bins,
            # so that xx[i][7] is biggest.
            xx = numpy.sort(m79[i0:i1], 1)
            winners = xx[:,7]
            losers = xx[:,0:7]

        if False:
            # 1123 of 1370
            # 1126 of 1370
            # this works best, ft8-n13/c.txt score 522 (470 of 553)
            winners = numpy.max(m79[i0:i1], 1)
            losers = numpy.sort(m79[i0:i1].flatten())[0:-79]

        if False:
            # 1094 of 1370
            # 1109 of 1370
            all = numpy.sort(m79[i0:i1].flatten())
            winners = all[-79:]
            losers = all[0:-79]

        if False:
            # 1041 of 1370
            # 1043 of 1370
            winners = numpy.max(m79[i0:i1], 1)
            losers = numpy.min(m79[i0:i1], 1)

        if False:
            # 595 (575 of 675)
            # just look at Costas arrays, where we know
            # which symbols are correct.
            cs = numpy.concatenate( [ m79[0:0+7], m79[36:36+7], m79[72:72+7] ] )
            winners = cs[range(0,21),self.ci3]
            losers = m79

        if False:
            # this works pretty well, old 575 (555 of 675).
            # just look at Costas arrays, where we know
            # which symbols are correct.
            winners = numpy.zeros(3*7)
            losers = numpy.zeros(3*7*7)
            wi = 0
            li = 0
            for i0 in [ 0, 36, 72 ]:
                for i1 in range(0, 7):
                    cs = costas_symbols[i1]
                    winners[wi] = m79[i0+i1][cs]
                    wi += 1
                    losers[li:li+cs] = m79[i0+i1][0:cs]
                    losers[li+cs:li+7] = m79[i0+i1][cs+1:]
                    li += 7

        if False:
            # this works pretty well, old 578 (558 of 675)
            n = len(m79)
            winners = numpy.zeros(n)
            losers = numpy.zeros(n * 7)
            for mi in range(0, n):
                e = m79[mi]
                wini = numpy.argmax(e) # guess the winning tone.
                winners[mi] = e[wini]
                li = mi * 7
                losers[li:li+wini] = e[0:wini]
                losers[li+wini:li+7] = e[wini+1:]

        if False:
            # this is ok, 573 (555 of 675)
            # look at all symbols.
            # we know the correct Costas values.
            # guess the others based on strength.
            winners = [ ]
            losers = [ ]
            for i in range(i0, i1):
                e = m79[i]
                good = False
                loudest = numpy.argmax(e)
                if (i >= 0 and i < 7) or (i >= 36 and i < 36+7) or (i >= 72 and i < 72+7):
                    # in a costas block, so we know the correct symbol.
                    if i >= 72:
                        ci = i - 72
                    elif i >= 36:
                        ci = i - 36
                    else:
                        ci = i
                    wini = costas_symbols[ci]
                    good = (wini == loudest)
                else:
                    # we don't know correct symbol; use loudest.
                    wini = loudest
                winners.append(e[wini])
                if good:
                    # higher weight.
                    winners.append(e[wini])

                losers += list(e[0:wini])
                losers += list(e[wini+1:])

        if False:
            import matplotlib.pyplot as plt
            plt.plot(winners)
            #plt.plot(losers[::7])
            plt.plot(losers.flatten())
            plt.show()

        winmean = numpy.mean(winners)
        winstd = numpy.std(winners)
        losemean = numpy.mean(losers)
        losestd = numpy.std(losers)

        if False:
            print("%.2f %.2f, %.2f %.2f" % (winmean, winstd, losemean, losestd))
            numpy.savetxt("win.txt", winners, fmt="%f")
            numpy.savetxt("lose.txt", losers.flatten(), fmt="%f")
            sys.exit(1)

        return [ winmean, winstd, losemean, losestd ]

    # m58[58][8]. returns, for each of 3*58 bits,
    # an array of two values: the strength of the strongest tone
    # that would make the bit a zero, and a one.
    # m58 should have been de-gray-coded.
    def zeroone(self, m58):
        n = len(m58)

        x174 = numpy.zeros([ 3 * n, 2 ])

        bi = 0
        for [ v0, v1 ] in [
                # symbol numbers that make this bit zero and one.
                # most-significant bit first.
                [ [ 0, 1, 2, 3 ], [ 4, 5, 6, 7 ] ],
                [ [ 0, 1, 4, 5 ], [ 2, 3, 6, 7 ] ],
                [ [ 0, 2, 4, 6 ], [ 1, 3, 5, 7 ] ],
                ]:
            # treat each of the three bits in a symbol separately.

            # eX[i] is max signal over the four bins that could
            # cause the symbol to yield an X for this bit.

            e0 = numpy.maximum(
                   numpy.maximum(m58[range(0,n),v0[0]], m58[range(0,n),v0[1]]),
                   numpy.maximum(m58[range(0,n),v0[2]], m58[range(0,n),v0[3]]))
            e1 = numpy.maximum(
                   numpy.maximum(m58[range(0,n),v1[0]], m58[range(0,n),v1[1]]),
                   numpy.maximum(m58[range(0,n),v1[2]], m58[range(0,n),v1[3]]))

            x174[bi::3] = numpy.stack([e0, e1], axis=1)

            bi += 1

        return x174

    # m58[58][8]. returns, for each of 3*58 bits,
    # an array of 8 values: the strengths of the tones
    # that would make the bit a zero, strongest first;
    # and similarly for one.
    # m58 should have been de-gray-coded.
    def zeroone8(self, m58):
        n = len(m58)

        x174 = numpy.zeros([ 3 * n, 8 ])

        bi = 0
        for [ v0, v1 ] in [
                # symbol numbers that make this bit zero and one.
                # most-significant bit first.
                [ [ 0, 1, 2, 3 ], [ 4, 5, 6, 7 ] ],
                [ [ 0, 1, 4, 5 ], [ 2, 3, 6, 7 ] ],
                [ [ 0, 2, 4, 6 ], [ 1, 3, 5, 7 ] ],
                ]:
            # treat each of the three bits in a symbol separately.

            # eX[i] is max signal over the four bins that could
            # cause the symbol to yield an X for this bit.

            e0 = numpy.stack( [
                m58[range(0,n),v0[0]],
                m58[range(0,n),v0[1]],
                m58[range(0,n),v0[2]],
                m58[range(0,n),v0[3]]
              ], axis=1 )
            e0 = numpy.sort(e0, axis=1)

            e1 = numpy.stack( [
                m58[range(0,n),v1[0]],
                m58[range(0,n),v1[1]],
                m58[range(0,n),v1[2]],
                m58[range(0,n),v1[3]]
              ], axis=1 )
            e1 = numpy.sort(e1, axis=1)

            st = numpy.stack([ e0[:,0], e0[:,1], e0[:,2], e0[:,3],
                               e1[:,0], e1[:,1], e1[:,2], e1[:,3] ], axis=1)

            x174[bi::3] = st

            bi += 1

        return x174

    # like zeroone() but special case costas.
    # after de-gray-code, so Costas is 2 1 6 0 5 4 3
    # return is [ winners, losers ]
    # one per bit.
    # intended for statistics.
    def winlose(self, m79):
        graycostas = numpy.array([2, 1, 6, 0, 5, 4, 3], dtype=numpy.int)

        if False:
            n = len(m79)
            winsym = numpy.argmax(m79, 1)
            winsym[0:7] = graycostas
            winsym[36:36+7] = graycostas
            winsym[72:72+7] = graycostas

        if True:
            winsym = numpy.concatenate([ graycostas, graycostas, graycostas ])
            m79 = numpy.concatenate([ m79[0:7], m79[36:36+7], m79[72:72+7] ])
            n = len(m79)

        # turn each 8-level symbol into 3 bits,
        # most significant bit first.
        winbit = numpy.zeros(n*3, dtype=numpy.int)
        winbit[0::3] = numpy.bitwise_and(numpy.right_shift(winsym, 2), 1) # 4
        winbit[1::3] = numpy.bitwise_and(numpy.right_shift(winsym, 1), 1) # 2
        winbit[2::3] = numpy.bitwise_and(numpy.right_shift(winsym, 0), 1) # 1

        winners = numpy.zeros(3 * n)
        losers = numpy.zeros(3 * n)

        bi = 0
        for [ v0, v1 ] in [
                # symbol numbers that make this bit zero and one.
                # most-significant bit first.
                [ [ 0, 1, 2, 3 ], [ 4, 5, 6, 7 ] ],
                [ [ 0, 1, 4, 5 ], [ 2, 3, 6, 7 ] ],
                [ [ 0, 2, 4, 6 ], [ 1, 3, 5, 7 ] ],
                ]:
            # treat each of the three bits in a symbol separately.

            # eX[i] is max signal over the four bins that could
            # cause the symbol to yield an X for this bit.

            e0 = numpy.maximum(
                   numpy.maximum(m79[range(0,n),v0[0]], m79[range(0,n),v0[1]]),
                   numpy.maximum(m79[range(0,n),v0[2]], m79[range(0,n),v0[3]]))
            e1 = numpy.maximum(
                   numpy.maximum(m79[range(0,n),v1[0]], m79[range(0,n),v1[1]]),
                   numpy.maximum(m79[range(0,n),v1[2]], m79[range(0,n),v1[3]]))

            #winners[bi::3] = numpy.maximum(e0, e1)
            #losers[bi::3] = numpy.minimum(e0, e1)

            winners[bi::3] = numpy.where(winbit[bi::3]!=0, e1, e0)
            losers[bi::3] = numpy.where(winbit[bi::3]==0, e1, e0)

            bi += 1

        if False:
            losers = numpy.concatenate([winners,losers])

        return [ winners, losers ]

    def probyes(self, sig, winmean, winstd):
        v = numpy.copy(sig)

        v *= yes_mul
        v += yes_add

        v = vproblt(v, winmean, winstd)

        #v *= yes_mul
        #v += yes_add

        v = numpy.minimum(v, 1.0)
        v = numpy.maximum(v, 0.0)

        return v

    def probno(self, sig, losemean, losestd):
        v = numpy.copy(sig)

        v *= no_mul
        v += no_add

        v = vprobgt(v, losemean, losestd)

        #v *= no_mul
        #v += no_add

        v = numpy.minimum(v, 1.0)
        v = numpy.maximum(v, 0.0)

        return v

    # m79 has been de-gray coded (including the three Costas arrays).
    # returns 174 log-likelihood values for whether each
    # bit is a zero.
    # really returns [ loglikelihood, P(0) ]
    def loglikelihood(self, m79):

        if True:
            # winmean, winstd, losemean, losestd
            [ wm, ws, lm, ls ] = self.softstats(m79)

        if False:
            [ winners, losers ] = self.winlose(m79)

            wm = numpy.mean(winners)
            ws = numpy.std(winners)

            # separately for each of 3 bits in a symbol.
            if True:
                lm = numpy.mean(losers)
                ls = numpy.std(losers)
            else:
                losers3 = [ losers[0::3], losers[1::3], losers[2::3] ]
                lma = [ ]
                lsa = [ ]
                for ll in losers3:
                    lma.append(numpy.mean(ll))
                    lsa.append(numpy.std(ll))
                lm = numpy.zeros(174)
                lm[0::3] = lma[0]
                lm[1::3] = lma[1]
                lm[2::3] = lma[2]
                ls = numpy.zeros(174)
                ls[0::3] = lsa[0]
                ls[1::3] = lsa[1]
                ls[2::3] = lsa[2]

        if wm < 0.000001:
            # this happens when we look again at a signal that
            # has been subtracted by zeroing FFTCache bins.
            return None

        # drop the three 7-symbol Costas arrays.
        m58 = numpy.concatenate( [ m79[7:36], m79[43:72] ] )

        zo174 = self.zeroone(m58)

        e0 = zo174[:,0]
        e1 = zo174[:,1]

        # start with measured a-priori bit probabilities,
        # one for each of the 174 bit positions.
        # i.e. P(one) and P(zero)
        if use_apriori:
            pone = apriori174
        else:
            pone = null_apriori174
        pone = pone[0:len(e0)]
        pzero = 1.0 - pone

        # Bayes says:
        #                P(e0 | zero) P(zero)
        # P(zero | e0) = --------------------
        #                       P(e0)

        if False:
            # this works well.
            # it uses CDF rather than PDF.

            # a = P(zero)P(e0|zero)P(e1|zero)
            a = pzero * vproblt(e0, wm, ws) * vprobgt(e1, lm, ls)

            # b = P(one)P(e0|one)P(e1|one)
            b = pone * vprobgt(e0, lm, ls) * vproblt(e1, wm, ws)

            # Bayes combining rule normalization from:
            # http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf

            denom = a + b

            p0 = numpy.divide(a, denom, out=numpy.repeat(0.5, len(a)), where=denom>0)

        if True:
            # a = P(zero)P(e0|zero)P(e1|zero)
            a = pzero * self.probyes(e0, wm, ws) * self.probno(e1, lm, ls)

            # b = P(one)P(e0|one)P(e1|one)
            b = pone * self.probno(e0, lm, ls) * self.probyes(e1, wm, ws)

            # Bayes combining rule normalization from:
            # http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf

            denom = a + b

            p0 = numpy.divide(a, denom, out=numpy.repeat(0.5, len(a)), where=denom>0)

        if False:
            # a = P(zero)P(e0|zero)P(e1|zero)
            a = pzero * numpy.minimum(1.0, 2.0*vproblt(e0, wm, ws)) * numpy.minimum(1.0, 2.0*vprobgt(e1, lm, ls))

            # b = P(one)P(e0|one)P(e1|one)
            b = pone * numpy.minimum(1.0, 2.0*vprobgt(e0, lm, ls)) * numpy.minimum(1.0, 2.0*vproblt(e1, wm, ws))

            # Bayes combining rule normalization from:
            # http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf

            denom = a + b

            p0 = numpy.divide(a, denom, out=numpy.repeat(0.5, len(a)), where=denom>0)

        if False:
            # this is totally broken.

            # a = P(zero)P(e0|zero)P(e1|zero)
            a = pzero * vproblt(e0, wm, ws) * vprobgt(e1, lm, ls)

            # b = P(one)P(e0|one)P(e1|one)
            b = pone * vprobgt(e0, lm, ls) * vproblt(e1, wm, ws)

            # denominator = P(e0)P(e1)
            denom_e0 = 0.5 * (vproblt(e0, wm, ws) + vproblt(e0, lm, ls))
            denom_e1 = 0.5 * (vprobgt(e1, lm, ls) + vprobgt(e1, wm, ws))
            denom = denom_e0 * denom_e1

            p0 = numpy.divide(a, denom, out=numpy.repeat(0.5, len(a)), where=denom>0)

        if False:
            # Bayes rule for P(zero | strength of zero bin).
            # using PDF of winmean/winstd distribution.
            # norm.pdf(x) yields PDF, x is in units of std dev away from mean.
            # 266 (261 of 553)
            e0zero = scipy.stats.norm.pdf((e0 - winmean) / winstd)
            e0one = scipy.stats.norm.pdf((e0 - losemean) / losestd)
            numerator = e0zero * pzero
            denominator = e0zero*pzero + e0one*pone
            p0 = numpy.divide(numerator, denominator, out=numpy.repeat(0.5, len(numerator)),
                              where=denominator>0)

        if False:
            # 485 (442 of 553)
            # Bayes rule with two pieces of evidence: look at
            # strengths of both the tone for bit=0 (e0) and
            # the tone for bit=1 (e1).
            #
            # the combining rule is straightforward if you assume
            # independence, e.g. P(e0 and e1) = P(e0)P(e1):
            #
            #                       P(e0 | zero) P(e1 | zero) P(zero)
            # P(zero | e0 and e1) = ----------------------------------
            #                                P(e0) P(e1)
            #
            # however, e0 and e1 certainly aren't independent: if one
            # is high, the other is very likely to be low, since they
            # represent FSK tones.
            
            e0zero = scipy.stats.norm.pdf((e0 - winmean) / winstd)
            e0one = scipy.stats.norm.pdf((e0 - losemean) / losestd)

            e1zero = scipy.stats.norm.pdf((e1 - losemean) / losestd)
            e1one = scipy.stats.norm.pdf((e1 - winmean) / winstd)

            numerator = e0zero * e1zero * pzero

            denominator = (e0zero*pzero + e0one*pone) * (e1zero*pzero + e1one*pone)

            p0 = numpy.divide(numerator, denominator, out=numpy.repeat(0.5, len(numerator)),
                              where=denominator>0)

        if False:
            # score 507 (462 of 553)
            # 
            # this is the party line for using bayes with two FSK tones.
            #
            # Bayes combining rule from:
            # http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf
            # for numerator, assumes conditional independence of e0 and e1.
            # for denominator, conditional independence and normalization.
            # conditional independence is probably an OK assumption:
            # P(e1|zero) probably doesn't depend on value of e1, since
            # the assumption of zero puts e1 in the "lose" distribution.
            
            # P(e0 | zero)
            e0zero = scipy.stats.norm.pdf((e0 - winmean) / winstd)

            # P(e0 | one)
            e0one = scipy.stats.norm.pdf((e0 - losemean) / losestd)

            # P(e1 | zero)
            e1zero = scipy.stats.norm.pdf((e1 - losemean) / losestd)

            # P(e1 | one)
            e1one = scipy.stats.norm.pdf((e1 - winmean) / winstd)

            zero_numerator = e0zero * e1zero * pzero
            one_numerator = e0one * e1one * pone

            denominator = zero_numerator + one_numerator

            p0 = numpy.divide(zero_numerator, denominator, out=numpy.repeat(0.5, len(zero_numerator)),
                              where=denominator>0)

        if False:
            # 56 of 147
            # from build_prob() on ft8-n11/big.txt
            # maps metric to probability of the higher
            # of e0/e1 being correct.
            probs = numpy.array([ 
                0.50, 0.53, 0.57, 0.61, 0.65, 0.68, 0.70, 0.72, 0.73, 0.74,
                0.74, 0.74, 0.74, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                0.74, 0.74, 0.74, 0.73, 0.72, 0.72, 0.72, 0.73, 0.73, 0.74,
                0.76, 0.77, 0.79, 0.80, 0.81, 0.83, 0.84, 0.85, 0.87, 0.87,
                0.89, 0.89, 0.89, 0.90, 0.91, 0.92, 0.91, 0.92, 0.93, 0.94,
                0.93, 0.93, 0.95, 0.98, 0.98, 0.99 ])
            
            mhigh = numpy.maximum(e0, e1)
            mlow = numpy.minimum(e0, e1)
            metric = mhigh / mlow
            metric = numpy.log(metric)
            metric = metric * 10.0
            metric = numpy.round(metric)
            metric = metric.astype(numpy.int32)
            metric = numpy.minimum(metric, len(probs)-1)
            px = probs[metric]

            p0 = numpy.where(e0 > e1, px, 1.0 - px)

        # log likelihood of t0 being the correct symbol.
        # ll0 = log(p0 / (1 - p0))
        # log(148) = 4.99
        dd = numpy.divide(p0, (1.0 - p0), out=numpy.repeat(148.0, len(p0)), where=p0<0.99)
        ll = numpy.log(dd, out=numpy.repeat(-5.0, len(dd)), where=dd>0)

        ll = numpy.maximum(ll, -5.0)
        ll = numpy.minimum(ll, 5.0)

        return [ ll, p0 ]

    # are the 91 bits after LDPC ok?
    # in particular, is the CRC ok?
    # ldpc_ok should be 83, i.e. all LDPC parity checked OK.
    def check_ldpc(self, a91, ldpc_ok):
        if numpy.all(a91==0):
            return False

        if crc_and_83:
            return (ldpc_ok >= ldpc_thresh) and check_crc(a91)
        else:
            return (ldpc_ok >= ldpc_thresh) or check_crc(a91)


    # m79 is 79 8-bucket mini FFTs, for 8-FSK demodulation.
    # m79 holds complex FFT results (with phase).
    # m79[0..79][0..8]
    # returns None or a Decode.
    # offset is just for dec and debugging.
    def process2(self, m79complex, hz, offset, pass_, down_hz):
        if len(m79complex) < 79:
            return None

        if False:
            # this actually works!
            # the point is that it's cheap,
            # and might give me a way to shift
            # frequency to center the signal
            # exactly in the FFT bins.
            for i in range(0, len(m79complex)):
                aaa = m79complex[i]
                bbb = numpy.fft.ifft(aaa)
                ccc = numpy.fft.fft(bbb)
                m79complex[i] = ccc

        snr = self.snr(abs(m79complex))

        m79 = abs(m79complex)

        if snr_overlap >= 0:
            # convert to S/N. this seems to help;
            # perhaps it makes successive symbols more comparable so
            # that softstats() works better.
            m79a = self.convert_to_snr(m79)
        else:
            m79a = numpy.copy(m79)
        dec = self.process2a(m79a, hz, offset, snr, False, pass_, down_hz)
        if dec != None:
            return dec

        if osd_no_snr and osd_depth >= 0:
            # but on the other hand some signals are wrecked by
            # converting to SNR.
            # in many cases these are saved by OSD alone, not LDPC.
            dec = self.process2a(m79, hz, offset, snr, True, pass_, down_hz)
            if dec != None:
                return dec

        return None

    # convert to S/N. the benefit is probably that it
    # makes the statistics from softstats more meaningful,
    # since it probably makes the input more stationary.
    # the point of the statistics is so that LDPC can decide
    # which bits are most likely to be correct, so we
    # need to be able to compare strength of different
    # bits, which SNR seems to help.
    def convert_to_snr(self, m79):
        sort79 = numpy.sort(m79, 1)

        # for each symbol time, the maximum signal level of the 8 bins.
        maxm = sort79[:,7]

        # for each symbol time, the median signal level,
        # as a measure of noise.
        # this works better than e.g. min or mean.
        mm = (sort79[:,3] + sort79[:,4]) / 2.0

        #mm = (sort79[:,0] + sort79[:,4]) / 2.0 # 1171
        #mm = (sort79[:,3] + sort79[:,4]) / 1.0 # 1175
        #mm = sort79[:,6] # 1161
        #mm = (sort79[:,5] + sort79[:,6]) # 1167
        #mm = (sort79[:,0] + sort79[:,6]) # 1156
        #mm = (sort79[:,4] + sort79[:,5]) # 1169
        # mm = (sort79[:,3] + sort79[:,4] + sort79[:,5]) / 3.0 # 1159

        # derive a local noise level for each symbol time by running a
        # smallish window over mm.

        winlen = 1 + 2*snr_overlap

        if snr_wintype == "bartlett":
            win = numpy.bartlett(winlen)
        if snr_wintype == "blackman":
            win = numpy.blackman(winlen)
        if snr_wintype == "hamming":
            win = numpy.hamming(winlen)
        if snr_wintype == "hanning":
            win = numpy.hanning(winlen)
        if snr_wintype == "kaiser":
            win = numpy.kaiser(winlen, 14)
        if snr_wintype == "boxcar":
            win = scipy.signal.boxcar(winlen)
        if snr_wintype == "cosine":
            win = scipy.signal.cosine(winlen)
        if snr_wintype == "tukey":
            win = scipy.signal.tukey(winlen)
        if snr_wintype == "triang":
            win = scipy.signal.triang(winlen)
        if snr_wintype == "flattop":
            win = scipy.signal.flattop(winlen)
        if snr_wintype == "gaussian":
            win = scipy.signal.gaussian(winlen, winlen / 6.0)
        if snr_wintype == "nuttall":
            win = scipy.signal.nuttall(winlen)
        if snr_wintype == "parzen":
            win = scipy.signal.parzen(winlen)
        if snr_wintype == "dpss":
            win = scipy.signal.windows.dpss(winlen, 3)
                
        #win = win / numpy.sum(win)
        noise = numpy.convolve(mm, win)
        noise = noise[winlen//2:]
        if len(noise) > len(mm):
            noise = noise[0:len(mm)]

        noisefloor = maxm / 10000.0

        noise = numpy.select( [ noise > noisefloor ], [ noise ], default=noisefloor)
        noise8 = numpy.stack([noise]*8, 1)
        m79a = numpy.divide(m79, noise8)
        return m79a

    # return a Decode or None
    def try_decode(self, ll174, hz, offset, snr, do_osd):
        self.ldpc_calls += 1 # for -cpubench

        # decode LDPC(174,91)
        [ ldpc_ok, a91 ] = ldpc_decode(ll174, ldpc_iters)

        if self.check_ldpc(a91, ldpc_ok):
            # fast path for success.
            dec = self.process3(hz, offset, snr, a91)
            return dec

        if osd_depth >= 0 and do_osd:
            [ a91, osd_score ] = osd_decode(ll174, osd_depth)
            if len(a91) > 0 and check_crc(a91) and numpy.all(a91==0) == False:
                dec = self.process3(hz, offset, snr, a91)
                if dec != None:
                    return dec

        return None

    def process2a(self, m79, hz, offset, snr, just_osd, pass_, down_hz):

        # un-gray-code tone strengths (before LDPC).
        revmap = numpy.array([ 0, 1, 3, 2, 6, 4, 5, 7 ], dtype=numpy.int32)
        m79a = numpy.zeros([len(m79),8])
        for i in range(0, len(m79)):
            m79a[i][revmap] = m79[i]
        m79 = m79a

        [ ll174, _ ] = self.loglikelihood(m79)

        maxll = numpy.max(abs(ll174))

        dec = self.try_decode(ll174, hz, offset, snr, True)
        if dec != None:
            return dec
        if just_osd:
            return None

        if pass_ > 0 or pass0_hints:
            for hint in self.hints:
                if hint.hz != None:
                    hhz = hint.hz - down_hz
                    if abs(hhz - hz) > hint_tol:
                        continue
                llx = numpy.copy(ll174)
                if len(hint.packed1) > 0:
                    llx[0:28] = hint.packed1 * maxll
                if len(hint.packed2) > 0:
                    llx[29:29+28] = hint.packed2 * maxll
                if len(hint.packed1) == 0 or len(hint.packed2) == 0:
                    # force i3=1
                    llx[74] = maxll # 0
                    llx[75] = maxll # 0
                    llx[76] = -maxll # 1
                    # force both R to 0
                    llx[28] = maxll # 0
                    llx[57] = maxll # 0
                dec = self.try_decode(llx, hz, offset, snr, osd_hints)
                if dec != None:
                    dec.hint = hint
                    # sys.stdout.write("hint %s -> %s\n" % (hint, dec.msg))
                    return dec

        return None

    # unpack new 77-bit format.
    # a91 includes the CRC14, though it's already been checked.
    # CRC has already been checked.
    def process3(self, hz, offset, snr, a91):
        dec = self.unpack(a91)
        if dec == None:
            return None
        dec.hza = [ hz, hz ]
        dec.snr = snr
        dec.bits77 = a91[0:77]
        snd = FT8Send()
        dec.symbols = snd.make_symbols(dec.bits77) # needed for subtraction
        dec.decode_time = time.time()
        dec.start = offset
        if False:
            self.save_apriori(dec)
        return dec

    # remember codeword bits in order to calculate
    # a-priori bit value probabilities, for apriori174[].
    # python bprob.py < ft8bits
    def save_apriori(self, dec):
        # re-encode
        n91 = numpy.zeros(91, dtype=numpy.int32)
        n91[0:77] = dec.bits77
        cksum = crc(n91[0:82], crc14poly)
        n91[-14:] = cksum
        n174 = ldpc_encode(n91)

        if False:
            # verify that encoding is OK
            two = numpy.array([ 4.6, -4.6 ])
            ll174 = two[n174]
            assert ldpc_decode(ll174)[0] == 83
            assert len(ldpc_decode(ll174)[1]) == 91
            assert numpy.array_equal(ldpc_decode(ll174)[1], n91)
            assert check_crc(n174[0:91])
            assert self.unpack(n174[0:91]).msg == dec.msg

        if False:
            for i in range(0, 10):
                sys.stdout.write("%4.1f " % (ll174[i]))
            sys.stdout.write("\n")
            for i in range(0, 10):
                sys.stdout.write("%4.1f " % (n174[i]))
            sys.stdout.write("\n")

        f = open("ft8bits", "a")
        for i in range(0, 174):
            f.write("%d" % (n174[i]))
        f.write("\n")
        f.close()

    # unpack one of the two 28-bit call fields.
    # new 77-bit scheme.
    # from packjt77.f90
    def unpackcall(self, x):

        c1 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        c2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        c3 = "0123456789"
        c4 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if x == 0:
            return "DE"
        if x == 1:
            return "QRZ"
        if x == 2:
            return "CQ"
        if x <= 1002:
            # CQ_nnn
            return "CQ_" + str(int(x-3))
        if x <= 532443:
            # CQ_aaaa
            x -= 1003
            ci1 = x // (27*27*27)
            x %= 27*27*27
            ci2 = x // (27*27)
            x %= 27*27
            ci3 = x // 27
            x %= 27
            ci4 = x
            aaaa = c4[ci1] + c4[ci2] + c4[ci3] + c4[ci4]
            while len(aaaa) > 0 and aaaa[0] == " ":
                aaaa = aaaa[1:]
            return "CQ_" + aaaa
            

        if x < NTOKENS:
            return "???"

        x -= NTOKENS

        if x < MAX22:
            # x is a 22-bit hash of a (hopefully) previously seen call.
            if x in self.hashes22:
                return "<%s>" % (self.hashes22[x])
            else:
                return "<...>"

        x -= MAX22

        # space, digit, alpha
        # digit, alpha
        # digit
        # space, alpha
        # space, alpha
        # space, alpha

        a = [ 0, 0, 0, 0, 0, 0 ]
        a[5] = c4[x % 27]
        x = int(x / 27)
        a[4] = c4[x % 27]
        x = int(x / 27)
        a[3] = c4[x % 27]
        x = int(x / 27)
        a[2] = c3[x % 10]
        x = int(x / 10)
        a[1] = c2[x % 36]
        x = int(x / 36)
        a[0] = c1[x]
        return ''.join(a)

    # unpack a 15-bit grid square &c.
    # 77-bit version, from inspection of packjt77.f90.
    # ir is the bit after the two 28+1-bit callee/caller.
    # i3 is the message type, usually 1.
    def unpackgrid(self, ng, ir, i3):
        assert i3 == 1

        if ng < NGBASE:
            # maidenhead grid system:
            #   latitude from south pole to north pole.
            #   longitude eastward from anti-meridian.
            #   first: 20 degrees longitude.
            #   second: 10 degrees latitude.
            #   third: 2 degrees longitude.
            #   fourth: 1 degree latitude.
            # so there are 18*18*10*10 possibilities.
            x1 = ng // (18 * 10 * 10)
            ng %= 18 * 10 * 10
            x2 = ng // (10 * 10)
            ng %= 10 * 10
            x3 = ng // 10
            ng %= 10
            x4 = ng
            gggg = "%c%c%c%c" % (ord('A') + x1,
                                 ord('A') + x2,
                                 ord('0') + x3,
                                 ord('0') + x4)
            return gggg
        
        ng -= NGBASE

        if ng == 1:
            return "   " # ???
        if ng == 2:
            return "RRR "
        if ng == 3:
            return "RR73"
        if ng == 4:
            return "73  "

        db = ng - 35
        if db >= 0:
            dbs = "+" + ("%02d" % (db))
        else:
            dbs = "-" + ("%02d" % (0-db))
        if ir == 1:
            dbs = "R" + dbs
        return dbs

    # unpack a 77-bit new FT8 message.
    # from inspection of packjt77.f90
    # returns a Decode, or None.
    def unpack(self, a77):
        i3 = un(a77[74:74+3])
        n3 = un(a77[71:71+3])

        if i3 == 0 and n3 == 0:
            # free text
            return self.unpacktext(a77)

        if i3 == 4:
            # a call that doesn't fit in 28 bits.
            # 12 bits: hash of a previous call
            # 58 bits: 11 characters
            # 1 bit: swap
            # 2 bits: 1 RRR, 2 RR73, 3 73
            # 1 bit: 1 means CQ

            # 38 possible characters:
            chars = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"

            n58 = un(a77[12:12+58])
            call = ""
            for i in range(0, 11):
                call = call + chars[n58 % 38]
                n58 = n58 // 38
            # reverse
            call = call[::-1]
            call = call.strip()

            if un(a77[73:73+1]) == 1:
                msg = "CQ " + call
                dec = Decode(msg)
                dec.hashcalls = [ call ] # please hash this and remember
                return dec

            x12 = un(a77[0:12])
            if x12 in self.hashes12:
                ocall = "<" + self.hashes12[x12] + ">"
            else:
                ocall = "<...>"
            swap = un(a77[70:70+1])
            if swap:
                msg = "%s %s" % (call, ocall)
            else:
                msg = "%s %s" % (ocall, call)
            suffix = un(a77[71:71+2])
            msg += [ "", " RRR", " RR73", " 73" ][suffix]
            dec = Decode(msg)
            dec.hashcalls = [ call ]

            return dec

        if i3 == 3:
            # ARRL RTTY Round-Up

            #  1 TU
            # 28 call1
            # 28 call2
            #  1 R
            #  3 RST 529 to 599
            # 13 state/province/serialnumber

            i = 0
            tu = a77[i]
            i += 1
            call1 = un(a77[i:i+28])
            i += 28
            call2 = un(a77[i:i+28])
            i += 28
            r = a77[i]
            i += 1
            rst = un(a77[i:i+3])
            i += 3
            serial = un(a77[i:i+13])
            i += 13

            call1text = self.unpackcall(call1).strip()
            call2text = self.unpackcall(call2).strip()
            rst = 529 + 10*rst

            statei = serial - 8001
            if serial > 8000 and statei < len(ru_states):
                serialstr = ru_states[statei]
            else:
                serialstr = "%04d" % (serial)

            msg = ""
            if tu == 1:
                msg += "TU; "
            msg += "%s %s " % (call1text, call2text)
            if r == 1:
                msg += "R "
            msg += "%d %s" % (rst, serialstr)

            dec = Decode(msg)

            if len(call1text) > 2 and re.search(r'^[0-9A-Z]*$', call1text) != None:
                dec.hashcalls.append(call1text)
            if len(call2text) > 2 and re.search(r'^[0-9A-Z]*$', call2text) != None:
                dec.hashcalls.append(call2text)

            return dec

        if i3 == 0 and n3 == 1:
            # 0.1   K1ABC RR73; W9XYZ <KH1/KH7Z> -12   28 28 10 5       71   DXpedition Mode

            i = 0
            call1 = un(a77[i:i+28])
            i += 28
            call2 = un(a77[i:i+28])
            i += 28
            x10 = un(a77[i:i+10]) # 10-bit hash of sender's callsign
            i += 10
            x5 = un(a77[i:i+5]) # report
            i += 5

            call1text = self.unpackcall(call1).strip()
            call2text = self.unpackcall(call2).strip()
            call3text = "<...>"
            if x10 in self.hashes10:
                call3text = "<" + self.hashes10[x10] + ">"

            msg = "%s RR73; %s %s %d" % (call1text,
                                         call2text,
                                         call3text,
                                         2*x5 - 30)
            dec = Decode(msg)

            return dec

        if i3 != 1:
            if False and (i3 == 0 and n3 in [ 1, 3, 4, 6 ] ):
                print("unknown i3.n3 %s.%s" % (i3, n3))
                #print(a77)
            return None

        # type 1:
        # 28 call1
        # 1 P/R
        # 28 call2
        # 1 P/R
        # 1 ???
        # 15 grid
        # 3 type
        # 14 CRC

        i = 0
        call1 = un(a77[i:i+28])
        i += 28
        rover1 = a77[i]
        i += 1
        call2 = un(a77[i:i+28])
        i += 28
        rover2 = a77[i]
        i += 1
        ir = a77[i]
        i += 1
        grid = un(a77[i:i+15])
        i += 15
        i3 = un(a77[i:i+3])
        i += 3
        assert i == 77

        call1text = self.unpackcall(call1).strip()
        call2text = self.unpackcall(call2).strip()
        gridtext = self.unpackgrid(grid, ir, i3).strip()

        msg = "%s %s %s" % (call1text, call2text, gridtext)

        msg = re.sub(r'^CQ_', 'CQ ', msg)

        dec = Decode(msg)

        if len(call2text) > 2 and re.search(r'^[0-9A-Z]*$', call2text) != None:
            dec.hashcalls.append(call2text)

        return dec

    # i3=0 n3=0, free text, unpack 71 bits,
    # 13 characters, each one of 42 choices.
    # returns a Decode, or None.
    # packjt77.f90
    def unpacktext(self, a71):
        # the 42 possible characters.
        cc = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"

        a71 = numpy.copy(a71[0:71])

        msg = ""
        x = un(a71[0:71])
        for i in range(0, 13):
            ch = cc[x % 42]
            msg = msg + ch
            x = x // 42

        # reverse
        msg = msg[::-1]

        return Decode(msg)

very_first_time = True
profiling = False

class FT8Send:
    def __init__(self):
        pass

    # returns a 28-bit number.
    # 28-bit number's high bits correspond to first call sign character.
    # new FT8.
    # returns -1 if it doesn't know how to encode the call.
    def packcall(self, call, hashes22=None):
        c1 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        c2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        c3 = "0123456789"
        c4 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        call = call.strip()
        call = call.upper()

        if call == "DE":
            return 0
        if call == "QRZ":
            return 1
        if call == "CQ":
            return 2

        m = re.search(r'^CQ_([A-Z]*)$', call)
        if m != None:
            # CQ_aaaa
            aaaa = m.group(1)
            x = 0
            for i in range(0, 4):
                x *= 27
                if len(aaaa) > i:
                    x += c4.find(aaaa[i])
            x += 1003
            return x

        if len(call) > 2 and len(call) < 6 and not call[2].isdigit():
            call = " " + call
        while len(call) < 6:
            call = call + " "

        if re.search(r'^[A-Z0-9 ][A-Z0-9 ][0-9][A-Z ][A-Z ][A-Z ]$', call) == None:
            # previously seen complex call (e.g. VE2/UT3UA) and hash was saved?
            call = call.strip()
            h = ihashcall(call, 22)
            if hashes22 != None and hashes22.get(h) == call:
                assert h > 0 and h < MAX22
                return NTOKENS + h
            return -1

        # c1 c2 c3 c4 c4 c4

        x = 0
        x += c1.find(call[0])

        x *= len(c2)
        x += c2.find(call[1])

        x *= len(c3)
        x += c3.find(call[2])

        x *= len(c4)
        x += c4.find(call[3])

        x *= len(c4)
        x += c4.find(call[4])

        x *= len(c4)
        x += c4.find(call[5])

        return x + MAX22 + NTOKENS

    # returns 15-bit number.
    # g is maidenhead grid, or signal strength, or 73.
    # i3 is intended new FT8 format, usually 1.
    # returns (15 bits, ir)
    # ir is a bit in the new FT8 format, here indicating
    # the R in R-14 &c.
    def packgrid(self, g, i3):
        assert i3 == 1

        g = g.strip()
        g = g.upper()

        if g == "RRR":
            return (NGBASE + 2, 0)
        if g == "RR73":
            return (NGBASE + 3, 0)
        if g == "73":
            return (NGBASE + 4, 0)

        m = re.match(r'^(R?)([+-])([0-9][0-9])$', g)
        if m != None:
            x = int(m.group(3))
            if m.group(2) == "-":
                x = 0 - x
            x = NGBASE + x + 35
            if m.group(1) == "R":
                ir = 1
            else:
                ir = 0
            return (x, ir)

        if re.match(r'^[A-R][A-R][0-9][0-9]$', g) == None:
            return (-1, 0)

        x1 = ord(g[0]) - ord('A')
        x2 = ord(g[1]) - ord('A')
        x3 = ord(g[2]) - ord('0')
        x4 = ord(g[3]) - ord('0')

        x = 0
        x += x1 * 18 * 10 * 10
        x += x2 * 10 * 10
        x += x3 * 10
        x += x4

        return (x, 0)

    # returns 77 bits.
    # on failure, returns an empty array.
    def pack(self, msg, i3, hashes22=None):
        if i3 == 1:
            return self.pack_type1(msg, i3, hashes22)
        if i3 == 3:
            return self.pack_type3(msg, i3, hashes22)
        sys.stderr.write("ft8.pack() unknown i3 %d\n" % (i3))
        return numpy.array([], dtype=numpy.int32)

    def pack_type1(self, msg, i3, hashes22):
        msg = msg.strip()
        msg = re.sub(r'  *', ' ', msg)
        msg = re.sub(r'^CQ DX ', 'CQ_DX ', msg)

        # XXX what about /R and /P, i.e. the 29th bit?

        # try CALL CALL GRID
        a = msg.split(' ')
        if len(a) == 3:
            nc1 = self.packcall(a[0], hashes22)
            nc2 = self.packcall(a[1], hashes22)
            (g, ir) = self.packgrid(a[2], i3)
            if nc1 >= 0 and nc2 >= 0 and g >= 0:
                a = numpy.concatenate([
                    bv(nc1, 28),
                    [ 0 ], # XXX
                    bv(nc2, 28),
                    [ 0 ], # XXX
                    [ ir ],
                    bv(g, 15),
                    bv(i3, 3)
                    ])
                return a

        sys.stderr.write("FT8Send.pack_type1(%s) -- cannot parse\n" % (msg))
        return numpy.array([], dtype=numpy.int32)

    def pack_type3(self, msg, i3, hashes22):
        msg = msg.strip()
        msg = re.sub(r'  *', ' ', msg)

        # can only handle e.g. N4TTE AB1HL 539 MA

        m = re.match(r'^([A-Z0-9]+) ([A-Z0-9]+) (5[2-9]9) ([A-Z]+)$', msg)
        if m != None:
            call1 = m.group(1)
            call2 = m.group(2)
            rst = m.group(3)
            state = m.group(4)

            nc1 = self.packcall(call1, hashes22)
            nc2 = self.packcall(call2, hashes22)

            if nc1 >= 0 and nc2 >= 0 and state in ru_states:
                statei = ru_states.index(state)
                staten = 8001 + statei
                rstn = ord(rst[1]) - ord('2')
                a = numpy.concatenate([
                    [ 0 ], # TU
                    bv(nc1, 28),
                    bv(nc2, 28),
                    [ 0 ], # R
                    bv(rstn, 3),
                    bv(staten, 13),
                    bv(i3, 3)])
                assert len(a) == 77
                return a

        sys.stderr.write("FT8Send.pack_type3(%s) -- cannot parse\n" % (msg))
        return numpy.array([], dtype=numpy.int32)

    def testpack(self):
        r = FT8()
        for g in [ "FN42", "-22", "R-01", "RR73", "RRR", "73", "AA00", "RR99", "+05", "R+11", "R-11" ]:
            (pg, ir) = self.packgrid(g, 1)
            upg = r.unpackgrid(pg, ir, 1)
            if g != upg.strip():
                print("packgrid oops, wanted %s, got %s" % (g, upg.strip()))

        # ordinary calls.
        for call in [ "AB1HL", "K1JT", "M0TRJ", "KK4BMV", "2E0CIN", "HF9D",
                      "6Y4K", "D4Z", "8P6DR", "ZS2I", "3D2RJ",
                      "WB3D", "S59GCD", "T77C", "4Z5AD", "A45XR", "OJ0V",
                      "6Y6N", "S57V", "3Z0R" ]:
            pc = self.packcall(call)
            upc = r.unpackcall(pc)
            if call != upc.strip():
                print("packcall oops %s %d %s" % (call, pc, upc))

        # calls that need to be sent as hashes.
        r.hashes22 = { }
        for call in [ "3XY4D", "3DA0AY", "P4/K3DMG", "JH0UUY/1", "VE2/UT3UA",
                      "LZ1354PM" ]:
            r.hashes22[ihashcall(call, 22)] = call
            pc = self.packcall(call, r.hashes22)
            upc = r.unpackcall(pc)
            if "<" + call + ">" != upc.strip():
                print("hash packcall oops %s %d %s" % (call, pc, upc))

        for msg in [ "AB1HL K1JT FN42", "CQ DX CO3HMR EL82", "KD6HWI PY7VI R-12",
                     "CQ N5OSK EM25", "PD9BG KG7EZ RRR",
                     "W1JET KE0HQZ 73", "WB3D OM4SX -16", "WA3ETR IZ2QGB RR73",
                     "K3DMG AB1HL +06",
                     "P4/K3DMG AB1HL +06", "3XY4D AB1HL FN42", "JH0UUY/1 AB1HL RR73" ]:
            pm = self.pack(msg, 1, r.hashes22)
            if len(pm) == 0:
                print("pack() failed for %s" % (msg))
                continue
            dec = r.unpack(pm)
            if dec == None:
                print("unpack() failed for %s" % (msg))
                continue
            upm = dec.msg
            upm = upm.replace('<', ' ')
            upm = upm.replace('>', ' ')
            upm = re.sub(r'  *', ' ', upm)
            if msg != upm.strip():
                print("pack oops %s %s %s" % (msg, pm, upm))

    # bits77 is the result of pack().
    # returns an array of 79 symbols 0..8, ready for FSK.
    def make_symbols(self, bits77):
        assert len(bits77) == 77
        cksum = crc(numpy.append(bits77, numpy.zeros(5, dtype=numpy.int32)),
                    crc14poly)
        a91 = numpy.zeros(91, dtype=numpy.int32)
        a91[0:77] = bits77
        a91[77:91] = cksum

        a174 = ldpc_encode(a91)

        a174 = gray_code(a174)

        # turn array of 174 bits into 58 3-bit symbols,
        # most significant bit first.
        dsymbols = numpy.zeros(58, dtype=numpy.int32)
        for i in range(0, 58):
            ii = i * 3
            dsymbols[i] = (a174[ii+0] << 2) | (a174[ii+1]<<1) | (a174[ii+2]<<0)

        # insert three 7-symbol Costas arrays.
        costas = numpy.array(costas_symbols, dtype=numpy.int32)
        symbols = numpy.zeros(79, dtype=numpy.int32)
        symbols[0:7] = costas
        symbols[7:36] = dsymbols[0:29]
        symbols[36:43] = costas
        symbols[43:72] = dsymbols[29:]
        symbols[72:] = costas

        return symbols

    # a77 is 77 bits, the result of pack().
    # tone is Hz of lowest tone.
    # returns an array of audio samples.
    def tones(self, a77, tone, rate):
        symbols = self.make_symbols(a77)

        samples_per_symbol = int(round(rate * (1920 / 12000.0)))
        samples = weakutil.fsk(symbols, [tone, tone], 6.25, rate, samples_per_symbol)

        return samples

    def testsend(self):
        random.seed(0)
        rate = 12000

        a77 = self.pack("G3LTF DL9KR JO40", 1)
        x1 = self.tones(a77, 1000, rate)
        x1 = numpy.concatenate(([0]*1,  x1, [0]*(8192-1) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x1 = x1 * rv

        a77 = self.pack("RA3Y VE3NLS 73", 1)
        x2 = self.tones(a77, 1050, rate)
        x2 = numpy.concatenate(([0]*4096,  x2, [0]*(8192-4096) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x2 = x2 * rv

        a77 = self.pack("CQ DL7ACA JO40", 1)
        x3 = self.tones(a77, 1100, rate)
        x3 = numpy.concatenate(([0]*5120,  x3, [0]*(8192-5120) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x3 = x3 * rv

        a77 = self.pack("VA3UG   F1HMR 73", 1)
        x4 = self.tones(a77, 1150, rate)
        x4 = numpy.concatenate(([0]*1,  x4, [0]*(8192-1) ))
        #rv = numpy.concatenate( [ [random.random()]*4096 for i in range(0, 128) ] )
        #x4 = x4 * rv

        x = 3*x1 + 2*x2 + 1.0*x3 + 0.5*x4

        x += numpy.random.rand(len(x)) * 1.0
        x *= 1000.0

        x = numpy.append([0]*(rate // 2), x)
        x = numpy.append(x, [0]*rate)

        r = FT8()
        r.verbose = True
        r.cardrate = rate
        r.process(x, 0)

if False:
    # nov 30 2018: it works (old FT8)
    # dec 4 2018: it works (new FT8)
    s = FT8Send()
    r = FT8()
    plain = [ ]
    for i in range(0, 91):
        plain.append(random.randint(0, 1))
    plain = numpy.array(plain, dtype=numpy.int32)

    cw = ldpc_encode(plain)

    # turn hard bits into 0.99 vs 0.01 log-likelihood,
    # log( P(0) / P(1) )
    # log base e.
    two = numpy.array([ 4.6, -4.6 ])
    ll174 = two[cw]

    d = ldpc_decode(ll174)

    assert d[0] == 83
    assert numpy.array_equal(d[1], plain)

    sys.exit(1)

if False:
    s = FT8Send()
    s.testpack()
    sys.exit(1)

if False:
    # nov 30 2018: prints
    # P0 -  2.8 1050.00 17040 0.34 -4 RA3Y   VE3NLS 73  
    # P0 -  2.9 1100.00 17580 0.43 -9 CQ DL7ACA JO40
    # P0 -  2.9 1000.00 15000 0.00 1  G3LTF DL9KR  JO40
    # P0 -  2.9 1150.00 15000 0.00 -8 VA3UG   F1HMR 73  
    # dec 4 2018: same
    s = FT8Send()
    s.testsend()
    sys.exit(1)

if False:
    # test decoding sequence on known correct symbols,
    # nov 30 2018: it works
    # from wsjt-x's ft8sim.
    # should yield K1ABC W9XYZ EN37
    r = FT8()

    # 79 3-bit 8-FSK symbols, including the Costas arrays.
    a79 = "2560413335544231617326364127543164332560413756060434371356756660051002662560413"
    b79 = [ int(x) for x in a79 ]
    assert len(b79) == 79

    # get rid of the three 7-symbol Costas arrays.
    a58 = b79[7:36] + b79[43:72]
    assert len(a58) == 58

    # turn 3-bits into 1-bits.
    # most-significant bit first.
    z = [ [ (x>>2)&1, (x>>1)&1, x&1 ] for x in a58 ]
    a174 = numpy.concatenate(z)
    assert len(a174) == 174

    if True:
        # flip some bits, for testing.
        a174[17] ^= 1
        a174[30] ^= 1
        a174[31] ^= 1
        a174[102] ^= 1

    # turn hard bits into 0.99 vs 0.01 log-likelihood,
    # log( P(0) / P(1) )
    # log base e.
    two = numpy.array([ 4.6, -4.6 ])
    ll174 = two[a174]

    # decode LDPC(174,87)
    a87 = ldpc_decode(ll174)

    # failure -> numpy.array([])
    assert a87[0] == 87
    assert len(a87[1]) == 87

    # CRC12
    # this mimics the way the sender computes the 12-bit checksum:
    c = crc(numpy.append(a87[1][0:72], numpy.zeros(4, dtype=numpy.int32)),
            crc12poly)
    assert numpy.array_equal(c, a87[1][-12:])

    # a87[1] is 72 bits of msg and 12 bits of CRC.
    # turn the 72 bits into twelve 6-bit numbers,
    # for compatibility with FT8 unpack().
    # MSB.
    a72 = a87[1][0:72]
    twelve = [ ]
    for i in range(0, 72, 6):
        a = a72[i:i+6]
        x = (a[0] * 32 +
             a[1] * 16 +
             a[2] *  8 +
             a[3] *  4 +
             a[4] *  2 +
             a[5] *  1)
        twelve.append(x)

    msg = r.unpack(twelve)
    assert msg == " K1ABC  W9XYZ EN37"

    sys.exit(0)

def usage():
    sys.stderr.write("Usage: ft8.py -card CARD CHAN\n")
    sys.stderr.write("       ft8.py -file fff [-chan xxx]\n")
    sys.stderr.write("       ft8.py -bench ft8files/xxx.txt\n")
    sys.stderr.write("       ft8.py -opt ft8files/xxx.txt\n")
    sys.stderr.write("       ft8.py -cpubench\n")
    # list sound cards
    weakaudio.usage()
    sys.exit(1)

if False:
    # check that we can decode a perfect signal. checking that
    # the highest-ranked hz/offset is the desired one is a
    # pretty good diagnostic.

    # nov 30 2018: works, prints CQ W1HZ BM12

    rate = 12000
    #hz = (107 * 6.25) + 0.4
    #hz = (random.random() * 2400) + 100
    #hz = 100 * 6.25 - 1.56
    hz = 100 * 6.25
    hz -= 0.78

    npad = 0.5
    #npad += (random.random() * 1.0) - 0.5
    npad = int(rate * npad)
    npad -= 120

    s = FT8Send()
    twelve = s.pack("CQ W1HZ BM12")

    if True:
        x1 = s.tones(twelve, hz, rate)

    if False:
        # 30 yields reported SNR of -19, and 100% decodes with abs().
        # 40 yields -21, and about 50% decodes with abs().
        x1 += 40.0 * (numpy.random.rand(len(x1)) - 0.5)

    if False:
        symbols = s.make_symbols(twelve)
        x1 = numpy.zeros(79 * 1920)
        bin0 = int(round(hz / 6.25))
        for i in range(0, 79):
            off = i * 1920
            bins = numpy.zeros(961, dtype=numpy.complex128)
            for ii in range(bin0-10, bin0+18):
                # random complex numbers with magnitude one.
                x = random.random() + 1j*random.random()
                x /= abs(x)
                bins[ii] = x
            x = 0 + 1j
            x /= abs(x) # same amplitude as garbage signals
            if (i % 4) != 0:
                x *= 1.35
            bins[bin0 + symbols[i]] = x
            block2 = weakutil.irfft(bins)
            x1[off:off+1920] = block2
        x1 = x1 / numpy.mean(abs(x1))
        weakutil.writewav(x1, "b.wav", 12000)

    x1 = numpy.append(numpy.random.rand(npad) - 0.5, x1)
    x1 = numpy.append(x1, numpy.random.rand(rate) - 0.5)

    r = FT8()
    r.cardrate = rate
    r.verbose = True
    r.process(x1, 0)

    msgs = r.get_msgs()
    if len(msgs) > 0:
        dec = msgs[0]

    sys.exit(0)

# hz_err in hz, off_err in fraction of a symbol time.
def test_subtract(hz_err, off_err):
    s = FT8Send()
    r = FT8()
    r.jrate = 3000
    r.jblock = 480
    bin_hz = r.jrate / float(r.jblock)

    if False:
        hz = (bin_hz * 100) + (bin_hz / 3.0)
        o = numpy.zeros(3)
        for phase in numpy.arange(0.0, 2*math.pi, 2*math.pi/10):
            tone = weakutil.costone(r.jrate, hz, r.jblock, phase)
            a = weakutil.rfft(tone)
            bin = int(round(hz / bin_hz))
            aa = numpy.angle(a[bin-1:bin+2])
            print(aa - o)
            o = aa

        sys.exit(0)

    base = numpy.zeros(15 * r.jrate)

    base += numpy.random.random(len(base)) - 0.5

    dec = Decode("")
    dec.start = 3111
    hz = (bin_hz * 100) + (bin_hz / 3.0)
    dec.bits77 = numpy.random.randint(0, 2, 77)
    tones = s.tones(dec.bits77, hz-hz_err, r.jrate)
    tones *= 3.111

    a = numpy.copy(base)
    off_err_samples = int(round(off_err * r.jblock))
    a[dec.start+off_err_samples:dec.start+off_err_samples+len(tones)] += tones

    if False:
        aa = r.subtract_v5(numpy.copy(a), dec, hz)
        weakutil.writewav(aa, "v5.wav", r.jrate)
        score = numpy.sum(numpy.square(aa - base)) / len(base)
        print("v5 %f" % (score))

    if False:
        aa = r.subtract_v6(numpy.copy(a), dec, hz)
        weakutil.writewav(aa, "v6.wav", r.jrate)
        score = numpy.sum(numpy.square(aa - base)) / len(base)
        print("v6 %f" % (score))

    if False:
        aa = r.subtract_v7(numpy.copy(a), dec, hz)
        weakutil.writewav(aa, "v7.wav", r.jrate)
        score = numpy.sum(numpy.square(aa - base)) / len(base)
        print("v7 %f" % (score))

    weakutil.writewav(base, "base.wav", r.jrate)
    weakutil.writewav(a, "a.wav", r.jrate)

if False:
    test_subtract(0, 0)
    test_subtract(0.11, 0)
    test_subtract(0, 0.123)
    test_subtract(0, -0.123)
    test_subtract(0.11, 0.123)
    test_subtract(0.11, -0.123)
    sys.exit(0)

def set_start_adj(wsjtfile):
    global start_adj

    # time apparently wrong on laptop when I had wsjt-x record these files.
    if "ft8-40" in wsjtfile:
        start_adj = 0.5
    if "ft8-20" in wsjtfile:
        start_adj = 0.5
    if "ft8files" in wsjtfile:
        start_adj = 0.8

def benchmark(wsjtfile, verbose):
    dir = os.path.dirname(wsjtfile)
    minutes = { } # keyed by hhmmss
    wsjtf = open(wsjtfile, "r")
    for line in wsjtf:
        # 161230 -15  0.2 1779 ~  CQ W8MSC EN82 !U.S.A.
        # 161230 -16  0.8 2352 ~  VE2FON KM4MDT R-09
        # 161245 -21  0.3  538 ~  K3OWX KG5AUW -03
        # 161245   2  0.1  955 ~  KJ1J NS9I -06
        line = re.sub(r'\xA0', ' ', line) # 0xA0 -> space
        line = re.sub(r'[\r\n]', '', line)
        m = re.match(r'^([0-9]{6}) +.*$', line)
        if m == None:
            print("oops: " + line)
            continue
        hhmmss = m.group(1)
        if not hhmmss in minutes:
            minutes[hhmmss] = ""
        minutes[hhmmss] += line + "\n"
    wsjtf.close()

    info = [ ]
    for hhmmss in sorted(minutes.keys()):
        ff = [ x for x in os.listdir(dir) if re.match('......_' + hhmmss + '.wav', x) != None ]
        if len(ff) == 1:
            filename = ff[0]
            info.append([ True, filename, minutes[hhmmss] ])
        elif len(ff) == 0:
            sys.stderr.write("could not find .wav file in %s for %s\n" % (dir, hhmmss))
        else:
            sys.stderr.write("multiple files in %s for %s: %s\n" % (dir, hhmmss, ff))

    return benchmark1(dir, info, verbose)

def benchmark1(dir, bfiles, verbose):
    global chan
    chan = 0
    crcok = 0 # how many we decoded
    jtscore = 0 # how many we decoded that wsjt-x also decoded
    jtwanted = 0 # how many wsjt-x decoded
    hints = [ Hint("CQ") ]
    heard1 = [ ] # callsigns heard in previous cycle
    cqs = { } # all CQs ever heard
    hashes22 = { }
    hashes12 = { }
    hashes10 = { }
    r = FT8()
    r.verbose = False
    for bf in bfiles:
        if not bf[0]: # only the short list
            continue
        if verbose:
            print(bf[1])
        filename = dir + "/" + bf[1]
        r.hashes22 = hashes22
        r.hashes12 = hashes12
        r.hashes10 = hashes10
        if False:
            r.restrict_hz = [ 1500, 1750 ]
        if False:
            # hints other than CQ are not worthwhile, particularly not since
            # (when operating) we're only interested in exotic CQ and in replies to us.
            r.hints = hints
        if False:
            # just testing, common misses from ft8-n13/big.txt
            r.hints = [ Hint("CQ"), Hint("YV6QD"), Hint("YI3WHR"),
                        Hint("9M2MRS"), Hint("Z36W"),
                        Hint(call2="IW4BNN"), Hint(call2="OK2BZ"),
                        Hint(call2="IK4ISR"), Hint(call2="G3PXT") ]
        if True:
            # hint CQs previously heard, limited by hz at which last heard.
            r.hints = [ Hint("CQ") ]
            for call in cqs.keys():
                dec = cqs[call]
                hint = Hint(call1="CQ", call2=call, hz=dec.hz())
                r.hints.append(hint)
        r.gowav(filename, chan)
        all = r.get_msgs()
        crcok += len(all)
        got = { } # did wsjt-x see this? indexed by msg.
        any_no = False

        # populate hashes of non-standard calls for next cycle.
        for m in all:
            for hc in m.hashcalls:
                hashes22[ihashcall(hc, 22)] = hc
                hashes12[ihashcall(hc, 12)] = hc
                hashes10[ihashcall(hc, 10)] = hc

        # populate hints for next cycle, using callsigns that
        # others apparently heard but we did not.
        hints = [ Hint("CQ") ]
        for m in all:
            txt = m.msg
            txt = txt.strip()
            txt = re.sub(r'  *', ' ', txt)
            if txt[0:2] == "CQ":
                continue
            a = txt.split(" ")
            if len(a) != 3:
                continue
            if not (a[0] in heard1):
                # only calls that were called in this cycle,
                # and that we didn't hear from in the previous cycle.
                if not (a[0] in [ h.call2 for h in hints ]):
                    hints.append(Hint(call2=a[0]))
            #if not (a[1] in [ h.call1 for h in hints ]):
            #    hints.append(Hint(call1=a[1]))

        # for hints driven by CQs, add to list of CQs we have heard.
        for m in all:
            txt = m.msg
            txt = txt.strip()
            txt = re.sub(r'  *', ' ', txt)
            a = txt.split(" ")
            if len(a) == 3 and a[0] == "CQ":
                cqs[a[1]] = m

        # populate heard1, for hints.
        heard1 = [ ]
        for m in all:
            txt = m.msg
            txt = txt.strip()
            txt = re.sub(r'  *', ' ', txt)
            a = txt.split(" ")
            if len(a) != 3:
                continue
            if len(a[1]) < 3:
                continue
            if not (a[1] in heard1):
                heard1.append(a[1])

        wsa = bf[2].split("\n")
        this_got = 0
        this_wanted = 0
        for wsx in wsa:
            # 161245 -21  0.3  538 ~  K3OWX KG5AUW -03
            # 161245   2  0.1  955 ~  KJ1J NS9I -06
            wsx = wsx.strip()
            if wsx != "":
                jtwanted += 1
                this_wanted += 1
                wsx = re.sub(r'  *', ' ', wsx)
                found = None
                for dec in all:
                    mymsg = dec.msg
                    mymsg = mymsg.strip()
                    mymsg = re.sub(r'  *', ' ', mymsg)
                    if mymsg in wsx:
                        found = dec
                        got[dec.msg] = True

                wa = wsx.split(' ')
                wmsg = ' '.join(wa[5:8])
                whz = float(wa[3])
                if whz >= 10 and whz < 11:
                    whz = (whz - 10.1387) * 1000000.0
                elif whz >= 14 and whz < 15:
                    whz = (whz - 14.0956) * 1000000.0
                elif whz < 1.0:
                    whz = whz * 1000000.0

                if found != None:
                    jtscore += 1
                    this_got += 1
                    if verbose:
                        print("yes %4.0f %s %s (%.1f %.1f) %s" % (float(whz), wa[2], wa[1], found.hz(), found.dt, wmsg))
                else:
                    any_no = True
                    if verbose:
                        print("no  %4.0f %s %s %s" % (float(whz), wa[2], wa[1], wmsg))
                sys.stdout.flush()
        extras = [ ]
        for dec in all:
            if not (dec.msg in got) and not (dec.msg in extras):
                extras.append(dec.msg)
                if verbose:
                    print("EXTRA: %6.1f %s" % (dec.hz(), dec.msg))
        if False and any_no:
            # emit lines only for files in which we
            # some decodes that wsjt-x got.
            print(bf[2])
        if False:
            # per-file scores, to pick out the weakest.
            print("filescore %s %.2f" % (bf[1], this_got/float(this_wanted)))

    if verbose:
        print("score %d (%d of %d)" % (crcok, jtscore, jtwanted))

    return [ crcok, jtscore, jtwanted ]

vars = [
    [ "cheb_ripple_pass", [ 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0 ] ],
    [ "decimate_order", [ 4, 5, 6, 7, 8, 10, 12, 14, 16 ] ],
    [ "top_high_order", [ 0, 9, 11, 13, 15, 17, 19, 21, 23, 25 ] ],
    [ "cheb_atten_stop", [ 30, 35, 40, 45, 47, 50, 53, 55, 60, 65, 70  ] ],
    [ "cheb_high_minus", [ 70, 60, 50, 40, 30, 20 ] ],
    [ "cheb_high_plus", [ 120, 110, 100, 90, 80, 70, 60, 50, 40, 30 ] ],
    [ "nchildren", [ 4, 3, 2, 1 ] ],
    [ "child_overlap", [ 0, 10, 20, 30, 40, 50, 60 ] ],
    [ "pass0_tstep", [ 8, 4, 2, 1 ] ],
    [ "passN_tstep", [ 8, 4, 2, 1 ] ],
    [ "fine_tstep", [ 16, 8, 4, 2, 1 ] ],
    [ "pass0_fstep", [ 8, 4, 2, 1 ] ],
    [ "passN_fstep", [ 8, 4, 2, 1 ] ],
    [ "fine_fstep", [ 8, 4, 2, 1 ] ],
    [ "pass0_frac", [ 0.6, 0.8, 1.0, 1.2, 1.5 ] ],
    [ "ldpc_iters", [ 13, 17, 20, 25, 30, 40, 50, 70 ] ],
    [ "no_mul", [ 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2 ] ],
    [ "no_add", [  -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3 ] ],
    [ "yes_mul", [ 0.6, 0.8, 0.9, 1.0, 1.1, 1.2 ] ],
    [ "yes_add", [ -0.2,  -0.1, 0.0, 0.1, 0.2, 0.3 ] ],
    [ "snr_overlap", [ -1, 0, 1, 2, 3, 4, 5, 7, 9, 11, 20, 30, 40 ] ],
    [ "subpasses", [ 0, 1, 2, 3 ] ],
    [ "do_subtract", [ 0, 1, 2, 3, 4 ] ],
    [ "pass0_hints", [ True, False ] ],
    [ "use_apriori", [ True, False ] ],
    [ "start_adj", [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ] ],
    [ "top_down", [ True, False ] ],
    [ "bottom_slow", [ True, False ] ],
    [ "down200", [ True, False ] ],
    [ "coarse_no", [ 1, 2, 3, 4 ] ],
    [ "pass0_tplus", [ 2.4, 2.5, 2.6, 2.7 ] ],
    [ "osd0_crc", [ True, False ] ],
    [ "osd_crc", [ True, False ] ],
    [ "osd_hints", [ True, False ] ],
    [ "osd_depth", [ -1, 0, 2, 4, 5, 6, 7, 8, 10, 20, 30, 40 ] ],
    [ "down_cutoff", [ 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65 ] ],
    [ "pass0_tminus", [ 1.1, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6 ] ],
    [ "crc_and_83", [ True, False ] ],
    [ "ldpc_thresh", [ 50, 60, 70, 75, 80, 81, 82, 83 ] ],
    [ "hint_tol", [ -1, 3, 9, 20 ] ],
    [ "passN_tplus", [ 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0 ] ],
    [ "passN_tminus", [ 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 ] ],
    [ "adjust_off_for_sub", [ True, False ] ],
    [ "adjust_hz_for_sub", [ True, False ] ],
    [ "weakutil.which_fft", [ "\"numpy\"", "\"scipy\"", "\"fftw\"" ] ],
    [ "osd_thresh", [ -300, -500, -800, -1000, -1200, -1500, -1800, -2000 ] ],
    [ "osd_no_snr", [ True, False ] ],
    [ "budget", [ 2, 4, 11 ] ],
    # [ "noise_factor", [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.6, 2.0 ] ],
    # [ "high_cutoff", [ 0.9, 0.96, 0.98, 1.0, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2 ] ],
    # [ "strength_div", [ 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 8, 12, 16, 20 ] ],
    # [ "contrast_weight", [ 0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5 ] ],
    # [ "phase_gran", [ 200, 150, 125, 100, 75, 50, 25 ] ],
    # [ "sub_amp_win", [ 0, 1, 2, 3, 4 ] ],
    # [ "soft1", [ 4, 5, 6, 7 ] ],
    # [ "soft2", [ 7, 8 ] ],
    # [ "soft3", [ 3, 4, 5, 6 ] ],
    # [ "soft4", [ 6, 7, 8 ] ],
    # [ "low_pass_order", [ 0, 5, 7, 10, 13, 15, 17, 20, 25 ] ],
    # [ "cheb_cut1", [ 0.46, 0.47, 0.475, 0.48, 0.485, 0.49 ] ],
    # [ "cheb_cut2", [ 0.55, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.65 ] ],
    # [ "snr_wintype", [ "\"bartlett\"", "\"blackman\"", "\"hamming\"", "\"hanning\"", "\"kaiser\"", "\"boxcar\"", "\"cosine\"", "\"tukey\"", "\"triang\"", "\"flattop\"", "\"gaussian\"", "\"nuttall\"", "\"parzen\"" ] ],
    # [ "real_min_hz", [ 120, 130, 140, 150, 160, 170, 180, 190, 200 ] ],
    # [ "real_max_hz", [ 2840, 2850, 2860, 2870, 2880, 2890, 2900, 2910, 2920, 2930, 2940, 2950 ] ],
    # [ "subgap", [ 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4 ] ],
    # [ "substeps", [ 5, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30 ] ],
    # [ "already_o", [ 0, 1, 2, 3, 4 ] ],
    # [ "already_f", [ 0, 1, 2, 3, 4 ] ],
    # [ "softboost", [ 0, 0.125, 0.25, 0.33, 0.5, 1.0, 1.5, 2.0 ] ],
    ]

def printvars():
    s = ""
    for v in vars:
        s += "%s=%s " % (v[0], eval(v[0]))
    return s

def optimize(wsjtfile):
    global very_first_time

    set_start_adj(wsjtfile)

    sys.stdout.write("# %s %s\n" % (opt, printvars()))

    for v in vars:
        for val in v[1]:
            # warm up any caches, JIT, &c.
            very_first_time = True
            r = FT8()
            r.verbose = False
            #r.gowav("ft8files/170717_161300.wav", 0)
            r.gowav("ft8-n9/181218_100845.wav", 0)
            r = None

            old = None
            if "." in v[0]:
                xglob = ""
            else:
                xglob = "global %s ; " % (v[0])
            exec("%sold = %s" % (xglob, v[0]))
            exec("%s%s = %s" % (xglob, v[0], val))

            [ crcok, jtscore, jtwanted ] = benchmark(wsjtfile, False)

            exec("%s%s = old" % (xglob, v[0]))
            sys.stdout.write("%s=%s : " % (v[0], val))
            sys.stdout.write("%d %d %d\n" % (crcok, jtscore, jtwanted))
            sys.stdout.flush()

filename = None
card = None
bench = None
opt = None

def main():
    global filename, card, bench, opt, very_first_time
    global budget
    do_cpubench = False
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-card":
            card = [ sys.argv[i+1], sys.argv[i+2] ]
            i += 3
        elif sys.argv[i] == "-file":
            filename = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-bench":
            bench = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-opt":
            opt = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-cpubench":
            do_cpubench = True
            i += 1
        else:
            usage()

    if do_cpubench:
        # warm up
        very_first_time = True
        r = FT8()
        r.verbose = False
        r.gowav("ft8-n15/190513_205800.wav", 0)
        r = None

        very_first_time = True # don't fork
        budget = 1000 # typically needs 20 or 30 seconds
        r = FT8()
        r.verbose = False
        t0 = time.time()
        r.gowav("ft8-n15/190513_210715.wav", 0)
        t1 = time.time()
        all = r.get_msgs()
        print("%d decodes, %d ldpc calls, %.1f seconds" % (len(all), r.ldpc_calls, t1 - t0))
        sys.exit(0)

    if False:
        xr = FT8()
        xr.test_guess_offset()
        sys.exit(0)

    if bench != None:
        set_start_adj(bench)
        sys.stdout.write("# %s %s\n" % (bench, printvars()))
        benchmark(bench, True)
        sys.exit(0)

    if opt != None:
        optimize(opt)
        sys.exit(0)

    if filename != None and card == None:
        very_first_time = False # don't warm up, since only single call
        set_start_adj(filename)
        r = FT8()
        r.verbose = True
        r.gowav(filename, 0)
    elif filename == None and card != None:
        r = FT8()
        r.verbose = True
        r.opencard(card)
        r.gocard()
    else:
        usage()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        sys.argv = sys.argv[1:]
        profiling = True
        pfile = "cprof.out"
        import cProfile
        import pstats
        cProfile.run('main()', pfile)
        p = pstats.Stats(pfile)
        p.strip_dirs().sort_stats('time')
        p.print_callers()
    else:
        main()
