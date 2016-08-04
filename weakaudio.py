#
# get at sound cards on both Mac and FreeBSD.
# Mac wants pyaudio; FreeBSD wants ossaudiodev.
# 
# pyaudio maybe should work on FreeBSD, but various
# things go wrong.
#

import sys
import numpy
import scipy
import scipy.signal
from scipy.signal import lfilter
import time
import thread
import threading
import re
import os

import weakutil

import sdrip
import sdriq
import eb200

# desc is "6:0" for a sound card -- sixth card, channel 0 (left).
def new(desc, rate):
    # sound card?
    m = re.search(r'^([0-9]+):([0-9]+)$', desc)
    if m != None:
        return Stream(int(m.group(1)), int(m.group(2)), rate)

    m = re.search(r'^sdrip:([0-9.]+)$', desc)
    if m != None:
        return SDRIP(m.group(1), rate)

    m = re.search(r'^sdriq:(/.+)$', desc)
    if m != None:
        return SDRIQ(m.group(1), rate)

    m = re.search(r'^eb200:([0-9.]+)$', desc)
    if m != None:
        return EB200(m.group(1), rate)

    sys.stderr.write("weakaudio: unknown desc %s" % (desc))
    sys.exit(1)

# need a single one of these even if multiple streams.
global_pya = None

def pya():
    global global_pya
    import pyaudio
    if global_pya == None:
        # suppress Jack and ALSA error messages on Linux.
        nullfd = os.open("/dev/null", 1)
        oerr = os.dup(2)
        os.dup2(nullfd, 2)

        global_pya = pyaudio.PyAudio()

        os.dup2(oerr, 2)
        os.close(oerr)
        os.close(nullfd)
    return global_pya

class Stream:
    def __init__(self, card, chan, rate):
        self.use_oss = ("freebsd" in sys.platform)
        self.card = card
        self.chan = chan
        self.rate = rate # the sample rate the app wants.
        self.cardrate = rate # the rate at which the card is running.

        self.cardbuf = numpy.array([])
        self.cardtime = time.time() # UNIX time just after last sample in cardbuf
        self.cardlock = thread.allocate_lock()

        if self.use_oss:
            self.oss_open()
        else:
            self.pya_open()

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    def read(self):
        self.cardlock.acquire()
        buf = self.cardbuf
        buf_time = self.cardtime
        self.cardbuf = numpy.array([])
        self.cardlock.release()

        if self.rate != self.cardrate and len(buf) > 0:
            # low-pass filter, then down-sample from self.cardrate to self.rate.
            zi = lfilter(self.filter[0], self.filter[1], buf, zi=self.zi)
            buf = zi[0]
            self.zi = zi[1]

            secs =  len(buf)*(1.0/self.cardrate)
            ox = numpy.arange(0, secs, 1.0 / self.cardrate)
            ox = ox[0:len(buf)]
            nx = numpy.arange(0, secs, 1.0 / self.rate)
            buf = numpy.interp(nx, ox, buf)

        return [ buf, buf_time ]

    # PyAudio calls this in a separate thread.
    def pya_callback(self, in_data, frame_count, time_info, status):
        import pyaudio

        # time_info['input_buffer_adc_time'] is time of first sample.
        # but what is it time since? we need a good guess since
        # the point is to avoid having to search for the start of
        # each minute.
      
        if status != 0:
            sys.stderr.write("pya_callback status %d\n" % (status))

        #http://portaudio.com/docs/v19-doxydocs-dev/structPaStreamCallbackTimeInfo.html
        #Time values are expressed in seconds and are synchronised with the time base used by Pa_GetStreamTime() for the associated stream.
        #unspecified origin
        # so on startup we need to find diff against time.time()

        pcm = numpy.fromstring(in_data, dtype=numpy.int16)
        pcm = pcm[self.chan::2] # chan is 0/1 for left/right

        # time of first sample in pcm[].
        adc_time = time_info['input_buffer_adc_time']

        # translate to UNIX time
        ut = time.time()
        st = self.pya_strm.get_time()
        adc_time = (adc_time - st) + ut

        # make it time of last sample in self.cardbuf[]
        adc_time += (len(pcm) / float(self.rate))

        self.cardlock.acquire()
        self.cardbuf = numpy.concatenate((self.cardbuf, pcm))
        self.cardtime = adc_time
        self.cardlock.release()

        return ( None, pyaudio.paContinue )

    def pya_open(self):
        import pyaudio

        self.cardrate = self.pya_rate(self.rate)

        self.pya_strm = pya().open(format=pyaudio.paInt16,
                                   input_device_index=self.card,
                                   channels=2,
                                   rate=self.cardrate,
                                   frames_per_buffer=self.cardrate,
                                   stream_callback=self.pya_callback,
                                   output=False,
                                   input=True)

        if self.cardrate != self.rate:
            sys.stderr.write("weakaudio: down-sampling from %d to %d\n" % (self.cardrate,
                                                                           self.rate))
            # prepare a filter to precede resampling.
            self.filter = weakutil.butter_lowpass(0.45 * self.rate, self.cardrate, 10)
            self.zi = scipy.signal.lfiltic(self.filter[0],
                                           self.filter[1],
                                           [0])

    # find the lowest supported rate >= rate.
    # needed on Linux but not the Mac (which converts as needed).
    def pya_rate(self, rate):
        import pyaudio
        rates = [ rate, 8000, 11025, 12000, 16000, 22050, 44100, 48000 ]
        for r in rates:
            if r >= rate:
                ok = False
                try:
                    ok = pya().is_format_supported(r,
                                                   input_device=self.card,
                                                   input_format=pyaudio.paInt16,
                                                   input_channels=2)
                except:
                    pass
                if ok:
                    return r
        sys.stderr.write("weakaudio: can't find a supported rate >= %d\n" % (rate))
        sys.exit(1)

    def oss_open(self):
        import ossaudiodev
        self.oss = ossaudiodev.open("/dev/dsp" + str(self.card) + ".0", "r")
        self.oss.setfmt(ossaudiodev.AFMT_S16_LE)
        self.oss.channels(2)
        assert self.oss.speed(self.rate) == self.rate
        self.th = threading.Thread(target=lambda : self.oss_thread())
        self.th.daemon = True
        self.th.start()

    # dedicating reading thread because oss's buffering seems
    # to be pretty limited, and wspr.py spends 50 seconds in
    # process() while not calling read().
    def oss_thread(self):
        # XXX the card probably doesn't read the first sample at this
        # exact point, and probably doesn't read at exactly self.rate
        # samples per second.
        self.cardtime = time.time()

        while True:
            # the read() blocks.
            buf = self.oss.read(8192)
            assert len(buf) > 0
            both = numpy.fromstring(buf, dtype=numpy.int16)
            got = both[self.chan::2]

            self.cardlock.acquire()
            self.cardbuf = numpy.concatenate((self.cardbuf, got))
            self.cardtime += len(got) / float(self.rate)
            self.cardlock.release()

    # print levels, to help me adjust volume control.
    def levels(self):
        while True:
            time.sleep(1)
            [ buf, junk ] = self.read()
            if len(buf) > 0:
                print "avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf))

class SDRIP:
    def __init__(self, ip, rate):
        self.rate = rate
        self.sdrrate = 32000

        self.bufbuf = [ ]
        self.cardtime = time.time() # UNIX time just after last sample in bufbuf
        self.cardlock = thread.allocate_lock()

        # prepare down-sampling filter.
        self.filter = weakutil.butter_lowpass(0.45 * self.rate, self.sdrrate, 10)
        self.zi = scipy.signal.lfiltic(self.filter[0],
                                       self.filter[1],
                                       [0])

        self.sdr = sdrip.open(ip)
        self.sdr.setrate(self.sdrrate)
        self.sdr.setrun()
        self.sdr.setgain(-10)

        self.th = threading.Thread(target=lambda : self.sdr_thread())
        self.th.daemon = True
        self.th.start()

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    def read(self):
        self.cardlock.acquire()
        bufbuf = self.bufbuf
        buf_time = self.cardtime
        self.bufbuf = [ ]
        self.cardlock.release()

        if len(bufbuf) == 0:
            return [ numpy.array([]), buf_time ]

        buf = numpy.concatenate(bufbuf)
        buf = sdrip.iq2usb(buf) # I/Q -> USB

        # low-pass filter, then down-sample from 32000 to self.rate.
        zi = lfilter(self.filter[0], self.filter[1], buf, zi=self.zi)
        buf = zi[0]
        self.zi = zi[1]

        secs =  len(buf)*(1.0/self.sdrrate)
        ox = numpy.arange(0, secs, 1.0 / self.sdrrate)
        ox = ox[0:len(buf)]
        nx = numpy.arange(0, secs, 1.0 / self.rate)
        buf = numpy.interp(nx, ox, buf)

        return [ buf, buf_time ]

    def sdr_thread(self):
        self.cardtime = time.time()

        while True:
            # read i/q blocks, to reduce CPU time in
            # this thread, which drains the UDP socket.
            got = self.sdr.readiq()

            self.cardlock.acquire()
            self.bufbuf.append(got)
            self.cardtime += len(got) / float(self.sdrrate)
            self.cardlock.release()

    # print levels, to help me adjust volume control.
    def levels(self):
        while True:
            time.sleep(1)
            [ buf, junk ] = self.read()
            if len(buf) > 0:
                print "avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf))

class SDRIQ:
    def __init__(self, ip, rate):
        self.rate = rate
        self.sdrrate = 8138

        self.bufbuf = [ ]
        self.cardtime = time.time() # UNIX time just after last sample in bufbuf
        self.cardlock = thread.allocate_lock()

        if self.rate < self.sdrrate:
            # prepare down-sampling filter.
            self.filter = weakutil.butter_lowpass(0.45 * self.rate, self.sdrrate, 10)
            self.zi = scipy.signal.lfiltic(self.filter[0],
                                           self.filter[1],
                                           [0])

        self.sdr = sdriq.open(ip)
        self.sdr.setrate(self.sdrrate)
        self.sdr.setgain(-10)
        self.sdr.setifgain(12) # I don't know how to set this!
        self.sdr.setrun(True)

        self.th = threading.Thread(target=lambda : self.sdr_thread())
        self.th.daemon = True
        self.th.start()

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    def read(self):
        self.cardlock.acquire()
        bufbuf = self.bufbuf
        buf_time = self.cardtime
        self.bufbuf = [ ]
        self.cardlock.release()

        if len(bufbuf) == 0:
            return [ numpy.array([]), buf_time ]

        buf = numpy.concatenate(bufbuf)
        buf = sdrip.iq2usb(buf) # I/Q -> USB

        if self.rate < self.sdrrate:
            # low-pass filter, then down-sample from self.sdrrate to self.rate.
            zi = lfilter(self.filter[0], self.filter[1], buf, zi=self.zi)
            buf = zi[0]
            self.zi = zi[1]
            
        secs =  len(buf)*(1.0/self.sdrrate)
        ox = numpy.arange(0, secs, 1.0 / self.sdrrate)
        ox = ox[0:len(buf)]
        nx = numpy.arange(0, secs, 1.0 / self.rate)
        buf = numpy.interp(nx, ox, buf)

        # no matter how I set its RF or IF gain,
        # the SDR-IP generates peaks around 145000,
        # or I and Q values of 65535. cut this down
        # so application doesn't think the SDR-IP is clipping.
        buf = buf / 10.0

        return [ buf, buf_time ]

    def sdr_thread(self):
        self.cardtime = time.time()

        while True:
            # read i/q blocks, to reduce CPU time in
            # this thread, which drains the UDP socket.
            got = self.sdr.readiq()

            self.cardlock.acquire()
            self.bufbuf.append(got)
            self.cardtime += len(got) / float(self.sdrrate)
            self.cardlock.release()

    # print levels, to help me adjust volume control.
    def levels(self):
        while True:
            time.sleep(1)
            [ buf, junk ] = self.read()
            if len(buf) > 0:
                print "avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf))

class EB200:
    def __init__(self, ip, rate):
        self.rate = rate

        self.time_mu = thread.allocate_lock()
        self.cardtime = time.time() # UNIX time just after last sample in bufbuf

        self.sdr = eb200.open(ip)
        self.sdrrate = self.sdr.getrate()

        if self.sdrrate > self.rate:
            # prepare down-sampling filter.
            self.filter = weakutil.butter_lowpass(0.45 * self.rate, self.sdrrate, 10)
            self.zi = scipy.signal.lfiltic(self.filter[0],
                                           self.filter[1],
                                           [0])


    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    # blocks until input is available.
    def read(self):
        buf = self.sdr.readaudio()

        self.time_mu.acquire()
        self.cardtime += len(buf) / float(self.sdrrate)
        buf_time = self.cardtime
        self.time_mu.release()

        if self.sdrrate > self.rate:
            # low-pass filter.
            zi = lfilter(self.filter[0], self.filter[1], buf, zi=self.zi)
            buf = zi[0]
            self.zi = zi[1]

        if self.sdrrate != self.rate:
            # change sample rate.
            secs =  len(buf)*(1.0/self.sdrrate)
            ox = numpy.arange(0, secs, 1.0 / self.sdrrate)
            ox = ox[0:len(buf)]
            nx = numpy.arange(0, secs, 1.0 / self.rate)
            buf = numpy.interp(nx, ox, buf)

        return [ buf, buf_time ]

    # print levels, to help me adjust volume control.
    def levels(self):
        while True:
            time.sleep(1)
            [ buf, junk ] = self.read()
            if len(buf) > 0:
                print "avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf))

#
# for Usage(), print out a list of audio cards
# and associated number (for the "card" argument).
#
def usage():
    import pyaudio
    ndev = pya().get_device_count()
    sys.stderr.write("sound card numbers:\n")
    for i in range(0, ndev):
        info = pya().get_device_info_by_index(i) 
        sys.stderr.write("  %d: %s, channels=%d" % (i,
                                                      info['name'],
                                                      info['maxInputChannels']))
        if True and info['maxInputChannels'] > 0:
            rates = [ 11025, 12000, 16000, 22050, 44100, 48000 ]
            for rate in rates:
                try:
                    ok = pya().is_format_supported(rate,
                                                   input_device=i,
                                                   input_format=pyaudio.paInt16,
                                                   input_channels=2)
                except:
                    ok = False
                if ok:
                    sys.stderr.write(" %d" % (rate))
        sys.stderr.write("\n")
    sys.stderr.write("  or sdrip:IPADDR\n")
    sys.stderr.write("  or sdriq:/dev/SERIALPORT\n")
    sys.stderr.write("  or eb200:IPADDR\n")
