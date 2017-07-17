#
# get at sound cards on both Mac and FreeBSD.
# Mac wants pyaudio; FreeBSD wants ossaudiodev.
# 
# pyaudio maybe should work on FreeBSD, but various
# things go wrong.
#

import sys
import numpy
import time
import threading
import os

import weakutil

import sdrip
import sdriq
import eb200
import sdrplay
import fmdemod

# desc is [ "6", "0" ] for a sound card -- sixth card, channel 0 (left).
# desc is [ "sdrip", "192.168.1.2" ] for RFSpace SDR-IP.
def new(desc, rate):
    # sound card?
    if desc[0].isdigit():
        return Stream(int(desc[0]), int(desc[1]), rate)

    if desc[0] == "sdrip":
        return SDRIP(desc[1], rate)

    if desc[0] == "sdriq":
        return SDRIQ(desc[1], rate)

    if desc[0] == "eb200":
        return EB200(desc[1], rate)

    if desc[0] == "sdrplay":
        return SDRplay(desc[1], rate)

    sys.stderr.write("weakaudio: cannot understand card %s\n" % (desc[0]))
    usage()
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

# find the lowest supported input rate >= rate.
# needed on Linux but not the Mac (which converts as needed).
def pya_input_rate(card, rate):
    import pyaudio
    rates = [ rate, 8000, 11025, 12000, 16000, 22050, 44100, 48000 ]
    for r in rates:
        if r >= rate:
            ok = False
            try:
                ok = pya().is_format_supported(r,
                                               input_device=card,
                                               input_format=pyaudio.paInt16,
                                               input_channels=1)
            except:
                pass
            if ok:
                return r
    sys.stderr.write("weakaudio: no input rate >= %d\n" % (rate))
    sys.exit(1)

def pya_output_rate(card, rate):
    import pyaudio
    rates = [ rate, 8000, 11025, 12000, 16000, 22050, 44100, 48000 ]
    for r in rates:
        if r >= rate:
            ok = False
            try:
                ok = pya().is_format_supported(r,
                                               output_device=card,
                                               output_format=pyaudio.paInt16,
                                               output_channels=1)
            except:
                pass
            if ok:
                return r
    sys.stderr.write("weakaudio: no output rate >= %d\n" % (rate))
    sys.exit(1)

class Stream:
    def __init__(self, card, chan, rate):
        self.use_oss = ("freebsd" in sys.platform)
        self.card = card
        self.chan = chan

        if rate == None:
            rate = pya_output_rate(card, 8000)

        self.rate = rate # the sample rate the app wants.
        self.cardrate = rate # the rate at which the card is running.

        self.cardbuf = numpy.array([])
        self.cardtime = time.time() # UNIX time just after last sample in cardbuf
        self.cardlock = threading.Lock()

        if self.use_oss:
            self.oss_open()
        else:
            self.pya_open()

        self.resampler = weakutil.Resampler(self.cardrate, self.rate)

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    # non-blocking.
    def read(self):
        self.cardlock.acquire()
        buf = self.cardbuf
        buf_time = self.cardtime
        self.cardbuf = numpy.array([], dtype=numpy.int16)
        self.cardlock.release()

        if len(buf) > 0:
            buf = self.resampler.resample(buf)

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
        if self.chan == 1:
            pcm = pcm[self.chan::2]

        # time of first sample in pcm[].
        adc_time = time_info['input_buffer_adc_time']

        # translate to UNIX time
        ut = time.time()
        st = self.pya_strm.get_time()
        adc_time = (adc_time - st) + ut

        # make it time of last sample in self.cardbuf[]
        adc_time += (len(pcm) / float(self.cardrate))

        self.cardlock.acquire()
        self.cardbuf = numpy.concatenate((self.cardbuf, pcm))
        self.cardtime = adc_time
        self.cardlock.release()

        return ( None, pyaudio.paContinue )

    def pya_open(self):
        import pyaudio

        self.cardrate = pya_input_rate(self.card, self.rate)

        # only ask for 2 channels if we want channel 1,
        # since some sound cards are mono.
        chans = self.chan + 1

        self.pya_strm = pya().open(format=pyaudio.paInt16,
                                   input_device_index=self.card,
                                   channels=chans,
                                   rate=self.cardrate,
                                   frames_per_buffer=self.cardrate,
                                   stream_callback=self.pya_callback,
                                   output=False,
                                   input=True)

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
                print("avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf)))

class SDRIP:
    def __init__(self, ip, rate):
        if rate == None:
            rate = 11025

        self.rate = rate
        self.sdrrate = 32000
        self.fm = fmdemod.FMDemod(self.sdrrate)

        self.resampler = weakutil.Resampler(self.sdrrate, self.rate)

        self.sdr = sdrip.open(ip)
        self.sdr.setrate(self.sdrrate)
        #self.sdr.setgain(-10)

        # now weakcat.SDRIP.read() calls setrun().
        #self.sdr.setrun()

        self.cardtime = time.time() # UNIX time just after last sample in bufbuf

        self.bufbuf = [ ]
        self.cardlock = threading.Lock()
        self.th = threading.Thread(target=lambda : self.sdr_thread())
        self.th.daemon = True
        self.th.start()

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    def read(self):
        # delay setrun() until the last moment, so that
        # all other parameters have likely been set.
        if self.sdr.running == False:
            self.sdr.setrun()

        self.cardlock.acquire()
        bufbuf = self.bufbuf
        buf_time = self.cardtime
        self.bufbuf = [ ]
        self.cardlock.release()

        #buf = self.sdr.readiq()
        #self.cardtime += len(buf) / float(self.sdrrate)
        #buf_time = self.cardtime

        if len(bufbuf) == 0:
            return [ numpy.array([]), buf_time ]

        buf = numpy.concatenate(bufbuf)

        # XXX maybe should be moved to sdrip.py?
        if self.sdr.mode == "usb":
            buf = weakutil.iq2usb(buf) # I/Q -> USB
        elif self.sdr.mode == "fm":
            [ buf, junk ] = self.fm.demod(buf) # I/Q -> FM
        else:
            sys.stderr.write("weakaudio: SDRIP unknown mode %s\n" % (self.sdr.mode))
            sys.exit(1)

        buf = self.resampler.resample(buf)
        buf = buf.astype(numpy.float32) # save some space.

        return [ buf, buf_time ]

    def sdr_thread(self):

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
                print("avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf)))

class SDRIQ:
    def __init__(self, ip, rate):
        if rate == None:
            rate = 11025

        self.rate = rate
        self.sdrrate = 8138

        self.bufbuf = [ ]
        self.cardtime = time.time() # UNIX time just after last sample in bufbuf
        self.cardlock = threading.Lock()

        self.resampler = weakutil.Resampler(self.sdrrate, self.rate)

        self.sdr = sdriq.open(ip)
        self.sdr.setrate(self.sdrrate)
        self.sdr.setgain(0)
        self.sdr.setifgain(18) # I don't know how to set this!
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
        buf = weakutil.iq2usb(buf) # I/Q -> USB

        buf = self.resampler.resample(buf)

        # no matter how I set its RF or IF gain,
        # the SDR-IQ generates peaks around 145000,
        # or I and Q values of 65535. cut this down
        # so application doesn't think the SDR-IQ is clipping.
        buf = buf / 10.0

        return [ buf, buf_time ]

    def sdr_thread(self):
        self.cardtime = time.time()

        while True:
            # read i/q blocks, float64, to reduce CPU time in
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
                print("avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf)))

class EB200:
    def __init__(self, ip, rate):
        if rate == None:
            rate = 8000

        self.rate = rate

        self.time_mu = threading.Lock()
        self.cardtime = time.time() # UNIX time just after last sample in bufbuf

        self.sdr = eb200.open(ip)
        self.sdrrate = self.sdr.getrate()

        self.resampler = weakutil.Resampler(self.sdrrate, self.rate)

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    # blocks until input is available.
    def read(self):
        buf = self.sdr.readaudio()

        self.time_mu.acquire()
        self.cardtime += len(buf) / float(self.sdrrate)
        buf_time = self.cardtime
        self.time_mu.release()

        buf = self.resampler.resample(buf)

        return [ buf, buf_time ]

    # print levels, to help me adjust volume control.
    def levels(self):
        while True:
            time.sleep(1)
            [ buf, junk ] = self.read()
            if len(buf) > 0:
                print("avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf)))

class SDRplay:
    def __init__(self, dev, rate):
        if rate == None:
            rate = 11025

        self.rate = rate

        self.sdr = sdrplay.open(dev)
        self.sdrrate = self.sdr.getrate()

        self.resampler = weakutil.Resampler(self.sdrrate, self.rate)

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    def read(self):
        [ buf, buf_time ] = self.sdr.readiq()

        buf = weakutil.iq2usb(buf) # I/Q -> USB

        buf = self.resampler.resample(buf)

        return [ buf, buf_time ]

    # print levels, to help me adjust volume control.
    def levels(self):
        while True:
            time.sleep(1)
            [ buf, junk ] = self.read()
            if len(buf) > 0:
                print("avg=%.0f max=%.0f" % (numpy.mean(abs(buf)), numpy.max(buf)))

#
# for Usage(), print out a list of audio cards
# and associated number (for the "card" argument).
#
def usage():
    import pyaudio
    ndev = pya().get_device_count()
    sys.stderr.write("sound card numbers for -card and -out:\n")
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
                                                   input_channels=1)
                except:
                    ok = False
                if ok:
                    sys.stderr.write(" %d" % (rate))
        sys.stderr.write("\n")
    sys.stderr.write("  or -card sdrip IPADDR\n")
    sys.stderr.write("  or -card sdriq /dev/SERIALPORT\n")
    sys.stderr.write("  or -card eb200 IPADDR\n")
    sys.stderr.write("  or -card sdrplay sdrplay\n")

# implement -levels.
# print sound card avg/peak once per second, to adjust level.
# never returns.
def levels(card):
    if card == None:
        sys.stderr.write("-levels requires -card\n")
        sys.exit(1)
    c = new(card, 11025)
    c.levels()
    sys.exit(0)
