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
import multiprocessing
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
def x_pya_input_rate(card, rate):
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

# sub-process to avoid initializing pyaudio in main
# process, since that makes subsequent forks and
# multiprocessing not work.
def pya_input_rate(card, rate):
    rpipe, wpipe = multiprocessing.Pipe(False)
    pid = os.fork()
    if pid == 0:
        rpipe.close()
        x = x_pya_input_rate(card, rate)
        wpipe.send(x)
        os._exit(0)
    wpipe.close()
    x = rpipe.recv()
    os.waitpid(pid, 0)
    rpipe.close()
    return x

def x_pya_output_rate(card, rate):
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

def pya_output_rate(card, rate):
    rpipe, wpipe = multiprocessing.Pipe(False)
    pid = os.fork()
    if pid == 0:
        rpipe.close()
        x = x_pya_output_rate(card, rate)
        wpipe.send(x)
        os._exit(0)
    wpipe.close()
    x = rpipe.recv()
    os.waitpid(pid, 0)
    rpipe.close()
    return x

class Stream:
    def __init__(self, card, chan, rate):
        self.use_oss = ("freebsd" in sys.platform)
        self.card = card
        self.chan = chan

        if rate == None:
            rate = pya_output_rate(card, 8000)

        self.rate = rate # the sample rate the app wants.
        self.cardrate = rate # the rate at which the card is running.

        self.cardbufs = [ ]
        self.cardlock = threading.Lock()

        self.last_adc_end = None
        self.last_end_time = None

        if self.use_oss:
            self.oss_open()
        else:
            self.pya_open()

        self.resampler = weakutil.Resampler(self.cardrate, self.rate)

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    # non-blocking.
    # reads from a pipe from pya_dev2pipe in the pya sub-process.
    # XXX won't work for oss.
    def read(self):
        bufs = [ ]
        end_time = self.last_end_time
        while self.rpipe.poll():
            e = self.rpipe.recv()
            # e is [ pcm, unix_end_time ]
            bufs.append(e[0])
            end_time = e[1]

        if len(bufs) > 0:
            buf = numpy.concatenate(bufs)
        else:
            buf = numpy.array([])

        if len(buf) > 0:
            buf = self.resampler.resample(buf)

        self.last_end_time = end_time

        return [ buf, end_time ]

    def junklog(self, msg):
      msg1 = "[%d, %d] %s\n" % (self.card, self.chan, msg)
      sys.stderr.write(msg1)
      f = open("ft8-junk.txt", "a")
      f.write(msg1)
      f.close()

    # PyAudio calls this in a separate thread.
    def pya_callback(self, in_data, frame_count, time_info, status):
        import pyaudio
      
        if status != 0:
            self.junklog("pya_callback status %d\n" % (status))

        pcm = numpy.fromstring(in_data, dtype=numpy.int16)
        if self.chan == 1:
            pcm = pcm[self.chan::2]

        assert frame_count == len(pcm)

        # time of first sample in pcm[], in seconds since start.
        adc_time = time_info['input_buffer_adc_time']
        # time of last sample
        adc_end = adc_time + (len(pcm) / float(self.cardrate))

        if self.last_adc_end != None:
            expected = (adc_end - self.last_adc_end) * float(self.cardrate)
            expected = int(round(expected))
            shortfall = expected - len(pcm)
            if abs(shortfall) > 20:
                self.junklog("pya expected %d got %d" % (expected, len(pcm)))
                #if shortfall > 100:
                #    pcm = numpy.append(numpy.zeros(shortfall, dtype=pcm.dtype), pcm)
                    
        self.last_adc_end = adc_end

        # translate time of last sample to UNIX time
        ut = time.time()
        st = self.pya_strm.get_time()
        unix_end = (adc_end - st) + ut

        self.cardlock.acquire()
        self.cardbufs.append([ pcm, unix_end ])
        self.cardlock.release()

        return ( None, pyaudio.paContinue )

    def pya_open(self):
        self.cardrate = pya_input_rate(self.card, self.rate)
        
        # read from sound card in a separate process, since Python
        # scheduler seems sometimes not to run the py audio thread
        # often enough.
        sys.stdout.flush()
        rpipe, wpipe = multiprocessing.Pipe(False)
        proc = multiprocessing.Process(target=self.pya_dev2pipe, args=[rpipe,wpipe])
        proc.start()
        wpipe.close()
        self.rpipe = rpipe

    # executes in a sub-process.
    def pya_dev2pipe(self, rpipe, wpipe):
        import pyaudio

        rpipe.close()

        # only ask for 2 channels if we want channel 1,
        # since some sound cards are mono.
        chans = self.chan + 1

        # perhaps this controls how often the callback is called.
        # too big and ft8.py's read() is delayed long enough to
        # cut into FT8 decoding time. too small and apparently the
        # callback thread can't keep up.
        bufsize = int(self.cardrate / 4)

        # pya.open in this sub-process so that pya starts the callback thread
        # here too.
        xpya = pya()
        self.pya_strm = xpya.open(format=pyaudio.paInt16,
                                   input_device_index=self.card,
                                   channels=chans,
                                   rate=self.cardrate,
                                   frames_per_buffer=bufsize,
                                   stream_callback=self.pya_callback,
                                   output=False,
                                   input=True)

        # copy buffers from self.cardbufs, where pya_callback left them,
        # to the pipe to the parent process. can't do this in the callback
        # because the pipe write might block.
        # each object on the pipe is [ pcm, unix_end ].
        while True:
            self.cardlock.acquire()
            bufs = self.cardbufs
            self.cardbufs = [ ]
            self.cardlock.release()
            if len(bufs) > 0:
                for e in bufs:
                    try:
                        wpipe.send(e)
                    except:
                        os._exit(1)
            else:
                time.sleep(0.1)
            

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
            self.cardbufs.append(got)
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

        self.ip = ip
        self.rate = rate
        self.sdrrate = 32000
        self.fm = fmdemod.FMDemod(self.sdrrate)

        self.resampler = weakutil.Resampler(self.sdrrate, self.rate)

        self.sdr = sdrip.open(ip)
        self.sdr.setrate(self.sdrrate)
        #self.sdr.setgain(-10)

        # now weakcat.SDRIP.read() calls setrun().
        #self.sdr.setrun()

        self.starttime = time.time() # for faking a sample clock
        self.cardcount = 0 # for faking a sample clock

        self.bufbuf = [ ]
        self.cardlock = threading.Lock()
        self.th = threading.Thread(target=lambda : self.sdr_thread())
        self.th.daemon = True
        self.th.start()

    def junklog(self, msg):
      msg1 = "[%s] %s\n" % (self.ip, msg)
      #sys.stderr.write(msg1)
      f = open("ft8-junk.txt", "a")
      f.write(msg1)
      f.close()

    # returns [ buf, tm ]
    # where tm is UNIX seconds of the last sample.
    def read(self):
        # delay setrun() until the last moment, so that
        # all other parameters have likely been set.
        if self.sdr.running == False:
            self.sdr.setrun()

        self.cardlock.acquire()
        bufbuf = self.bufbuf
        cardcount = self.cardcount
        self.bufbuf = [ ]
        self.cardlock.release()

        buf_time = self.starttime + cardcount / float(self.sdrrate)

        if len(bufbuf) == 0:
            return [ numpy.array([]), buf_time ]

        buf1 = numpy.concatenate(bufbuf)

        # XXX maybe should be moved to sdrip.py?
        if self.sdr.mode == "usb":
            buf2 = weakutil.iq2usb(buf1) # I/Q -> USB
        elif self.sdr.mode == "fm":
            [ buf2, junk ] = self.fm.demod(buf1) # I/Q -> FM
        else:
            sys.stderr.write("weakaudio: SDRIP unknown mode %s\n" % (self.sdr.mode))
            sys.exit(1)

        buf3 = self.resampler.resample(buf2)
        buf4 = buf3.astype(numpy.float32) # save some space.

        return [ buf4, buf_time ]

    def sdr_thread(self):
        self.starttime = time.time()

        while True:
            # read pipe from sub-process.
            got = self.sdr.readiq()

            self.cardlock.acquire()
            self.bufbuf.append(got)
            self.cardcount += len(got)
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
        self.starttime = time.time() # for faking a sample clock
        self.cardcount = 0 # for faking a sample clock
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
        cardcount = self.cardcount
        self.bufbuf = [ ]
        self.cardlock.release()

        buf_time = self.starttime + cardcount / float(self.sdrrate)

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
        self.starttime = time.time()

        while True:
            # read i/q blocks, float64, to reduce CPU time in
            # this thread, which drains the UDP socket.
            got = self.sdr.readiq()

            self.cardlock.acquire()
            self.bufbuf.append(got)
            self.cardcount += len(got)
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
