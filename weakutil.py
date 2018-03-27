#
# shared support routines for weak*.py
#

#
# read weak.ini
# e.g. weakcfg.get("wsprmon", "mycall") -> None or "W1XXX"
#

try:
  import configparser
except:
  from six.moves import configparser
import threading
import numpy
import scipy
import scipy.signal
import scipy.fftpack
import wave
import time
import sys
import math
import random

def cfg(program, key):
    cfg = configparser.SafeConfigParser()
    cfg.read(['weak-local.cfg', 'weak.cfg'])

    if cfg.has_option(program, key):
        return cfg.get(program, key)

    return None

# make a butterworth IIR bandpass filter
def butter_bandpass(lowcut, highcut, samplerate, order=5):
  # http://wiki.scipy.org/Cookbook/ButterworthBandpass
  nyq = 0.5 * samplerate
  low = lowcut / nyq
  high = highcut / nyq
  b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
  return b, a

def butter_lowpass(cut, samplerate, order=5):
  nyq = 0.5 * samplerate
  cut = cut / nyq
  b, a = scipy.signal.butter(order, cut, btype='lowpass')
  return b, a

# FIR bandpass filter
# http://stackoverflow.com/questions/16301569/bandpass-filter-in-python
def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = scipy.signal.firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                               window=window, scale=False)
    return taps

#
# frequency shift via hilbert transform (like SSB).
# Lev Givon, https://gist.github.com/lebedov/4428122
#
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""
    return int(numpy.ceil(numpy.log2(numpy.abs(x))))

# f_shift in Hz.
# dt is 1 / sample rate (e.g. 1.0/11025).
# frequencies near the low and high ends of the spectrum are
# reflected, so large shifts will be a problem.
def freq_shift(x, f_shift, dt):
    """Shift the specified signal by the specified frequency."""

    # Pad the signal with zeros to prevent the FFT invoked by the transform from
    # slowing down the computation:
    N_orig = len(x)
    N_padded = 2**nextpow2(N_orig)
    t = numpy.arange(0, N_padded)
    lo = numpy.exp(2j*numpy.pi*f_shift*dt*t)
    x0 = numpy.append(x, numpy.zeros(N_padded-N_orig, x.dtype))
    h = scipy.signal.hilbert(x0)*lo
    ret = h[:N_orig].real
    return ret

def one_test_freq_shift(rate, hz, n, f_shift):
    t1 = costone(rate, hz, n)

    t2 = freq_shift(t1, f_shift, 1.0 / rate)

    expected = costone(rate, hz + f_shift, n)

    if False:
        import matplotlib.pyplot as plt
        i = 1000
        j = i + 1000
        plt.plot(t1[i:j])
        plt.plot(t2[i:j])
        plt.plot(expected[i:j])
        plt.show()

    diff = t2 - expected
    x = math.sqrt(numpy.mean(diff * diff))
    return x

def test_freq_shift():
    x = one_test_freq_shift(12000, 511, 12000*13, 6.25 / 4)
    print(x)

    t1 = numpy.random.rand(12000 * 13)
    t1 += costone(12000, 5970, 12000 * 13)
    t1 += costone(12000, 511, 12000 * 13)
    #t1 += costone(12000, 10, 12000 * 13)

    t2 = freq_shift(t1, 60, 1.0 / 12000)

    t3 = freq_shift(t2, -60, 1.0 / 12000)

    diff = t3 - t1
    x = math.sqrt(numpy.mean(diff * diff))
    print(x)

    writewav1(t1, "t1.wav", 12000)
    writewav1(t2, "t2.wav", 12000)
    writewav1(t3, "t3.wav", 12000)

# caller supplies two shifts, in hza[0] and hza[1].
# shift x[0] by hza[0], ..., x[-1] by hza[1]
# corrects for phase change:
# http://stackoverflow.com/questions/3089832/sine-wave-glissando-from-one-pitch-to-another-in-numpy
# sadly, more expensive than piece-wise freq_shift() because
# the ramp part does noticeable work.
def freq_shift_ramp(x, hza, dt):
    N_orig = len(x)
    N_padded = 2**nextpow2(N_orig)
    t = numpy.arange(0, N_padded)
    f_shift = numpy.linspace(hza[0], hza[1], len(x))
    f_shift = numpy.append(f_shift, hza[1]*numpy.ones(N_padded-len(x)))

    pc1 = f_shift[:-1] - f_shift[1:]
    phase_correction = numpy.add.accumulate(
      t * dt * numpy.append(numpy.zeros(1), 2*numpy.pi*pc1))

    lo = numpy.exp(1j*(2*numpy.pi*dt*f_shift*t + phase_correction))
    x0 = numpy.append(x, numpy.zeros(N_padded-N_orig, x.dtype))
    h = scipy.signal.hilbert(x0)*lo
    ret = h[:N_orig].real
    return ret

# avoid most of the round-up-to-power-of-two penalty by
# doing log-n shifts. discontinuity at boundaries,
# but that's OK for JT65 2048-sample symbols.
def freq_shift_hack(x, hza, dt):
    a = freq_shift_hack_iter(x, hza, dt)
    return numpy.concatenate(a)

def freq_shift_hack_iter(x, hza, dt):
    if len(x) <= 4096:
        return [ freq_shift(x, (hza[0] + hza[1]) / 2.0, dt) ]
    lg = nextpow2(len(x))
    if len(x) == 2**lg:
        return [ freq_shift_ramp(x, hza, dt) ]
    
    i1 = 2**(lg-1)
    hz_i1 = hza[0] + (i1 / float(len(x)))*(hza[1] - hza[0])
    ret = [ freq_shift_ramp(x[0:i1], [ hza[0], hz_i1 ], dt) ]
    ret +=  freq_shift_hack_iter(x[i1:], [ hz_i1, hza[1] ], dt) 
    return ret

# pure sin tone, n samples long.
def sintone(rate, hz, n):
    x = numpy.linspace(0, 2 * hz * (n / float(rate)) * numpy.pi, n,
                       endpoint=False, dtype=numpy.float32)
    tone = numpy.sin(x)
    return tone

# pure cos tone, n samples long.
def costone(rate, hz, n):
    x = numpy.linspace(0, 2 * hz * (n / float(rate)) * numpy.pi, n,
                       endpoint=False, dtype=numpy.float32)
    tone = numpy.cos(x)
    return tone

# parameter
fos_threshold = 0.5

# cache
fos_mu = threading.Lock()
fos_hz = None
fos_fft = None
fos_n = None

# shift signal a up by hz, returning
# the rfft of the resulting signal.
# do it by convolving ffts.
# use numpy.fft.irfft(ret, len(a)) to get the signal.
# intended for single power-of-two-length JT65 symbols.
# faster than freq_shift() if you need the fft (not the
# actual resulting signal).
def fft_of_shift(a, hz, rate):
    global fos_hz, fos_n, fos_fft

    afft = rfft(a)

    bin_hz = rate / float(len(a))

    # we want this many fft bins on either side of the tone.
    pad_bins = 10

    # shift the tone up by integral bins until it is
    # enough above zero that we get fft bins on either side.
    # XXX if hz is more than 10 bins above zero, negative shift.
    if hz >= 0:
        shift_bins = max(0, pad_bins - int(hz / bin_hz))
    else:
        shift_bins = int((-hz) / bin_hz) + pad_bins
    hz += shift_bins * bin_hz

    # fos_mu.acquire()
    if fos_n == len(a) and abs(fos_hz - hz) < fos_threshold:
        lofft = fos_fft
    else:
        lo = sintone(rate, hz, len(a))
        lofft = rfft(lo)
        fos_hz = hz
        fos_fft = lofft
        fos_n = len(a)
    # fos_mu.release()

    lo_bin = int(hz / bin_hz)
    lofft = lofft[0:lo_bin+pad_bins]

    outfft = numpy.convolve(lofft, afft, 'full')
    outfft = outfft[shift_bins:]

    outfft = outfft[0:len(afft)]
    return outfft

# https://gist.github.com/endolith/255291
# thank you, endolith.
def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f is a vector and x is an index for that vector.
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    denom = (f[x-1] - 2 * f[x] + f[x+1])
    if denom == 0.0:
        return None
    xv = 1/2. * (f[x-1] - f[x+1]) / denom + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

fff_cached_window_set = False
fff_cached_window = None

# https://gist.github.com/endolith/255291
def freq_from_fft(sig, rate, minf, maxf):
    global fff_cached_window, fff_cached_window_set
    if fff_cached_window_set == False or len(sig) != len(fff_cached_window):
      # this uses a bunch of CPU time.
      fff_cached_window = scipy.signal.blackmanharris(len(sig))
      fff_cached_window = fff_cached_window.astype(numpy.float32)
      fff_cached_window_set = True

    n = len(sig)

    fa = sig * fff_cached_window
    #fa = abs(numpy.fft.rfft(fa))
    fa = arfft(fa)

    # find max between minf and maxf
    mini = int(minf * n / rate)
    maxi = int(maxf * n / rate)

    i = numpy.argmax(fa[mini:maxi]) + mini # peak bin

    if i < 1 or i+1 >= len(fa) or fa[i] <= 0.0:
        return None

    if fa[i-1] == 0.0 or fa[i] == 0.0 or fa[i+1] == 0.0:
        return None

    xp = parabolic(numpy.log(fa[i-1:i+2]), 1) # interpolate
    if xp == None:
        return None
    true_i = xp[0] + i - 1

    return rate * true_i / float(n) # convert to frequency

def moving_average(a, n):
    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# IQ -> USB
# the result is bad at the start and end,
# so not ideal for processing a sequence of
# sample blocks.
def iq2usb_internal(iq):
    ii = iq.real
    qq = iq.imag
    nn = len(iq)
    ii = numpy.real(scipy.signal.hilbert(ii, nn)) # delay to match hilbert(Q)
    qq = numpy.imag(scipy.signal.hilbert(qq, nn))
    ssb = numpy.subtract(ii, qq) # usb from phasing method
    #ssb = numpy.add(ii, qq) # lsb from phasing method
    assert len(iq) == len(ssb)
    return ssb

# do it in overlapping smallish chunks, so the FFTs are smaller
# and faster.
# always returns the same number of samples as you give it.
def iq2usb(iq):
    guard = 256 # overlap between successive chunks
    chunksize = 8192 - 2*guard

    bufbuf = [ ]
    oi = 0 # output index, == sum of bufbuf[] lengths
    while oi < len(iq):
        sz = min(chunksize, len(iq) - oi)
        buf = iq[oi:oi+sz]

        # pad so always chunksize
        if len(buf) < chunksize:
            buf = numpy.append(buf, numpy.zeros(chunksize - len(buf)))
        
        # prepend the guard
        if oi >= guard:
            buf = numpy.append(iq[oi-guard:oi], buf)
        else:
            buf = numpy.append(numpy.zeros(guard), buf)

        # append the guard
        n1 = min(guard, len(iq) - (oi + chunksize))
        n1 = max(n1, 0)
        if n1 > 0:
            buf = numpy.append(buf, iq[oi+chunksize:oi+chunksize+n1])
        if n1 < guard:
            buf = numpy.append(buf, numpy.zeros(guard - n1))

        assert len(buf) == chunksize + 2*guard

        z = iq2usb_internal(buf)

        assert len(z) == chunksize + 2*guard

        bufbuf.append(z[guard:-guard])
        oi += len(bufbuf[-1])

    buf = numpy.concatenate(bufbuf)
    assert len(buf) >= len(iq)
    buf = buf[0:len(iq)]
    return buf

# simple measure of distortion introduced by iq2usb.
def one_test_iq2usb(rate, hz, n):
    ii = costone(rate, hz, n)
    qq = sintone(rate, hz, n)
    iq = ii + 1j*qq
    usb = iq2usb(iq)

    # usb ought to be 2*ii
    
    if False:
        import matplotlib.pyplot as plt
        plt.plot(ii[0:100])
        plt.plot(qq[0:100])
        plt.plot(usb[0:100])
        plt.show()

    diff = (usb / 2.0) - ii
    x = math.sqrt(numpy.mean(diff * diff))
    return x

# original (no chunking):
#   32000 511 1333: 0.038 0.001
#   32000 511 27777: 0.010 0.016
#   32000 511 320001: 0.000 6.345
# chunksize 8192, guard 128
#   32000 511 1333: 0.023 0.002
#   32000 511 27777: 0.006 0.006
#   32000 511 320001: 0.002 0.041
def test_iq2usb():
    for [ rate, hz, n ] in [
            [ 32000, 511, 1333 ],
            [ 32000, 511, 27777 ],
            [ 32000, 511, 320001 ]
            ]:
        t0 = time.time()
        x = one_test_iq2usb(rate, hz, n)
        t1 = time.time()
        print("%d %d %d: %.3f %.3f" % (rate, hz, n, x, t1 - t0))

resample_interp = False # use numpy.interp()? vs scipy.signal.resample()

# change sampling rate from from_rate to to_rate.
# buf must already be low-pass filtered if to_rate < from_rate.
# note that result probably has slightly different
# length in seconds, so caller may want to adjust.
def resample(buf, from_rate, to_rate):
    # how many samples do we want?
    target = int(round((len(buf) / float(from_rate)) * to_rate))

    if from_rate == to_rate * 2:
        buf = buf[0::2]
        return buf

    # 11025 -> 441, for wwvmon.py.
    if from_rate == to_rate * 25:
        buf = buf[0::25]
        return buf

    # 11025 -> 315, for wwvmon.py.
    if from_rate == to_rate * 35:
        buf = buf[0::35]
        return buf

    if from_rate == to_rate * 64:
        buf = buf[0::64]
        return buf

    if from_rate == to_rate:
        return buf

    if resample_interp == False:
        # seems to produce better results than numpy.interp() but
        # is slower, sometimes much slower.
        # pad to power of two length
        nn = 2**nextpow2(len(buf))
        buf = numpy.append(buf, numpy.zeros(nn - len(buf)))
        want = (len(buf) / float(from_rate)) * to_rate
        want = int(round(want))
        buf = scipy.signal.resample(buf, want)
    else:
        secs =  len(buf)*(1.0/from_rate)
        ox = numpy.arange(0, secs, 1.0 / from_rate)
        ox = ox[0:len(buf)]
        nx = numpy.arange(0, secs, 1.0 / to_rate)
        buf = numpy.interp(nx, ox, buf)

    if len(buf) > target:
        buf = buf[0:target]

    return buf

# gadget to low-pass-filter and re-sample a multi-block
# stream without losing fractional samples at block
# boundaries, which would hurt phase-shift demodulators
# like WWVB.
class Resampler:
    def __init__(self, from_rate, to_rate):
        self.from_rate = from_rate
        self.to_rate = to_rate

        if self.from_rate > self.to_rate:
            # prepare a filter to precede resampling.
            self.filter = butter_lowpass(0.45 * self.to_rate,
                                         from_rate,
                                         7)
            self.zi = scipy.signal.lfiltic(self.filter[0],
                                           self.filter[1],
                                           [0])

        # total number of input and output samples,
        # so we can insert/delete to keep long-term
        # rates correct.
        self.nin = 0
        self.nout = 0

    # how much will output be delayed?
    # in units of output samples.
    def delay(self, hz):
        if self.from_rate > self.to_rate:
            # convert hz to radians per sample,
            # at input sample rate.
            rps = (1.0 / hz) * (2 * math.pi) / self.from_rate
            gd = scipy.signal.group_delay(self.filter, w=[rps])
            n = (gd[1][0] / self.from_rate) * self.to_rate
            return n
        else:
            return 0

    def resample(self, buf):
        # if resample() uses FFT, then handing it huge chunks is
        # slow. so cut big buffers into one-second chunks.
        a = [ ]
        i = 0
        if self.from_rate > 20000:
            big = self.from_rate / 2
        else:
            big = self.from_rate
        while i < len(buf):
            left = len(buf) - i
            chunk = None
            if left > 1.5*big:
                chunk = big
            else:
                chunk = left
            a.append(self.resample1(buf[i:i+chunk]))
            i += chunk
        b = numpy.concatenate(a)
        return b

    def resample1(self, buf):
        inlen = len(buf)
        savelast = buf[-20:]

        insec = self.nin / float(self.from_rate)
        outsec = self.nout / float(self.to_rate)
        if insec - outsec > 0.5 / self.from_rate:
            ns = (insec - outsec) / (1.0 / self.from_rate)
            ns = int(round(ns))
            if ns < 1:
                ns = 1
            assert len(self.last) >= ns
            #print("add %d" % (ns))
            buf = numpy.append(self.last[-ns:], buf)
        if outsec - insec > 0.5 / self.from_rate:
            ns = (outsec - insec) / (1.0 / self.from_rate)
            ns = int(round(ns))
            if ns < 1:
                ns = 1
            #print("del %d" % (ns))
            buf = buf[ns:]
        
        self.last = savelast

        if self.from_rate > self.to_rate:
            # low-pass filter.
            zi = scipy.signal.lfilter(self.filter[0],
                                      self.filter[1],
                                      buf,
                                      zi=self.zi)
            buf = zi[0]
            self.zi = zi[1]

        buf = resample(buf, self.from_rate, self.to_rate)
            
        self.nin += inlen
        self.nout += len(buf)

        return buf

def one_test_resampler(from_rate, to_rate):
    hz = 511
    t1 = costone(from_rate, hz, from_rate*10)

    r = Resampler(from_rate, to_rate)

    i = 0
    bufbuf = [ ]
    while i < len(t1):
        n = random.randint(1, from_rate)
        buf = r.resample(t1[i:i+n])
        bufbuf.append(buf)
        i += n
    t2 = numpy.concatenate(bufbuf)

    expecting = costone(to_rate, hz, to_rate*10)
    delay = r.delay(hz)
    delay = int(round(delay))
    expecting = numpy.append(numpy.zeros(delay), expecting)
    expecting = expecting[0:to_rate*10]

    diff = t2 - expecting
    x = math.sqrt(numpy.mean(diff * diff))

    if True:
        import matplotlib.pyplot as plt
        i0 = len(diff) - 200
        i1 = i0 + 200
        plt.plot(expecting[i0:i1])
        plt.plot(t2[i0:i1])
        plt.plot(diff[i0:i1])
        plt.show()

    return x

# measure Resampler distortion.
# not very revealing since the filter delay
# prevents easy comparison.
def test_resampler():
    global resample_interp
    ori = resample_interp

    for [ r1, r2 ] in [
            [ 32000, 12000 ],
            ]:

        resample_interp = False
        x = one_test_resampler(r1, r2)
        print("%d %d False: %.3f" % (r1, r2, x))

        #resample_interp = True
        #x = one_test_resampler(r1, r2)
        #print("%d %d True: %.3f" % (r1, r2, x))

    resample_interp = ori

use_numpy_arfft = False

# wrapper for either numpy or scipy rfft(),
# followed by abs().
# scipy rfft() is nice b/c it supports float32.
# but not faster for 2048-point FFTs.
def arfft(x):
    if use_numpy_arfft:
        y = numpy.fft.rfft(x)
        y = abs(y)
        return y
    else:
        assert (len(x) % 2) == 0

        y = scipy.fftpack.rfft(x)

        if type(y[0]) == numpy.float32:
            cty = numpy.complex64
        elif type(y[0]) == numpy.float64:
            cty = numpy.complex128
        else:
            assert False

        # y = [ Re0, Re1, Im1, Re2, Im2, ..., ReN ]
        #y1 = numpy.sqrt(numpy.add(numpy.square(y[1:-1:2]),
        #                          numpy.square(y[2:-1:2])))
        y0 = abs(y[0])
        yn = abs(y[-1])
        y = abs(y[1:-1].view(cty))
        y = numpy.concatenate(([y0], y, [yn]))
        return y

use_numpy_rfft = False

# wrapper for either numpy or scipy rfft().
# scipy rfft() is nice b/c it supports float32.
# but not faster for 2048-point FFTs.
def rfft(x):
    if use_numpy_rfft:
        y = numpy.fft.rfft(x)
        return y
    else:
        assert (len(x) % 2) == 0

        y = scipy.fftpack.rfft(x)

        if type(y[0]) == numpy.float32:
            cty = numpy.complex64
        elif type(y[0]) == numpy.float64:
            cty = numpy.complex128
        else:
            assert False

        # y = [ Re0, Re1, Im1, Re2, Im2, ..., ReN ]
        y1 = y[1:-1].view(cty)
        y = numpy.concatenate((y[0:1], y1, y[-1:]))
        return y

# apply automatic gain control.
# causes each winlen window of samples
# to have average absolute value of 1.0.
# winlen is in units of samples.
def agc(samples, winlen):
    assert winlen >= 3 # tukey only works if winlen > 2
    agcwin = scipy.signal.tukey(winlen)
    agcwin = agcwin / numpy.sum(agcwin)
    mavg = numpy.convolve(abs(samples), agcwin)[winlen/2:]
    mavg = mavg[0:len(samples)]
    samples = numpy.divide(samples, mavg)
    samples = numpy.nan_to_num(samples)
    return samples

# write a mono file
def writewav1(left, filename, rate):
  ww = wave.open(filename, 'w')
  ww.setnchannels(1)
  ww.setsampwidth(2)
  ww.setframerate(rate)

  # ensure signal levels are OK.
  mx = numpy.max(left)
  left = (10000.0 * left) / mx

  # convert to 16-bit ints
  a = numpy.array(left, dtype=numpy.int16)

  # convert to python raw byte string
  a = a.tostring()

  ww.writeframes(a)

  ww.close()

# write a stereo file
def writewav2(left, right, filename, rate):
  ww = wave.open(filename, 'w')
  ww.setnchannels(2)
  ww.setsampwidth(2)
  ww.setframerate(rate)

  # ensure signal levels are OK.
  mx = numpy.max(left)
  left = (10000.0 * left) / mx
  mx = numpy.max(right)
  right = (10000.0 * right) / mx

  # interleave.
  a = numpy.zeros(len(left) + len(right))
  a[0::2] = left
  a[1::2] = right

  # convert to 16-bit ints
  a = numpy.array(a, dtype=numpy.int16)

  # convert to python raw byte string
  a = a.tostring()

  ww.writeframes(a)

  ww.close()

# read a whole mono wav file.
# return None or [ rate, samples ].
def readwav(filename):
    w = wave.open(filename)
    channels = w.getnchannels()
    width = w.getsampwidth()
    rate = w.getframerate()

    if width != 1 and width != 2:
        sys.stderr.write("oops width %d in %s" % (width, filename))
        w.close()
        return None
    if channels != 1 and channels != 2:
        sys.stderr.write("oops channels %d in %s" % (channels, filename))
        w.close()
        return None

    bufbuf = [ ]
    while True:
        z = w.readframes(8192)
        if len(z) < 1:
            break
        if width == 1:
            zz = numpy.fromstring(z, numpy.int8)
        else:
            assert (len(z) % 2) == 0
            zz = numpy.fromstring(z, numpy.int16)
        if channels == 2:
            zz = zz[0::2] # left channel
        bufbuf.append(zz)

    samples = numpy.concatenate(bufbuf)

    w.close()
    return [ rate, samples ]

#
# gray code
# https://rosettacode.org/wiki/Gray_code
#

def int2bin(n, nb):
  'From positive integer to list of binary bits, msb at index 0'
  if n:
    bits = []
    while n:
      n,remainder = divmod(n, 2)
      bits.insert(0, remainder)
    out = bits
  else:
    out = [0]
  while len(out) < nb:
    out = [0] + out
  return out
 
 
def bin2int(bits):
  'From binary bits, msb at index 0 to integer'
  i = 0
  for bit in bits:
    i = i * 2 + bit
  return i

def bin2gray(x, nb):
  bits = int2bin(x, nb)
  gbits = bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]
  return bin2int(gbits)
 
def gray2bin(x, nb):
  bits = int2bin(x, nb)
  b = [bits[0]]
  for nextb in bits[1:]: b.append(b[-1] ^ nextb)
  return bin2int(b)

# generate coherent FSK.
# symbols[] are the symbols to encode.
# hza[0] is start hz of base tone, hza[1] is end hz,
#   can be different if drift.
# spacing is the space in hz between FSK tones.
# rate is samples per second.
# symsamples is samples per symbol.
def fsk(symbols, hza, spacing, rate, symsamples, phase0=0.0):
    # compute frequency for each symbol, in hz[0]..hz[1].
    symhz = numpy.zeros(len(symbols))
    for bi in range(0, len(symbols)):
        base_hz = hza[0] + (hza[1] - hza[0]) * (bi / float(len(symbols)))
        fr = base_hz + (symbols[bi] * spacing)
        symhz[bi] = fr

    # frequency for each sample.
    hzv = numpy.repeat(symhz, symsamples)

    # cumulative angle.
    angles = numpy.cumsum(2.0 * numpy.pi / (float(rate) / hzv))

    # start at indicated phase.
    angles = angles + phase0

    a = numpy.sin(angles)

    return a

# weighted choice (to pick bands).
# a[i] = [ value, weight ]
def wchoice(a, n):
    total = 0.0
    for e in a:
        total += e[1]

    ret = [ ]
    while len(ret) < n:
        x = random.random() * total
        for ai in range(0, len(a)):
            e = a[ai]
            if x <= e[1]:
                ret.append(e[0])
                total -= e[1]
                a = a[0:ai] + a[ai+1:]
                break
            x -= e[1]

    return ret

def wchoice_test():
    a = [ [ "a", .1 ], [ "b", .1 ], [ "c", .4 ], [ "d", .3 ], [ "e", .1 ] ]
    counts = { }
    for iter in range(0, 500):
        x = wchoice(a, 2)
        assert len(x) == 2
        for e in x:
            counts[e] = counts.get(e, 0) + 1
    print(counts)
