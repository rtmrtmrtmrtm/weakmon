#
# random shared support routines for weak*.py
#

#
# read weak.ini
# e.g. weakcfg.get("wsprmon", "mycall") -> None or "W1XXX"
#

import ConfigParser
import numpy
import scipy

def cfg(program, key):
    cfg = ConfigParser.SafeConfigParser()
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
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

fff_cached_window_set = False
fff_cached_window = None

# https://gist.github.com/endolith/255291
def freq_from_fft(sig, rate, minf, maxf):
    global fff_cached_window, fff_cached_window_set
    if fff_cached_window_set == False:
      # this uses a bunch of CPU time.
      fff_cached_window = scipy.signal.blackmanharris(len(sig))
      fff_cached_window_set = True
    assert len(sig) == len(fff_cached_window)

    windowed = sig * fff_cached_window
    f = numpy.fft.rfft(windowed)
    fa = abs(f)

    # find max between minf and maxf
    mini = int(minf * len(windowed) / rate)
    maxi = int(maxf * len(windowed) / rate)
    i = numpy.argmax(fa[mini:maxi]) + mini # peak bin

    if fa[i] <= 0.0:
        return None

    true_i = parabolic(numpy.log(fa), i)[0] # interpolate

    return rate * true_i / float(len(windowed)) # convert to frequency

def moving_average(a, n):
    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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
