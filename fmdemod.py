#
# FM demodulation, from i/q to audio.
#

import numpy
import scipy
import scipy.signal

def butter_lowpass(cut, samplerate, order=5):
  nyq = 0.5 * samplerate
  cut = cut / nyq
  b, a = scipy.signal.butter(order, cut, btype='lowpass')
  return b, a

class FMDemod:

  def __init__(self, rate):
    self.rate = rate
    self.width = 8000
    self.order = 3
    self.filter = None
    self.zi = None

  #
  # input is complex i/q raw samples.
  # output is (audio, amplitudes).
  # audio is the demodulated audio.
  # amplitudes are the i/q amplitudes, so the caller
  # can compute SNR (if they can guess what is signal
  # and what is noise).
  #
  def demod(self, samples):
    # complex low-pass filter.
    # passes fmwidth on either side of zero,
    # so real width is 2*self.width.
    if self.filter == None:
      self.filter = butter_lowpass(self.width, self.rate, self.order)
      self.zi = scipy.signal.lfiltic(self.filter[0],
                                       self.filter[1],
                                       [0])
    bzi = scipy.signal.lfilter(self.filter[0],
                               self.filter[1],
                               samples,
                               zi=self.zi)
    cc2 = bzi[0]
    self.zi = bzi[1]

    # quadrature fm demodulation, as in gnuradio gr_quadrature_demod_cf.
    # seems to be the same as the Lyons scheme.
    product = numpy.multiply(cc2[1:], numpy.conjugate(cc2[0:-1]))
    diff = numpy.angle(product)
    diff = numpy.append(diff, diff[-1])

    # don't de-emphasize, since intended for aprs.

    # calculate amplitude at each sample,
    # for later snr calculation.
    # calculated as post-filter length of complex vector.
    amp = numpy.sqrt(numpy.add(numpy.square(cc2.real), numpy.square(cc2.imag)))

    return (diff, amp)
