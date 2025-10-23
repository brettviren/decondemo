from numpy.typing import ArrayLike
import numpy as np

from .util import fftfreq, nhalf as ffthalf

class Filter:
    """
    General class holding filter parameters
    """
    def __init__(self, scale: float = 1.0, power: float = 2.0, ignore_baseline = False):
        self.scale = scale
        self.power = power
        self.ignore_baseline = ignore_baseline

    def value(self, freq):
        '''
        Return filter values without regards to the sampling domain.

        Subclass should provide this method.  Default gives unity.

        '''
        return np.ones_like(freq)

    def __call__(self, N, d=1.0):
        '''
        Sample the filter across N positive frequency samples.
        '''
        freqs = fftfreq(N, d)
        half = ffthalf(N)

        start = 0
        if self.ignore_baseline: # zero-frequency has zero value
            start = 1

        if N%2:                 # odd, no nyquist sample
            end = half+1
        else:                   # even, have nyquist sample
            end = half+2
        assert N-end == half    # check my math

        values = np.zeros_like(freqs)
        values[start:end] = self.value(freqs[start:end])
        values[end:] = values[1:half+1][::-1]
        return values

class Lowpass(Filter):
    """
    Callable class providing low-pass frequency-space multiplicative filter values.
    """
    def value(self, freq):
        return np.exp(-0.5 * np.pow(freq / self.scale, self.power))

class Highpass(Filter):
    """
    Callable class providing high-pass frequency-space multiplicative filter values.
    """
    def value(self, freq):
        return 1.0 - np.exp(-np.pow(freq / self.scale, self.power))


    
