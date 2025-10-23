from numpy.typing import ArrayLike
import numpy as np

from .util import fftfreq

class Filter:
    """
    General class holding filter parameters
    """
    def __init__(self, scale: float = 1.0, power: float = 2.0):
        self.scale = scale
        self.power = power
    def __call__(self, freq):
        return np.ones_like(freq)

class Lowpass(Filter):
    """
    Callable class providing low-pass frequency-space multiplicative filter values.
    """
    def __call__(self, freq):
        return np.exp(-0.5 * np.pow(freq / self.scale, self.power))

class Highpass(Filter):
    """
    Callable class providing high-pass frequency-space multiplicative filter values.
    """
    def __call__(self, freq):
        return 1.0 - np.exp(-np.pow(freq / self.scale, self.power))


    
