from numpy.typing import ArrayLike
import numpy as np

from .util import fftfreq, nhalf as ffthalf, zero_pad

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

class Taper:
    """
    Implements a custom padding function using a window /taper function
    applied to the start and end of the array, followed by zero-padding.
    """
    def __init__(self, taper_length: int, window):
        if taper_length <= 0:
            raise ValueError("Taper length must be positive.")
        self.L = taper_length
        # k is the length of the taper applied to one end (L // 2)
        self.k = self.L // 2
        self.window = window(self.L)

    def __call__(self, array: np.ndarray, size: int) -> np.ndarray:
        N_A = len(array)
        
        if size < N_A:
            raise ValueError("Target size must be greater than or equal to array length.")
        
        if N_A < self.L:
            raise ValueError(f"Array length ({N_A}) must be greater than or equal to taper length ({self.L}).")

        # 1. Create the multiplier array M, initialized to 1.0
        multiplier = np.ones(N_A, dtype=array.dtype)
        
        # 2. Apply first half of the window (W[0:k]) to the start (M[0:k])
        multiplier[:self.k] = self.window[:self.k]
        
        # 3. Apply second half of the window (W[L-k:L]) to the end (M[N_A-k : N_A])
        # This indexing handles both odd and even L correctly, ensuring the middle of the array is untouched.
        multiplier[N_A - self.k:] = self.window[self.L - self.k:]
        
        # 4. Taper the array
        tapered_array = array * multiplier
        
        # 5. Zero pad up to size
        return zero_pad(tapered_array, size)

class Extrap:
    """
    Implements a custom extrapolation function using a window /taper
    function applied to the ends of the padded region.
    """
    def __init__(self, taper_length: int, window_func):
        if taper_length <= 0:
            raise ValueError("Taper length must be positive.")
        self.L = taper_length
        # k is the length of the taper applied to one end (L // 2)
        self.k = self.L // 2
        self.window = window_func(self.L)

    def __call__(self, array: np.ndarray, size: int) -> np.ndarray:
        N_A = len(array)
        N_pad = size - N_A

        if N_pad < 0:
            raise ValueError("Target size must be greater than or equal to array length.")
        
        pad = np.zeros(N_pad, dtype=array.dtype)
        
        half = min(self.k, N_pad // 2)

        print (f'{self.k=} {self.L=} {N_A=} {N_pad=} {half=}')

        pad[:half] = self.window[self.L - half:] * array[-1]
        pad[N_pad - half:] = self.window[:half] * array[0]

        extrap_array = np.zeros(size, dtype=array.dtype)
        extrap_array[:N_A] = array
        extrap_array[N_A:] = pad
        return extrap_array;

def expo_function(size):
    '''
    A tapering / windowing function that is an exponential decay.

    The decay rate is 1 sample.
    '''
    half = size // 2
    decay = np.exp(-np.linspace(0, half, half, endpoint=False)) 
    arr = np.ones(size)
    arr[half:] = decay
    arr[:half] = decay[::-1]
    return arr

def gauss_function(size):
    '''
    A tapering / windowing function that is an exponential-squared decay.

    The decay rate is 1 sample.
    '''
    half = size // 2
    decay = np.exp(-np.pow(np.linspace(0, half, half, endpoint=False), 2))
    arr = np.ones(size)
    arr[-half:] = decay
    arr[:half] = decay[::-1]
    return arr

def sine_function(size):
    '''
    A tapering function that is a half wavelength of 0.5(sin+1)
    '''
    return 0.5*(1+np.sin(np.linspace(0, np.pi, size, endpoint=False)))

def linear_function(size):
    '''
    A tapering function that linearly goes to zero.
    '''
    half = size // 2
    decay = np.linspace(0.0, 1.0, half, endpoint=False)
    arr = np.ones(size)
    arr[-half:] = decay[::-1]
    arr[:half] = decay
    return arr


def taper_function(name: str, taper_length: int, taper_signal=True):
    '''
    Switchyard for picking a tapering function.
    '''
    if name.lower() == "none":
        return zero_pad
    wfs = dict(hann=np.hanning,
               hamm=np.hamming,
               blac=np.blackman,
               bart=np.bartlett,
               tria=np.bartlett, # triangle
               expo=expo_function,
               gaus=gauss_function,
               sine=sine_function,
               line=linear_function,
               )
    wf = wfs[name.lower()[:4]]
    if taper_signal:
        return Taper(taper_length, wf)
    return Extrap(taper_length, wf)


def Hann(taper_length: int, taper=True):
    """
    A Hann tapering window
    """
    if taper:
        return Taper(taper_length, np.hanning)
    return Extrap(taper_length, np.hanning)
    
def Hamming(taper_length: int, taper=True):
    """
    A Hamming tapering window
    """
    if taper:
        return Taper(taper_length, np.hamming)
    return Extrap(taper_length, np.hamming)

def Blackman(taper_length: int, taper=True):
    """
    A Blackman tapering window
    """
    if taper:
        Taper(taper_length, np.blackman)
    return Extrap(taper_length, np.blackman)
    
