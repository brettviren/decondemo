import numpy as np
from typing import Callable

from .filters import Filter        # base filter is identity filter
from .util import fftfreq as pos_fftfreq

def decon(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Performs deconvolution using the DFT method.

    The process involves:
    1. Determining the required size N for zero-padding (N = len(signal) + len(kernel) - 1).
    2. Applying FFT to the zero-padded signal and kernel.
    3. Dividing the signal FFT by the kernel FFT.
    4. Applying IFFT to the result.
    5. Returning the real part of the result.

    Args:
        signal: The measured signal array.
        kernel: The kernel (point spread function) array.

    Returns:
        The deconvolved result (real part).
    """
    # 1. Calculate the required size N for zero-padding
    N = len(signal) + len(kernel) - 1

    # 2. Apply FFT to zero-padded arrays (np.fft.fft handles padding via n=N)
    signal_fft = np.fft.fft(signal, n=N)
    kernel_fft = np.fft.fft(kernel, n=N)

    # 3. Divide in Fourier space
    # Note: In practical applications, regularization (e.g., Wiener filtering)
    # is often required here to handle noise and zeros in kernel_fft.
    # For this simple demo, we perform direct division as requested.
    decon_fft = signal_fft / kernel_fft

    # 4. Apply IFFT
    result = np.fft.ifft(decon_fft)

    # 5. Return the real part
    return result.real

def zero_pad(array: np.ndarray, size: int) -> np.ndarray:
    """
    Pads an array with zeros up to the specified size.
    """
    if size < len(array):
        raise ValueError("Target size must be greater than or equal to array length.")
    
    padded_array = np.zeros(size, dtype=array.dtype)
    padded_array[:len(array)] = array
    return padded_array


def decon_pad(signal: np.ndarray, kernel: np.ndarray, pad_func: Callable[[np.ndarray, int], np.ndarray], filt_func: Callable[[np.ndarray], np.ndarray] = Filter()) -> np.ndarray:
    """
    Performs deconvolution using the DFT method with custom padding.

    The process involves:
    1. Determining the required size N for padding (N = len(signal) + len(kernel) - 1).
    2. Applying the custom padding function to the signal and kernel up to size N.
    3. Applying FFT to the padded arrays (without using the n=N argument).
    4. Dividing the signal FFT by the kernel FFT.
    5. Applying IFFT to the result.
    6. Returning the real part of the result.

    Args:
        signal: The measured signal array.
        kernel: The kernel (point spread function) array.
        pad_func: A function (array, size) -> padded_array used for padding.
        filt_func: A function (array) -> filter array to filter the decon.

    Returns:
        The deconvolved result (real part).
    """
    # 1. Calculate the required size N for padding
    N = len(signal) + len(kernel) - 1

    # 2. Apply custom padding
    padded_signal = pad_func(signal, N)
    padded_kernel = pad_func(kernel, N)

    # 3. Apply FFT to padded arrays
    signal_fft = np.fft.fft(padded_signal)
    kernel_fft = np.fft.fft(padded_kernel)

    filter_fft = filt_func(N)

    # 4. Divide in Fourier space with multiplicative filter
    decon_fft = signal_fft * filter_fft / kernel_fft

    # 5. Apply IFFT
    result = np.fft.ifft(decon_fft)

    # 6. Return the real part
    return result.real


class Hann:
    """
    Implements a custom padding function using a Hann window taper applied 
    to the start and end of the array, followed by zero-padding.
    """
    def __init__(self, taper_length: int):
        if taper_length <= 0:
            raise ValueError("Taper length must be positive.")
        self.L = taper_length
        # k is the length of the taper applied to one end (L // 2)
        self.k = self.L // 2
        self.window = np.hanning(self.L)

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

class Hamming:
    """
    Implements a custom padding function using a Hamming window taper applied 
    to the start and end of the array, followed by zero-padding.
    """
    def __init__(self, taper_length: int):
        if taper_length <= 0:
            raise ValueError("Taper length must be positive.")
        self.L = taper_length
        self.k = self.L // 2
        self.window = np.hamming(self.L)

    def __call__(self, array: np.ndarray, size: int) -> np.ndarray:
        N_A = len(array)
        
        if size < N_A:
            raise ValueError("Target size must be greater than or equal to array length.")
        
        if N_A < self.L:
            raise ValueError(f"Array length ({N_A}) must be greater than or equal to taper length ({self.L}).")

        multiplier = np.ones(N_A, dtype=array.dtype)
        
        # Apply first half
        multiplier[:self.k] = self.window[:self.k]
        
        # Apply second half
        multiplier[N_A - self.k:] = self.window[self.L - self.k:]
        
        tapered_array = array * multiplier
        
        return zero_pad(tapered_array, size)

class Blackman:
    """
    Implements a custom padding function using a Blackman window taper applied 
    to the start and end of the array, followed by zero-padding.
    """
    def __init__(self, taper_length: int):
        if taper_length <= 0:
            raise ValueError("Taper length must be positive.")
        self.L = taper_length
        self.k = self.L // 2
        self.window = np.blackman(self.L)

    def __call__(self, array: np.ndarray, size: int) -> np.ndarray:
        N_A = len(array)
        
        if size < N_A:
            raise ValueError("Target size must be greater than or equal to array length.")
        
        if N_A < self.L:
            raise ValueError(f"Array length ({N_A}) must be greater than or equal to taper length ({self.L}).")

        multiplier = np.ones(N_A, dtype=array.dtype)
        
        # Apply first half
        multiplier[:self.k] = self.window[:self.k]
        
        # Apply second half
        multiplier[N_A - self.k:] = self.window[self.L - self.k:]
        
        tapered_array = array * multiplier
        
        return zero_pad(tapered_array, size)
