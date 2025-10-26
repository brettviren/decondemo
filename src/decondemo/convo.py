import numpy as np
from typing import Callable

from .filters import Filter        # base filter is identity filter
from .util import fftfreq as pos_fftfreq, zero_pad


def decon(signal: np.ndarray, kernel: np.ndarray,
          pad_func: Callable[[np.ndarray, int], np.ndarray] = zero_pad,
          filt_func: Callable[[int], np.ndarray] = Filter()) -> np.ndarray:
    return convo(signal, kernel, pad_func, filt_func=filt_func, invert=True);

def convo(signal: np.ndarray, kernel: np.ndarray,
          pad_func: Callable[[np.ndarray, int], np.ndarray] = zero_pad,
          filt_func: Callable[[int], np.ndarray] = Filter(),
          invert=False) -> np.ndarray:
    """
    Performs convolution using the DFT method with custom padding.

    The process involves:
    1. Determining the required size N for padding (N = len(signal) + len(kernel) - 1).
    2. Applying the custom padding function to the signal and kernel up to size N.
    3. Applying FFT to the padded arrays (without using the n=N argument).
    4. Multiplying the signal FFT by the kernel FFT (dividing if invert=True).
    5. Multiplying the filter.
    6. Applying IFFT to the result.
    7. Returning the real part of the result.

    Args:
        signal: The measured signal array.
        kernel: The kernel (point spread function) array.
        pad_func: A function (array, size) -> padded_array used for padding.
        filt_func: A Fourier space function (N) multiplied to the spectrum.
        invert: A boolean, if true perform deconvolution of the kernel.
    Returns:
        The convolved result (real part).
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
    if invert:
        result_fft = signal_fft * filter_fft / kernel_fft
    else:
        result_fft = signal_fft * filter_fft * kernel_fft

    # 5. Apply IFFT
    result = np.fft.ifft(result_fft)

    # 6. Return the real part
    return result.real


