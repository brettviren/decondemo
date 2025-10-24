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


def decon_pad(signal: np.ndarray, kernel: np.ndarray, pad_func: Callable[[np.ndarray, int], np.ndarray], filt_func: Callable[[int], np.ndarray] = Filter()) -> np.ndarray:
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
        filt_func: A function (N) -> filter array used to multiply the deconvolved FFT.

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


