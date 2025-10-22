import numpy as np

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
