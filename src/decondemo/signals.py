import numpy as np

def gaussian(size: int, mean: float, sigma: float) -> np.ndarray:
    """
    Generates a sampled Gaussian function.

    Args:
        size: The size of the array to return (number of sample points).
        mean: The location of the peak of the Gaussian (in sample points).
        sigma: The Gaussian width (standard deviation, in sample points).

    Returns:
        A numpy array containing the sampled Gaussian shape.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    x = np.arange(size)
    # Calculate the Gaussian shape: exp(-(x - mean)^2 / (2 * sigma^2))
    return np.exp(-0.5 * ((x - mean) / sigma)**2)

def white_noise(size: int, rms: float = 1.0) -> np.ndarray:
    """
    Generates an array of white noise (Gaussian distribution) with a specified RMS value.

    Args:
        size: The size of the array to return.
        rms: The desired root-mean-square value of the noise. This corresponds 
             to the standard deviation since the mean is zero.

    Returns:
        A numpy array containing the white noise.
    """
    if rms < 0:
        raise ValueError("RMS must be non-negative.")
        
    # Generate standard normal noise (mean=0, std=1) and scale by desired RMS
    return np.random.randn(size) * rms
