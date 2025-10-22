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
