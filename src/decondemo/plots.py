import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

def plot3(signal: Sequence[float], kernel: Sequence[float], decon: Sequence[float]):
    """
    Plots the signal, kernel, and deconvolved result in three separate subplots.

    Args:
        signal: The input signal array.
        kernel: The kernel array.
        decon: The deconvolved result array.
    """
    
    # Convert inputs to numpy arrays if they aren't already, for consistent indexing
    signal = np.asarray(signal)
    kernel = np.asarray(kernel)
    decon = np.asarray(decon)

    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(10, 8))
    
    # Plot 1: Signal
    axes[0].step(np.arange(len(signal)), signal, where='mid', label='Signal')
    axes[0].set_title('Input Signal')
    axes[0].grid(True)
    
    # Plot 2: Kernel
    axes[1].step(np.arange(len(kernel)), kernel, where='mid', label='Kernel')
    axes[1].set_title('Kernel (PSF)')
    axes[1].grid(True)
    
    # Plot 3: Deconvolved Result
    axes[2].step(np.arange(len(decon)), decon, where='mid', label='Deconvolved')
    axes[2].set_title('Deconvolved Result')
    axes[2].set_xlabel('Sample Index')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
