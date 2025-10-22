import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional

def plot3(signal: Sequence[float], kernel: Sequence[float], decon: Sequence[float], output_path: Optional[str] = None):
    """
    Plots the signal (True), kernel, and deconvolved result in three separate subplots.

    Args:
        signal: The input signal array (representing the true, unblurred signal).
        kernel: The kernel array.
        decon: The deconvolved result array.
        output_path: If provided, saves the plot to this path instead of showing it interactively.
    """
    
    # Convert inputs to numpy arrays if they aren't already, for consistent indexing
    signal = np.asarray(signal)
    kernel = np.asarray(kernel)
    decon = np.asarray(decon)

    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(10, 8))
    
    # Plot 1: Signal (True)
    axes[0].step(np.arange(len(signal)), signal, where='mid', label='True Signal')
    axes[0].set_title('1. True Signal (Input to Blurring)')
    axes[0].grid(True)
    
    # Plot 2: Kernel
    axes[1].step(np.arange(len(kernel)), kernel, where='mid', label='Kernel')
    axes[1].set_title('2. Kernel (PSF)')
    axes[1].grid(True)
    
    # Plot 3: Deconvolved Result
    axes[2].step(np.arange(len(decon)), decon, where='mid', label='Deconvolved')
    axes[2].set_title('3. Deconvolved Result')
    axes[2].set_xlabel('Sample Index')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close(fig) # Close the figure after saving
    else:
        plt.show()
