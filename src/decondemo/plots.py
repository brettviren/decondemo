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

    # Note: sharex=True might be misleading if array lengths differ significantly.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    
    # Plot 1: Signal (True)
    axes[0].step(np.arange(len(signal)), signal, where='mid', label='True Signal')
    axes[0].set_title('1. True Signal')
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


def plot4(signal_true: Sequence[float], kernel: Sequence[float], signal_measured: Sequence[float], decon_result: Sequence[float], output_path: Optional[str] = None):
    """
    Plots the true signal, kernel, measured signal (convolution), and deconvolved result 
    in four separate subplots.

    Args:
        signal_true: The true, unblurred signal array.
        kernel: The kernel array.
        signal_measured: The measured signal (convolution of true signal and kernel).
        decon_result: The deconvolved result array.
        output_path: If provided, saves the plot to this path instead of showing it interactively.
    """
    
    # Convert inputs to numpy arrays
    signal_true = np.asarray(signal_true)
    kernel = np.asarray(kernel)
    signal_measured = np.asarray(signal_measured)
    decon_result = np.asarray(decon_result)

    # Use sharex=False so each plot displays its own index range correctly
    fig, axes = plt.subplots(4, 1, sharex=False, figsize=(10, 10))
    
    # Plot 1: True Signal
    axes[0].step(np.arange(len(signal_true)), signal_true, where='mid', label='True Signal')
    axes[0].set_title('1. True Signal (Input)')
    axes[0].grid(True)
    
    # Plot 2: Kernel
    axes[1].step(np.arange(len(kernel)), kernel, where='mid', label='Kernel')
    axes[1].set_title('2. Kernel (PSF)')
    axes[1].grid(True)
    
    # Plot 3: Measured Signal
    axes[2].step(np.arange(len(signal_measured)), signal_measured, where='mid', label='Measured Signal')
    axes[2].set_title('3. Measured Signal (Convolution)')
    axes[2].grid(True)
    
    # Plot 4: Deconvolved Result
    axes[3].step(np.arange(len(decon_result)), decon_result, where='mid', label='Deconvolved')
    axes[3].set_title('4. Deconvolved Result')
    axes[3].set_xlabel('Sample Index')
    axes[3].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
