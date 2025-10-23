import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, List
from numpy.typing import ArrayLike

def plot3(measure: Sequence[float], kernel: Sequence[float], decon: Sequence[float], output_path: Optional[str] = None):
    """
    Plots the measure, kernel, and deconvolved result in three separate subplots.

    Args:
        measure: The input measure array
        kernel: The kernel array.
        decon: The deconvolved result array.
        output_path: If provided, saves the plot to this path instead of showing it interactively.
    """
    
    # Convert inputs to numpy arrays if they aren't already, for consistent indexing
    measure = np.asarray(measure)
    kernel = np.asarray(kernel)
    decon = np.asarray(decon)

    # Use sharex=True to that points are the same spacing on every plot
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    
    # Plot 1: Measure (True)
    axes[0].step(np.arange(len(measure)), measure, where='mid', label='True Measure')
    axes[0].set_title('1. Measure')
    axes[0].grid(True)
    
    # Plot 2: Kernel
    axes[1].step(np.arange(len(kernel)), kernel, where='mid', label='Kernel')
    axes[1].set_title('2. Kernel')
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
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    
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


def plotn(arrays: List[np.ndarray], output_path: Optional[str] = None):
    """
    Plots N arrays in an N x 3 grid, showing the array values, 
    the magnitude of its Fourier Transform, and the unrolled phase of its Fourier Transform.

    Metadata (title) is expected to be stored in array.dtype.metadata['title'].

    Args:
        arrays: A list of numpy arrays.
        output_path: If provided, saves the plot to this path instead of showing it interactively.
    """
    N = len(arrays)
    if N == 0:
        return

    # Create figure with N rows and 3 columns
    # sharex=True for the first column (interval plots)
    # sharex=False for the second and third columns (frequency plots)
    fig, axes = plt.subplots(N, 3, figsize=(15, 3 * N), sharex='col')
    
    # If N=1, axes is a 1D array of length 3. We wrap it for consistent indexing.
    if N == 1:
        axes = [axes]

    # Set sharex=False explicitly for columns 2 and 3 (index 1 and 2)
    # Matplotlib's subplots(sharex='col') only shares the x-axis within a column.
    # We need to ensure columns 2 and 3 are independent of column 1, and independent of each other.
    # Since we used sharex='col', column 1 is shared. Columns 2 and 3 are also shared within themselves.
    # We need to ensure columns 2 and 3 are NOT shared with column 1.
    # Since we used sharex='col', only axes[i, 0] share x-axis, axes[i, 1] share x-axis, axes[i, 2] share x-axis.
    # Let's manually manage sharing to ensure only column 1 is shared across rows.
    
    # Recreate subplots to manage sharing precisely:
    # We want axes[i, 0] to share x-axis with axes[j, 0] (Col 1 shared)
    # We want axes[i, 1] and axes[i, 2] to be independent of all others.
    
    # We will use the first column's axes (axes[:, 0]) to define the shared x-axis.
    fig, axes = plt.subplots(N, 3, figsize=(15, 3 * N))
    
    # Set up sharing for the first column
    if N > 1:
        for i in range(1, N):
            axes[i, 0].sharex(axes[0, 0])
    
    for i, array in enumerate(arrays):
        
        # Ensure array is numpy array
        array = np.asarray(array)
        L = len(array)
        
        # Get title from metadata
        metadata = getattr(array.dtype, 'metadata', {})
        title = metadata.get('title', f'Array {i+1}')
        
        # --- Column 1: Interval Plot ---
        ax1 = axes[i, 0]
        ax1.step(np.arange(L), array, where='mid')
        ax1.set_title(f"{title} - interval")
        ax1.grid(True)
        if i == N - 1:
            ax1.set_xlabel('Sample Index')

        # Calculate FFT
        fft_result = np.fft.fft(array)
        
        # Frequency axis (normalized to 0 to 1/dt, assuming dt=1)
        freq = np.fft.fftfreq(L)
        
        # --- Column 2: Fourier Amplitude ---
        ax2 = axes[i, 1]
        # We plot the magnitude of the FFT, usually centered (fftshift) for visualization
        fft_mag = np.abs(fft_result)
        
        # Plot centered spectrum
        ax2.plot(np.fft.fftshift(freq), np.fft.fftshift(fft_mag))
        ax2.set_title(f"{title} - Fourier amplitude")
        ax2.grid(True)
        if i == N - 1:
            ax2.set_xlabel('Frequency Index')
        
        # --- Column 3: Fourier Angle (Unrolled Phase) ---
        ax3 = axes[i, 2]
        # Calculate phase and unwrap it
        fft_phase = np.angle(fft_result)
        unwrapped_phase = np.unwrap(fft_phase)
        
        # Plot centered phase
        ax3.plot(np.fft.fftshift(freq), np.fft.fftshift(unwrapped_phase))
        ax3.set_title(f"{title} - Fourier angle")
        ax3.grid(True)
        if i == N - 1:
            ax3.set_xlabel('Frequency Index')
            
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
