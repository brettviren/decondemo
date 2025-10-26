import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, List, Any, Union
from numpy.typing import ArrayLike

from .util import DataAttr
from .util import fftshift as no_fftshift
from .util import fftfreq as pos_fftfreq

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


def plotn(arrays: List[DataAttr], output_path: Optional[str] = None, waveform_logy=False):
    """
    Plots N arrays (wrapped in DataAttr objects) in an N x 3 grid, 
    showing the array values, the magnitude of its Fourier Transform, 
    and the unrolled phase of its Fourier Transform.

    Metadata (title) is expected to be stored in array_wrapper.attr['title'].

    Args:
        arrays: A list of DataAttr instances, each containing a numpy array and metadata.
        output_path: If provided, saves the plot to this path instead of showing it interactively.
    """
    N = len(arrays)
    if N == 0:
        return

    # Create figure with N rows and 3 columns
    fig, axes = plt.subplots(N, 3, figsize=(15, 3 * N))
    
    # Ensure axes is 2D even if N=1 for consistent indexing axes[i, j]
    if N == 1:
        axes = np.array([axes])
    
    # Set up sharing for the first column (Time domain plots)
    if N > 1:
        for i in range(1, N):
            axes[i, 0].sharex(axes[0, 0])
    
    # fftshift = np.fft.fftshift
    # fftfreq = np.fft.fftfreq
    fftshift = no_fftshift
    fftfreq = pos_fftfreq

    for i, array_wrapper in enumerate(arrays):
        
        array = array_wrapper.data
        L = len(array)
        
        # Get title from metadata dictionary (.attr)
        title = array_wrapper.attr.get('title', f'Array {i+1}')
        
        # --- Column 1: Interval Plot ---
        ax1 = axes[i, 0]
        ax1.step(np.arange(L), array, where='mid')
        ax1.set_title(f"{title} - interval")
        ax1.grid(True)
        if waveform_logy:
            ax1.set_yscale('log')
        if i == N - 1:
            ax1.set_xlabel('Sample Index')

        # Calculate FFT
        fft_result = np.fft.fft(array)
        
        # Frequency axis (normalized to 0 to 1/dt, assuming dt=1)
        freq = fftfreq(L)
        
        # --- Column 2: Fourier Amplitude ---
        ax2 = axes[i, 1]
        # We plot the magnitude of the FFT
        fft_mag = np.abs(fft_result)
        
        # Plot centered spectrum
        ax2.plot(fftshift(freq), fftshift(fft_mag))
        ax2.set_title(f"{title} - Fourier amplitude (log)")
        ax2.grid(True)
        ax2.set_yscale('log')
        if i == N - 1:
            ax2.set_xlabel('Frequency Index')
        
        # --- Column 3: Fourier Angle (Unrolled Phase) ---
        ax3 = axes[i, 2]
        # Calculate phase and unwrap it
        fft_phase = np.angle(fft_result)
        unwrapped_phase = np.unwrap(fft_phase)
        
        # Plot centered phase
        ax3.plot(fftshift(freq), fftshift(unwrapped_phase))
        ax3.set_title(f"{title} - Fourier angle (rad)")
        ax3.grid(True)
        if i == N - 1:
            ax3.set_xlabel('Frequency Index')
            
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


def plotcn(*columns: List[Union[DataAttr, ArrayLike]], output_path: Optional[str] = None):
    """
    Plots multiple columns of waveform arrays (DataAttr objects or raw arrays).
    Only the interval plot is included.

    Args:
        *columns: Each argument is a list of DataAttr/ArrayLike objects representing a column.
        output_path: If provided, saves the plot to this path instead of showing it interactively.
    """
    N_cols = len(columns)
    if N_cols == 0:
        return

    # Determine the maximum number of rows needed
    N_rows = max(len(col) for col in columns) if columns else 0
    if N_rows == 0:
        return

    # Create figure with N_rows rows and N_cols columns
    fig, axes = plt.subplots(N_rows, N_cols, sharex=False, figsize=(5 * N_cols, 3 * N_rows))

    # Ensure axes is 2D even if N_rows=1 or N_cols=1
    if N_rows == 1 and N_cols == 1:
        axes = np.array([[axes]])
    elif N_rows == 1:
        axes = np.array([axes])
    elif N_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for j, column in enumerate(columns):
        for i, item in enumerate(column):
            ax = axes[i, j]
            
            # Ensure item is a DataAttr object
            if isinstance(item, DataAttr):
                array_wrapper = item
            else:
                # Wrap raw array in DataAttr with default title
                array_wrapper = DataAttr(data=np.asarray(item), attr={'title': f'Column {j+1} Row {i+1}'})

            array = array_wrapper.data
            L = len(array)
            title = array_wrapper.attr.get('title', f'Chunk {i}')

            print(f'{i=} {j=} {title} {np.sum(array)}')

            # Plotting the interval
            ax.step(np.arange(L), array, where='mid')
            ax.set_title(title)
            ax.grid(True)
            
            if i == N_rows - 1:
                ax.set_xlabel('Sample Index')
            
        # Hide unused axes if the column is shorter than N_rows
        for k in range(len(column), N_rows):
            axes[k, j].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
