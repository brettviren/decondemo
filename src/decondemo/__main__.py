import click
import numpy as np

from . import signals
from . import decon
from . import plots
from .plots import DataAttr

@click.group(context_settings=dict(show_default = True,
                                   help_option_names=['-h', '--help']))
def cli():
    """
    Deconvolution Demo Package CLI
    """
    pass

@cli.command()
@click.option('--signal-size', type=int, default=100, help='Size of the true signal array.')
@click.option('--signal-mean', type=float, default=30.0, help='Mean position of the true signal Gaussian.')
@click.option('--signal-sigma', type=float, default=5.0, help='Sigma (width) of the true signal Gaussian.')
@click.option('--kernel-size', type=int, default=21, help='Size of the kernel array.')
@click.option('--kernel-mean', type=float, default=10.0, help='Mean position of the kernel Gaussian.')
@click.option('--kernel-sigma', type=float, default=2.0, help='Sigma (width) of the kernel Gaussian.')
@click.option('--window', type=click.Choice(['none', 'hann', 'hamming', 'blackman']), default='none', help='Window function to apply for tapering before padding.')
@click.option('--taper-length', type=int, default=10, help='Length of the window taper applied to the ends of the signal/kernel.')
@click.option('--signal-is-measure', default=False, is_flag=True, 
              help='The generated signal is interpreted as the measure instead of forming measure via convolution of signal with kernel')
@click.option('--output', type=click.Path(), default=None, help='Path to save the plot image (e.g., output.png). If not provided, the plot is shown interactively.')
def gaussian(signal_size, signal_mean, signal_sigma, kernel_size, kernel_mean, kernel_sigma, window, taper_length, signal_is_measure, output):
    """
    Perform both convolution and deconvolution of a Gaussian true signal and a Gaussian kernel. 
    """
    
    # 1. Generate True Signal and Kernel
    signal_true = signals.gaussian(size=signal_size, mean=signal_mean, sigma=signal_sigma)
    kernel = signals.gaussian(size=kernel_size, mean=kernel_mean, sigma=kernel_sigma)
    
    # Normalize kernel (PSF)
    if np.sum(kernel) != 0:
        kernel /= np.sum(kernel)
    
    # 2. Calculate Measured Signal (Convolution)
    # The measured signal is the convolution of the true signal and the kernel.
    # We use 'full' mode convolution, which results in an array of size N = len(signal_true) + len(kernel) - 1.
    if signal_is_measure:
        signal_measured = signal_true
    else:
        signal_measured = np.convolve(signal_true, kernel, mode='full')
    
    # 3. Determine Padding Function
    if window == 'none':
        pad_func = decon.zero_pad
        window_info = "None (Zero Padding)"
    elif window == 'hann':
        pad_func = decon.Hann(taper_length)
        window_info = f"Hann (Taper Length: {taper_length})"
    elif window == 'hamming':
        pad_func = decon.Hamming(taper_length)
        window_info = f"Hamming (Taper Length: {taper_length})"
    elif window == 'blackman':
        pad_func = decon.Blackman(taper_length)
        window_info = f"Blackman (Taper Length: {taper_length})"
    
    # 4. Perform Deconvolution using custom padding
    try:
        decon_result = decon.decon_pad(signal_measured, kernel, pad_func)
    except ValueError as e:
        click.echo(f"Error during deconvolution: {e}", err=True)
        return

    click.echo(f"--- Parameters ---")
    click.echo(f"True Signal Size: {len(signal_true)}")
    click.echo(f"Kernel Size: {len(kernel)}")
    click.echo(f"Measured Signal Size (N): {len(signal_measured)}")
    click.echo(f"Deconvolved Result Size: {len(decon_result)}")
    click.echo(f"Windowing Used: {window_info}")
    click.echo(f"------------------")

    # 5. Prepare DataAttr objects and Plot Results
    
    arrays = []
    
    if signal_is_measure:
        # Case 1: Plot 3 arrays (Measure, Kernel, Decon)
        
        # 1. Measure (signal_true is used as measure)
        arrays.append(DataAttr(
            data=signal_true, 
            attr={'name': 'measure', 'title': 'Measure'}
        ))
        
        # 2. Kernel
        arrays.append(DataAttr(
            data=kernel, 
            attr={'name': 'kernel', 'title': 'Kernel'}
        ))
        
        # 3. Deconvolved Result
        arrays.append(DataAttr(
            data=decon_result, 
            attr={'name': 'decon', 'title': 'Deconvolved Result'}
        ))
        
    else:
        # Case 2: Plot 4 arrays (True Signal, Kernel, Measured Signal, Decon)
        
        # 1. True Signal
        arrays.append(DataAttr(
            data=signal_true, 
            attr={'name': 'signal_true', 'title': 'True Signal (Input)'}
        ))
        
        # 2. Kernel
        arrays.append(DataAttr(
            data=kernel, 
            attr={'name': 'kernel', 'title': 'Kernel (PSF)'}
        ))
        
        # 3. Measured Signal
        arrays.append(DataAttr(
            data=signal_measured, 
            attr={'name': 'signal_measured', 'title': 'Measured Signal (Convolution)'}
        ))
        
        # 4. Deconvolved Result
        arrays.append(DataAttr(
            data=decon_result, 
            attr={'name': 'decon', 'title': 'Deconvolved Result'}
        ))

    plots.plotn(arrays, output_path=output)

if __name__ == '__main__':
    cli()
