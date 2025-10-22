import click
import numpy as np

from . import signals
from . import decon
from . import plots

@click.group()
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
@click.option('--output', type=click.Path(), default=None, help='Path to save the plot image (e.g., output.png). If not provided, the plot is shown interactively.')
def gaussian(signal_size, signal_mean, signal_sigma, kernel_size, kernel_mean, kernel_sigma, window, taper_length, output):
    """
    Generates a Gaussian signal and kernel, convolves them to create a measured signal, 
    performs deconvolution, and plots the results.
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
    # decon_pad(signal_measured, kernel, pad_func) attempts to recover signal_true
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

    # 5. Plot Results
    # We plot signal_true, kernel, and decon_result for comparison.
    plots.plot3(signal_true, kernel, decon_result, output_path=output)

if __name__ == '__main__':
    cli()
