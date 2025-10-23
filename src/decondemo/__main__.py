import os
import sys
import click
import numpy as np

from . import signals
from . import decon
from . import plots
from .util import DataAttr
from .filters import Filter, Lowpass, Highpass # Import filter classes

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
@click.option('--noise-rms', type=float, default=0.0, help='RMS value of white noise to add to the measured signal or measure.')
@click.option('--filter-name', type=click.Choice(['none', 'lowpass', 'highpass']), default='none', help='Type of frequency-space filter to apply during deconvolution.')
@click.option('--filter-scale', type=float, default=1.0, help='Scale parameter for the filter (e.g., cutoff frequency).')
@click.option('--filter-power', type=float, default=2.0, help='Power parameter for the filter steepness.')
@click.option('--filter-ignore-baseline', default=False, is_flag=True, help='If set, forces the zero-frequency component of the filter to zero.')
@click.option('--output', type=click.Path(), default=None, help='Path to save the plot image (e.g., output.png). If not provided, the plot is shown interactively.')
def plot(signal_size, signal_mean, signal_sigma, kernel_size, kernel_mean, kernel_sigma, window, taper_length, signal_is_measure, noise_rms, filter_name, filter_scale, filter_power, filter_ignore_baseline, output):
    """
    Perform both convolution and deconvolution of a Gaussian true signal and a Gaussian kernel. 
    """
    if output and os.path.exists(output):
        sys.stderr.write(f'file exists, remove to remake: {output}\n')
        sys.stdout.write(output)
        sys.stdout.flush()
        return
    
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
        
    # 2b. Apply Noise if requested
    if noise_rms > 0.0:
        if signal_is_measure:
            # Noise applied directly to the measure (signal_true)
            noise = signals.white_noise(size=len(signal_true), rms=noise_rms)
            signal_measured = signal_true + noise
            noise_target = "Measure"
        else:
            # Noise applied to the convolved signal (signal_measured)
            noise = signals.white_noise(size=len(signal_measured), rms=noise_rms)
            signal_measured = signal_measured + noise
            noise_target = "Measured Signal"
    else:
        noise_target = "None"
    
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
    
    # 4. Determine Filter Function
    filter_params = {
        'scale': filter_scale,
        'power': filter_power,
        'ignore_baseline': filter_ignore_baseline
    }
    
    if filter_name == 'none':
        filt_func = Filter(**filter_params)
    elif filter_name == 'lowpass':
        filt_func = Lowpass(**filter_params)
    elif filter_name == 'highpass':
        filt_func = Highpass(**filter_params)
    else:
        # Should be unreachable due to click.Choice validation
        raise ValueError(f"Unknown filter name: {filter_name}")

    # 5. Perform Deconvolution using custom padding and filter
    try:
        decon_result = decon.decon_pad(signal_measured, kernel, pad_func, filt_func=filt_func)
    except ValueError as e:
        click.echo(f"Error during deconvolution: {e}", err=True)
        return

    # 6. Prepare DataAttr objects and Plot Results
    
    arrays = []
    
    if signal_is_measure:
        # Case 1: Plot 3 arrays (Measure, Kernel, Decon)
        
        # 1. Measure (signal_measured now holds the potentially noisy measure)
        arrays.append(DataAttr(
            data=signal_measured, 
            attr={'name': 'measure', 'title': 'Measure' + (' + Noise' if noise_rms > 0 else '')}
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
        
        # 3. Measured Signal (signal_measured now holds the potentially noisy convolution)
        arrays.append(DataAttr(
            data=signal_measured, 
            attr={'name': 'signal_measured', 'title': 'Measured Signal (Convolution)' + (' + Noise' if noise_rms > 0 else '')}
        ))
        
        # 4. Deconvolved Result
        arrays.append(DataAttr(
            data=decon_result, 
            attr={'name': 'decon', 'title': 'Deconvolved Result'}
        ))

    if filter_name != "none":
        arrays.insert(-1, DataAttr(
            data=np.fft.fft(filt_func(decon_result.shape[0])).real,
            attr=dict(name='filter', title='Frequency Filter')))

    plots.plotn(arrays, output_path=output)
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()

if __name__ == '__main__':
    cli()
