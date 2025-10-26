import os
import sys
import click
import numpy as np

from scipy.signal import find_peaks

from . import signals
from . import decon
from . import plots
from .util import DataAttr, linear_size, zero_pad
from .filters import Filter, Lowpass, Highpass
from .filters import taper_function

@click.group(context_settings=dict(show_default = True,
                                   help_option_names=['-h', '--help']))
def cli():
    """
    Deconvolution Demo Package CLI
    """
    pass

@cli.command()
@click.option('--output', type=click.Path(), default=None, help='Path to save the plot image (e.g., output.png). If not provided, the plot is shown interactively.')
def roll(output):
    measure_size = 200
    measure_sigma = 2.0
    measure_mean = 10.0
    measure_peak1 = 10.0
    measure_peak2 = measure_size - measure_peak1

    kernel_size = 110
    kernel_sigma = 2.0
    kernel_mean = 100.0

    measure =  signals.gaussian(size=measure_size, mean=measure_mean, sigma=measure_sigma)
    measure += signals.gaussian(size=measure_size, mean=measure_size-measure_mean, sigma=measure_sigma, norm=2.0) 

    kernel = signals.gaussian(size=kernel_size, mean=kernel_mean, sigma=kernel_sigma)
    if np.sum(kernel) != 0:
        kernel /= np.sum(kernel)

    filt_func = Lowpass(scale=0.1, power=3.0)

    dsignal = decon.decon_pad(measure, kernel, zero_pad, filt_func)

    roll_size = kernel_size
    droll = np.roll(dsignal, roll_size)

    arrays = []

    # arrays.append(DataAttr(
    #     data=signal,
    #     attr={'name': 'signal', 'title': 'Signal'}
    # ))

    arrays.append(DataAttr(
        data=measure, 
        attr={'name': 'measure', 'title': f'Measure [{measure.shape[0]}], peaks @ {measure_peak1}, {measure_peak2}'}
    ))

    arrays.append(DataAttr(
        data=kernel,
        attr={'name': 'kernel', 'title': f'Kernel [{kernel.shape[0]}]'}
    ))

    roll_peaks, _ = find_peaks(dsignal, height=1.0, distance=2.0)
    arrays.append(DataAttr(
        data=dsignal, 
        attr={'name': 'decon', 'title': f'Decon [{dsignal.shape[0]}], points @ {roll_peaks}'}
    ))

    roll_peaks, _ = find_peaks(droll, height=1.0, distance=2.0)
    arrays.append(DataAttr(
        data=droll, 
        attr={'name': 'droll', 'title': f'Decon Rolled [{droll.shape[0]}] by [{roll_size}], peaks @ {roll_peaks}'}
    ))

    plots.plotn(arrays, output_path=output, waveform_logy=False)
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()
    

@cli.command()
@click.option('--signal-size', type=int, default=100, help='Size of the true signal array.')
@click.option('--signal-mean', type=float, default=30.0, help='Mean position of the true signal Gaussian.')
@click.option('--signal-sigma', type=float, default=5.0, help='Sigma (width) of the true signal Gaussian.')
@click.option('--kernel-size', type=int, default=21, help='Size of the kernel array.')
@click.option('--kernel-mean', type=float, default=10.0, help='Mean position of the kernel Gaussian.')
@click.option('--kernel-sigma', type=float, default=2.0, help='Sigma (width) of the kernel Gaussian.')
@click.option('--taper-name', type=click.Choice(['none', 'hann', 'hamming', 'blackman', 'bartlett', 'triangle', 'exponential', 'gauss', 'sine', 'linear']), default='none', help='Taper function to apply, none produces abrupt zero-padding.')
@click.option('--taper-length', type=int, default=10, help='Length of the taper.')
@click.option('--taper-signal', default=False, is_flag=True, help='If set, tapering is applied as a windowing of a signal, else (default) tapering is an extrapolation into the ends of a padded region.')
@click.option('--signal-is-measure', default=False, is_flag=True, 
              help='The generated signal is interpreted as the measure instead of forming measure via convolution of signal with kernel')
@click.option('--noise-rms', type=float, default=0.0, help='RMS value of white noise to add to the measured signal or measure.')
@click.option('--filter-name', type=click.Choice(['none', 'lowpass', 'highpass']), default='none', help='Type of frequency-space filter to apply during deconvolution.')
@click.option('--filter-scale', type=float, default=1.0, help='Scale parameter for the filter (e.g., cutoff frequency).')
@click.option('--filter-power', type=float, default=2.0, help='Power parameter for the filter steepness.')
@click.option('--filter-ignore-baseline', default=False, is_flag=True, help='If set, forces the zero-frequency component of the filter to zero.')
@click.option('--show-padded', default=False, is_flag=True, help='If set, show padded versions of signals instead of natural sizes.')
@click.option('--waveform-logy', default=False, is_flag=True, help='If set, show waveforms in log scale.')
@click.option('--output', type=click.Path(), default=None, help='Path to save the plot image (e.g., output.png). If not provided, the plot is shown interactively.')
def plot(signal_size, signal_mean, signal_sigma, kernel_size, kernel_mean, kernel_sigma, taper_name, taper_length, taper_signal, signal_is_measure, noise_rms, filter_name, filter_scale, filter_power, filter_ignore_baseline, show_padded, waveform_logy, output):
    """
    Illustrate various (de)convolutions using Gaussian signal and kernel and plot the results.
    """

    # We may call this command from org source blocks and want idempotent quick-exit behavior.
    if output and os.path.exists(output):
        # sys.stderr.write(f'file exists, remove to remake: {output}\n')
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
    pad_func = taper_function(taper_name, taper_length, taper_signal)

    # Padding happens inside decon().  In order to see the padded waveform and
    # spectrum, we may optionally pad the arrays here prior to adding them to
    # the list of arrays to plot.
    def maybe_pad(array, size):
        if show_padded:
            return pad_func(array, size)
        return array

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
        raise
        return

    # 6. Prepare DataAttr objects and Plot Results
    
    arrays = []

    if signal_is_measure:
        # Case 1: Plot 3 arrays (Measure, Kernel, Decon)
        
        size_decon = linear_size(signal_measured, kernel)

        # 1. Measure (signal_measured now holds the potentially noisy measure)
        arrays.append(DataAttr(
            data=maybe_pad(signal_measured, size_decon), 
            attr={'name': 'measure', 'title': 'Measure' + (' + Noise' if noise_rms > 0 else '')}
        ))
        
        # 2. Kernel
        arrays.append(DataAttr(
            data=maybe_pad(kernel, size_decon), 
            attr={'name': 'kernel', 'title': 'Kernel'}
        ))
        
        # 3. Deconvolved Result
        arrays.append(DataAttr(
            data=decon_result, 
            attr={'name': 'decon', 'title': 'Deconvolved Result'}
        ))
        
    else:
        # Case 2: Plot 4 arrays (True Signal, Kernel, Measured Signal, Decon)
        
        size_convo = linear_size(signal_true, kernel)
        size_decon = linear_size(size_convo, kernel)

        # 1. True Signal
        arrays.append(DataAttr(
            data=maybe_pad(signal_true, size_convo), 
            attr={'name': 'signal_true', 'title': 'True Signal (Input)'}
        ))
        
        # 2. Kernel
        arrays.append(DataAttr(
            data=maybe_pad(kernel, size_convo), 
            attr={'name': 'convo_kernel', 'title': 'Convolution Kernel'}
        ))
        
        # 3. Measured Signal (signal_measured now holds the potentially noisy convolution)
        arrays.append(DataAttr(
            data=maybe_pad(signal_measured, size_decon), 
            attr={'name': 'signal_measured', 'title': 'Measured Signal (Convolution)' + (' + Noise' if noise_rms > 0 else '')}
        ))
        
        arrays.append(DataAttr(
            data=maybe_pad(kernel, size_decon), 
            attr={'name': 'decon_kernel', 'title': 'Deconvolution Kernel'}
        ))

        # 4. Deconvolved Result
        arrays.append(DataAttr(
            data=decon_result, 
            attr={'name': 'decon', 'title': 'Deconvolved Result'}
        ))

    if filter_name != "none":
        arrays.insert(-1, DataAttr(
            data=maybe_pad(np.fft.fft(filt_func(decon_result.shape[0])).real, size_decon),
            attr=dict(name='filter', title='Frequency Filter')))

    plots.plotn(arrays, output_path=output, waveform_logy=waveform_logy)
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()

if __name__ == '__main__':
    cli()
