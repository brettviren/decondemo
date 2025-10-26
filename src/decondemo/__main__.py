import os
import sys
import click
import numpy as np

from scipy.signal import find_peaks

from . import signals
from .convo import convo, decon
from . import plots
from .util import DataAttr, linear_size, tee_and_capture
from .filters import Filter, Lowpass, Highpass
from .filters import taper_function
from .chunked import (
    TimeSource, ExpoTime, UniformTime, Latch,
    ConvoFunc, PostOverlap, PreOverlap
)

@click.group(context_settings=dict(show_default = True,
                                   help_option_names=['-h', '--help']))
def cli():
    """
    Deconvolution Demo Package CLI
    """
    pass

def _get_filter_func(name, scale, power, ignore_baseline):
    filter_params = {
        'scale': scale,
        'power': power,
        'ignore_baseline': ignore_baseline
    }
    if name == 'none':
        return Filter(**filter_params)
    elif name == 'lowpass':
        return Lowpass(**filter_params)
    elif name == 'highpass':
        return Highpass(**filter_params)
    else:
        # Should be unreachable due to click.Choice validation
        raise ValueError(f"Unknown filter name: {name}")

def _get_pad_func(taper_name, taper_length, taper_signal):
    # taper_function handles mapping names like 'hann' to the correct callable
    return taper_function(taper_name, taper_length, taper_signal)

def _get_kernel(size, mean, sigma):
    kernel = signals.gaussian(size=size, mean=mean, sigma=sigma)
    # Normalize kernel (PSF)
    if np.sum(kernel) != 0:
        kernel /= np.sum(kernel)
    return kernel


@cli.command()
@click.option('--chunks', type=int, default=10, help='Number of chunks/time steps to generate (limit for TimeSource).')
@click.option('--rate', type=float, default=1.0, help='Rate parameter for time distribution.')
@click.option('--time-distribution', type=click.Choice(['expo', 'uniform']), default='expo', help='Time step distribution.')
@click.option('--sample-period', type=float, default=1.0, help='Sample period for Latch.')
@click.option('--start-time', type=float, default=0.0, help='Start time for Latch.')
@click.option('--chunk-size', type=int, default=10, help='Chunk size for Latch and Overlap nodes.')
# CONVO PARAMETERS
@click.option('--convo-kernel-size', type=int, default=21, help='Size of the convolution kernel array.')
@click.option('--convo-kernel-mean', type=float, default=10.0, help='Mean position of the convolution kernel Gaussian.')
@click.option('--convo-kernel-sigma', type=float, default=2.0, help='Sigma (width) of the convolution kernel Gaussian.')
@click.option('--convo-taper-name', type=click.Choice(['none', 'hann', 'hamming', 'blackman', 'bartlett', 'triangle', 'exponential', 'gauss', 'sine', 'linear']), default='none', help='Taper function for convolution padding.')
@click.option('--convo-taper-length', type=int, default=10, help='Length of the convolution taper.')
@click.option('--convo-taper-signal', default=False, is_flag=True, help='If set, convolution tapering is applied as a windowing of a signal, else (default) tapering is an extrapolation.')
@click.option('--convo-filter-name', type=click.Choice(['none', 'lowpass', 'highpass']), default='none', help='Type of frequency-space filter to apply during convolution.')
@click.option('--convo-filter-scale', type=float, default=1.0, help='Scale parameter for the convolution filter.')
@click.option('--convo-filter-power', type=float, default=2.0, help='Power parameter for the convolution filter steepness.')
@click.option('--convo-filter-ignore-baseline', default=False, is_flag=True, help='If set, forces the zero-frequency component of the convolution filter to zero.')
# DECON PARAMETERS
@click.option('--decon-kernel-size', type=int, default=21, help='Size of the deconvolution kernel array.')
@click.option('--decon-kernel-mean', type=float, default=10.0, help='Mean position of the deconvolution kernel Gaussian.')
@click.option('--decon-kernel-sigma', type=float, default=2.0, help='Sigma (width) of the deconvolution kernel Gaussian.')
@click.option('--decon-taper-name', type=click.Choice(['none', 'hann', 'hamming', 'blackman', 'bartlett', 'triangle', 'exponential', 'gauss', 'sine', 'linear']), default='none', help='Taper function for deconvolution padding.')
@click.option('--decon-taper-length', type=int, default=10, help='Length of the deconvolution taper.')
@click.option('--decon-taper-signal', default=False, is_flag=True, help='If set, deconvolution tapering is applied as a windowing of a signal, else (default) tapering is an extrapolation.')
@click.option('--decon-filter-name', type=click.Choice(['none', 'lowpass', 'highpass']), default='lowpass', help='Type of frequency-space filter to apply during deconvolution.')
@click.option('--decon-filter-scale', type=float, default=0.1, help='Scale parameter for the deconvolution filter.')
@click.option('--decon-filter-power', type=float, default=3.0, help='Power parameter for the deconvolution filter steepness.')
@click.option('--decon-filter-ignore-baseline', default=False, is_flag=True, help='If set, forces the zero-frequency component of the deconvolution filter to zero.')
@click.option('--output', type=click.Path(), default=None, help='Path to save the plot image (e.g., output.png). If not provided, the plot is shown interactively.')
def chunked(
    chunks, rate, time_distribution, sample_period, start_time, chunk_size,
    convo_kernel_size, convo_kernel_mean, convo_kernel_sigma,
    convo_taper_name, convo_taper_length, convo_taper_signal,
    convo_filter_name, convo_filter_scale, convo_filter_power, convo_filter_ignore_baseline,
    decon_kernel_size, decon_kernel_mean, decon_kernel_sigma,
    decon_taper_name, decon_taper_length, decon_taper_signal,
    decon_filter_name, decon_filter_scale, decon_filter_power, decon_filter_ignore_baseline,
    output
):
    """
    Runs a streaming chunked processing pipeline:

      TimeSource -> Latch -> Convo -> Decon.
    """
    
    # 1. Time Source Setup
    if time_distribution == 'expo':
        step = ExpoTime(rate=rate)
    else:
        step = UniformTime(rate=rate)
        
    time_source = TimeSource(step=step, start=start_time, limit=chunks)
    
    # 2. Latch Setup
    latch = Latch(sample_period=sample_period, chunk_size=chunk_size, start_time=start_time)
    
    # 3. Convo Setup (PostOverlap)
    convo_kernel = _get_kernel(convo_kernel_size, convo_kernel_mean, convo_kernel_sigma)
    convo_pad_func = _get_pad_func(convo_taper_name, convo_taper_length, convo_taper_signal)
    convo_filt_func = _get_filter_func(convo_filter_name, convo_filter_scale, convo_filter_power, convo_filter_ignore_baseline)
    
    convo_func_obj = ConvoFunc(
        kernel=convo_kernel,
        pad_func=convo_pad_func,
        filt_func=convo_filt_func,
        invert=False
    )
    post_overlap = PostOverlap(transform=convo_func_obj, chunk_size=chunk_size)
    
    # 4. Decon Setup (PreOverlap)
    decon_kernel = _get_kernel(decon_kernel_size, decon_kernel_mean, decon_kernel_sigma)
    decon_pad_func = _get_pad_func(decon_taper_name, decon_taper_length, decon_taper_signal)
    decon_filt_func = _get_filter_func(decon_filter_name, decon_filter_scale, decon_filter_power, decon_filter_ignore_baseline)
    
    decon_func_obj = ConvoFunc(
        kernel=decon_kernel,
        pad_func=decon_pad_func,
        filt_func=decon_filt_func,
        invert=True
    )
    pre_overlap = PreOverlap(transform=decon_func_obj, chunk_size=chunk_size)
    
    # 5. Pipeline Execution and Capture
    
    # TimeSource -> Latch (Impulse train)
    impulse_train_source = latch(time_source())
    
    # Capture Latch output (Impulse Train Chunks)
    latch_chunks = []
    tapped_impulse_train = tee_and_capture(impulse_train_source, latch_chunks)
    
    # Latch -> PostOverlap (Convolution)
    convolved_source = post_overlap(tapped_impulse_train)
    
    # Capture Convo output (Convolved Chunks)
    convo_chunks = []
    tapped_convolved = tee_and_capture(convolved_source, convo_chunks)
    
    # PostOverlap -> PreOverlap (Deconvolution)
    decon_chunks = []
    deconvolved_results = tee_and_capture(pre_overlap(tapped_convolved), decon_chunks)
    
    # 6. Collect results (Consumes the generator, filling decon_chunks)
    list(deconvolved_results)
    
    print(f'{len(latch_chunks)} latched, {len(convo_chunks)} convos, {len(decon_chunks)} decons')

    # 7. Plotting/Output
    
    if not decon_chunks:
        click.echo("Pipeline produced no output chunks.", err=True)
        return

    arrays = []
    
    # A. Latch Output (Input to Convo)
    if latch_chunks:
        for i, chunk in enumerate(latch_chunks):
            arrays.append(DataAttr(
                data=chunk,
                attr={'name': f'latch_chunk_{i}', 'title': f'1. Latch Output (Chunk {i}/{len(latch_chunks)})'}
            ))

    # B. Convo Output (Input to Decon)
    if convo_chunks:
        for i, chunk in enumerate(convo_chunks):
            arrays.append(DataAttr(
                data=chunk,
                attr={'name': f'convo_chunk_{i}', 'title': f'2. Convo Output (Chunk {i}/{len(convo_chunks)})'}
            ))

    # C. Decon Output (Final Result)
    if decon_chunks:
        for i, chunk in enumerate(decon_chunks):
            arrays.append(DataAttr(
                data=chunk,
                attr={'name': f'decon_chunk_{i}', 'title': f'3. Decon Output (Chunk {i}/{len(decon_chunks)})'}
            ))
    
    plots.plotn(arrays, output_path=output, waveform_logy=False)
    
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()
    
    click.echo(f"Successfully processed {chunks} time steps into {len(decon_chunks)} output chunks.")


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

    dsignal = decon(measure, kernel, filt_func=filt_func)

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
        signal_measured = convo(signal_true, kernel)
        
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
        decon_result = decon(signal_measured, kernel, pad_func, filt_func=filt_func)
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
