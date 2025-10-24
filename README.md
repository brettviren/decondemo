- [Getting started](#org9a52953)
- [Demo overview](#org0782c01)
- [Matched convolution and deconvolution](#orgf4b6865)
- [Unmatched convolution and deconvolution](#org3a2ed10)
- [Shifts and cycles](#org76d514d)
- [Filtered case](#org58e0ef5)
- [Adding noise](#org160615b)
- [Noise filters](#org00c742e)
- [Spectral leakage](#orgddb9f7f)
- [Mitigation with tapering](#org3874aae)
- [Mitigation with extrapolation](#orge9a5867)



<a id="org9a52953"></a>

# Getting started

```
$ uv tool install git+https://github.com/brettviren/decondemo
$ decondemo --help
```


<a id="org0782c01"></a>

# Demo overview

Below we go through the demo in a series of steps. Each step gives a command line to run, its output plot and some discussion.

We use **signal** to mean any discreet sampling regardless of its nature ("true signal" is given a special definition). We will talk about **waveforms** when referring to a real-valued, **interval-space** representation (sampling) of a signal and **spectrum** when referring to its complex-valued, Fourier space representation. These two representations are equivalent in terms of the information they embody.

We identify a few types of signals and give them names:

-   **S:** A "true signal", typically not directly observable in nature.
-   **K:** A "kernel" used for convolution or deconvolution.
-   **M:** A "measure" of a true signal typically through some response kernel and potentially in the presence of noise.
-   **N:** A measure of pure "noise" lacking any "true signal".
-   **D:** A "deconvolved signal" or "recovered signal" or "signal estimate"
-   **F:** A "filter" applied as part of a (de)convolution.

We will say a size of a signal is Nx where "x" is the signal name. Nk is the size of a kernel.

Expressed in Fourier-space, a detector response is modeled essentially as:

```
M = S*K + N
```

A filtered deconvolution is defined as:

```
D = M/K
```

When the convolution and deconvolution kernels are identical

```
D = M/K = (S*K)/K = S
```

To deal with FP error and noise, a Filter may be introduce:

```
D = F(M/K) = F(S*K)/K = FS
```

Convolution and deconvolution are performed with the "DFT method" (see [Convolution theorem](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Convolution_theorem_duality)). In order to avoid cyclic artifacts, both arrays are padded to achieve **linear size**. The above is more properly written as this pseudo code.

```
Nm = linear_size(Ns, Nk) # = Ns+Nk-1
S -> Spad = pad(S, Nm)
K -> Kpad = pad(K, Nm)
M = Spad*Kpad
```

Padding for deconvolution is essentially the same. However, the filter F, despite being in the numerator, is merely multiplied and does not lead to additional padding. F is a real-valued function in the Fourier domain sampled with the sampling shared by S and K.


<a id="orgf4b6865"></a>

# Matched convolution and deconvolution

We start with the default demo output:

```sh
uv run decondemo plot --output basic-convo-decon.svg
```

<div class="html" id="org12d7a5c">

<div id="orgaf7ee1f" class="figure">
<p><img src="basic-convo-decon.svg" alt="basic-convo-decon.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

This plot and the ones that follow have three columns:

1.  A **waveform** in interval space (eg "time domain").
2.  It's **spectrum amplitude** in Fourier space.
3.  It's **spectrum angle** (unwrapped phase) in Fourier space.

The sample period in this demo is always a unitless 1.0 as is the sample frequency. The Nyquist frequency is thus always 0.5. The number of samples in interval and Fourier space are (of course) the same but differ between different signals. As the interval representation is real, the Fourier representation has Hermitian symmetry.

Some things to note about these plots

-   An identical kernel K is used in both the convolution to form M and the deconvolution to achieve D. To within floating point errors, D is exactly the true signal S.

-   The measure M takes linear size of S and K. The recovered signal D takes yet larger linear size of M and K.

-   The peak of the measure M is **shifted forwards** relative to the peaks in both the signal S and kernel K. This is due to M being produced by a **convolution** with the kernel K in the **numerator**. The location of M-peak is equal to the sum of the locations of the S-peak and K-peak.

-   The recovered signal D is **shifted backwards** relative to the peak in the measure M. This is due to offset peak in K and K being in the **denominator** of the **(de)convolution**. The location of D-peak is equal to the location of M-peak less the location of K-peak.

-   We see in the spectrum of D some high-frequency energy. This arises from a combination of floating point errors and dividing by small values of K in the deconvolution. Later, we will address this with a **filter** below.


<a id="org3a2ed10"></a>

# Unmatched convolution and deconvolution

Now consider a measure M that that is **not** formed as a convolution S\*K but is still deconvolved with K. The demo shows this by forming M directly as a Gaussian shape<sup><a id="fnr.twok" class="footref" href="#fn.twok" role="doc-backlink">1</a></sup>.

```sh
uv run decondemo plot --signal-is-measure --output basic-decon.svg
```

<div class="html" id="org4528627">

<div id="orgf5c1535" class="figure">
<p><img src="basic-decon.svg" alt="basic-decon.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

Things to note

-   As in the matched-kernel demo, the recovered signal D = M/K is **shifted backward** in relative to the measure M.

-   D gains high-frequency "wiggles". They are due to the kernel K not matching the (unknown) kernel used to produce the measure M. Specifically, since M here is constructed as a simple Gaussian waveform it has a single Gaussian spectrum whereas in the previous matched-kernel case we can clearly see two Gaussian shapes in that M-spectrum. Below we will address this with a **filter**.


<a id="org76d514d"></a>

# Shifts and cycles

A (de)convolution smears each sample in the input signal over a region of size Nk as governed by the content of the kernel K. The linear shape padding receives information from samples size Nk from the end of the signal. Without this padding, that information would wrap around and add to the information from the start of the input signal causing **cyclic artifacts**.

When the kernel has a peak that is away from its first sample, the "smearing" is biased and an apparent "shift" is induced. Peaks in the input signal appear **later in the convolution** result and **earlier in the deconvolution** result.

In the case of deconvolution, this K-peak may be further from the zero sample than is the input M-peak. The resulting D-peak will be shifted so far forward that it will **wrap around** and appear at **later in the deconvolution**. Interpreting this as "later" is an error. In fact the last Nk samples in D are **earlier** than the start of M.

The demo can show this by adjusting the location of the kernel to be later:

```sh
uv run decondemo plot --kernel-size=100 --kernel-mean=90 --signal-is-measure --output basic-decon-shift.svg
```

<div class="html" id="orgea5cc8a">

<div id="org438b794" class="figure">
<p><img src="basic-decon-shift.svg" alt="basic-decon-shift.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

One must take care to properly interpret the last Nk samples of D. The "end" of D is really at sample Nm=Nd-Nk-1, where Nm here is the original, pre-padded size of input M. It is possible to **roll** D by Nk to move these early time samples to the front of the array. One must then take care to interpret the rolled-D as starting Nk samples earlier in time than the original input M.


<a id="org58e0ef5"></a>

# Filtered case

In order to combat deconvolution artifacts (and later noise) we may apply an arbitrary filter as part of the deconvolution to form D = F(M/K). See previous discussion of the nature of F w.r.t. padding.

The filter will distort the recovered signal D. We attempt to craft the filter to provide desirable distortion while minimizing unwanted distortion. In practice this needs a careful optimization. Here is one example.

```sh
uv run decondemo plot --signal-is-measure --filter-name=lowpass --filter-scale=0.1  --output basic-filtered-decon.svg
```

<div class="html" id="org5bccc0c">

<div id="org3471d14" class="figure">
<p><img src="basic-filtered-decon.svg" alt="basic-filtered-decon.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

This inserts the filter F waveform and spectrum. The chosen filter is a "low-pass filter" (aka a "high frequency filter") in that it "passes" low frequency energy and attenuates (filters) the rest. In this example, the attenuation reduces the effect of dividing by small values of K and removes the high-frequency wiggles.

Note the filter waveform is cyclically symmetric about the zero interval sample. This is a result of the filter being symmetrically defined in Fourier space as a real valued sampling. This is good for as because it is effectively convolved with the measure M and we do not want it to introduce any artificial shifts.


<a id="org160615b"></a>

# Adding noise

Real signals always come with noise. The demo has a simple white noise model. We go back to the ideal matched case and add the smallest of noise and see that it utterly destroys the ability to recover the signal.

```sh
uv run decondemo plot --noise-rms=0.01 --output basic-convo-decon-noisyq.svg
```

<div class="html" id="org0eb5949">

<div id="orgb45343c" class="figure">
<p><img src="basic-convo-decon-noisyq.svg" alt="basic-convo-decon-noisyq.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

In fact, one may rerun the demo with noise that is too small to be visible in the measured waveform M and the D waveform is still unrecognizable as signal. Matters become even more hopeless when the convolution and deconvolution kernels are not matched.


<a id="org00c742e"></a>

# Noise filters

The effect of adding noise problem is similar to that of the floating point errors but much larger. In both cases, high frequency energy that is amplified by the division of small values of K. As with FP errors, we may apply a low-pass filter to combat the amplified HF noise. However, the filter must be more aggressive as the noise spectrum spans not just a small high-frquency region.

```sh
uv run decondemo plot --noise-rms 0.1 --filter-name=lowpass --filter-scale=0.1 --filter-power=3.0 --output basic-convo-decon-noise-filter.svg
```

<div class="html" id="org9158642">

<div id="orgfec7031" class="figure">
<p><img src="basic-convo-decon-noise-filter.svg" alt="basic-convo-decon-noise-filter.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

Note, the noise has been increased by an order of magnitude to give the filter a greater challenge and yet the signal is recovered reasonably well. The main peak is above the residual (unfiltered) noise and noise appears to have distorted the main peak away from its True Gaussian shape


<a id="orgddb9f7f"></a>

# Spectral leakage

The term **spectral leakage** generally refers to the fact that an operation on a **waveform** causes a change to its **spectrum**. Given that the waveform and spectrum representation of a signal are perfectly symmetric through the DFT, this co-change is not profound. Yet, some changes to a waveform produce surprising changes to the spectrum.

Introduced above is the requirement to pad signal and kernel for convolution or measure and kernel for deconvolution to the mutual **linear size** in order to avoid cyclic artifacts. Typically the padded region values are zero. This adds no energy to the signal. However, it will cause strong spectral leakage distortion when the end of the unpadded arrays have not reached zero value. This can be illustrated by modifying the case of the [unmatched kernel demo](#org3a2ed10) such that the high tail of the measure is truncated.

```sh
uv run decondemo plot  --signal-is-measure --signal-size=50 --output basic-decon-trunc.svg
```

<div class="html" id="orgab5b9fb">

<div id="org0a0a930" class="figure">
<p><img src="basic-decon-trunc.svg" alt="basic-decon-trunc.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

Note, this truncation cuts off the tail which is already very close to zero yet it introduces four orders of magnitude more high-frequency energy to the measure's spectrum. This is the spectral leakage. And, because the DFT conserves energy, this high-frequency energy must be taken from low energy region. That low-frequency loss is not significant. But, what is significant is that the amplification of this new high-frequency energy in the deconvolution completely destroys the recovered signal D.


<a id="org3874aae"></a>

# Mitigation with tapering

Spectral leakage due to the discontinuity introduced by zero padding can be mitigated by applying a **tapering** or **window** function to the input array (measure M in the example above). This **destroys information** at the ends of the waveform by multiplying a function that falls to zero. The demo can apply some well known tapering function forms, for example:

```sh
uv run decondemo plot --show-padded \
   --signal-is-measure --signal-size=50 \
   --taper-signal --taper-name=hann --taper-length=20 \
   --output basic-decon-trunc-taper.svg
```

<div class="html" id="org05d6d4b">

<div id="org748b65e" class="figure">
<p><img src="basic-decon-trunc-taper.svg" alt="basic-decon-trunc-taper.svg" class="org-svg" width="80%" />
</p>
</div>

</div>

As with filters, optimizing the tapering function requires a careful tuning and evaluation. Indeed, tapering functions are essentially equivalent to filters except the former operate in interval-space while the latter operate in Fourier-space. Both destroy information in an attempt to produce a desirable bias.


<a id="orge9a5867"></a>

# Mitigation with extrapolation

An extrapolation function is in a sense the opposite of a tapering function. Instead of destroying information at the ends of a measure, it invents information at the ends of the padded region. Both have the same goal to bring the waveform smoothly to zero to mitigate unwanted spectral leakage. There is support in the demo however I have been unable to hit upon an extrapolation model that will provide any benefit.

## Footnotes

<sup><a id="fn.1" class="footnum" href="#fnr.1">1</a></sup> A better demo would allow for the more realistic case where different kernels are used for convolution and deconvolution.
