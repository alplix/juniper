# Juniper — Einstein@Home Binary Radio Pulsar Search (BRP4), rebuilt for Apple Silicon GPUs

Unofficial Metal build of the [Einstein@Home](https://www.einsteinathome.org)
**Binary Radio Pulsar Search (BRP4)** GPU application, targeting Apple
Silicon (M-series) GPUs end to end — not a CPU fallback with a GPU-accelerated
step bolted on, and not the OpenCL path Apple/Einstein@Home officially ship.

Port and builds by **Alperen Yavuz**, based on the GPL-licensed upstream BRP4
source and sharing its host-side pipeline (demodulation, harmonic summing,
candidate ranking, BOINC integration) with the companion CUDA port,
[brp4-cuda-port](https://github.com/alplix/brp4-cuda-port). Where the CUDA
port targets every NVIDIA GPU since 2010 via the CUDA driver API, Juniper
targets Apple's GPU stack directly: resampling, FFT, power spectrum and
harmonic summing all run as native Metal compute kernels and an MPSGraph
FFT, with no CPU fallback anywhere in the search pipeline itself.

## Why this exists

The BRP4 GPU application Einstein@Home distributes for Mac users runs on
OpenCL. Apple has deprecated OpenCL in favor of Metal for years, and a
hand-written Metal implementation — including a Metal-native FFT via
`MPSGraph` instead of routing through a compatibility layer — has real room
to outperform an OpenCL path Apple no longer optimizes for. See
[Performance](#performance) below for where that stands today.

## Status

**Working and validated against the CUDA port on synthetic data; not yet
run against a real BOINC work unit or packaged for BOINC deployment.**

- The full GPU pipeline (resampling → FFT → power spectrum → harmonic
  summing) builds, links, and runs end to end on a real Apple Silicon Mac
  (M1) against real Metal/MPSGraph frameworks, real GSL/FFTW/libxml2, and a
  from-source BOINC client library build.
- Candidate output has been diffed against the CUDA port (running the
  identical synthetic input on an NVIDIA GPU) for both a strong-signal and a
  weak-signal synthetic work unit: **zero** frequency/orbital-template/
  harmonic-count mismatches in either case, and power values agreeing to
  within the floating-point rounding level expected between two independent
  FFT implementations (cuFFT vs. Apple's `MPSGraph` FFT) — see
  [Validation](#validation).
- Not yet done: a real Einstein@Home work unit (classic BRP4 `-t bank`
  format — current live Einstein@Home tasks use the newer BRP7/Ter5 format,
  which neither this port nor the CUDA port supports), BOINC anonymous-
  platform packaging (`app_info.xml`), code signing/notarization.

## Architecture

Unlike upstream's own (now-removed) 2023 Metal prototype — which only
accelerated resampling and the FFT/power-spectrum stage, leaving harmonic
summing on the CPU — Juniper runs the **entire** per-template search stage
on the GPU, matching the CUDA port's architecture:

| Stage | Implementation |
|---|---|
| Resampling (demodulation into pulsar time) | 5 chained Metal compute kernels — modulation, block-parallel modulated-length scan, resampling, `simd_sum`-based mean reduction, mean padding |
| FFT | `MPSGraph`'s real-to-Hermitean FFT (`realToHermiteanFFTWithTensor:axes:descriptor:name:`), unscaled, matching cuFFT's `CUFFT_R2C` bin-for-bin, including its interleaved-complex `float32` output layout |
| Power spectrum | 1 Metal compute kernel, DC bin zeroed |
| Harmonic summing (2nd–16th harmonic) | 2 Metal compute kernels (`harmonic_summing_kernel` + `harmonic_summing_kernel_gaps`), a line-for-line port of the CUDA port's kernels — no equivalent existed in any prior Metal build of this application |

**Per-template GPU submission budget**: one command buffer covering
resampling + FFT + power spectrum (committed once, no intermediate host
round trip — the data-dependent modulated length and the resampling mean
both stay device-side between kernels), then one more for harmonic summing
— **2 sync points per template**, matching the CUDA port's own fully-chained
default-stream design exactly. Harmonic summing needs only one sync where
the CUDA port needs two, because Apple Silicon's unified memory
(`MTLResourceStorageModeShared`) makes the GPU-written candidate buffers
directly host-readable after one `waitUntilCompleted` — no separate
device→host copy command the way CUDA's discrete host/device memory
requires.

MPSGraph has no metal-cpp (Apple's official C++ Metal bindings, vendored
here under `third_party/metal-cpp`) bindings, so the FFT step is bridged
through a small Objective-C++ file (`src/metal/demod_binary_metal_fft.mm`)
that the rest of the backend — written in plain C++ against metal-cpp —
calls into via a handful of `extern "C"` functions.

## Performance

Measured on the user's M1 MacBook Air (8-core GPU) against a companion
NVIDIA RTX 5070 Ti running the CUDA port, both processing the *same*
synthetic work unit (`test/make_synthetic.c`, 4,194,304 samples):

| Templates | Juniper (M1) | brp4-cuda-port (RTX 5070 Ti) |
|---|---|---|
| 512 | 6 s | 2 s |
| 20,000 | 159 s | 15 s |

The gap widening with template count (3× → ~10.6×, not staying constant)
reflects the RTX 5070 Ti's ~2 s at 512 templates being dominated by fixed
CUDA context/module-load overhead rather than per-template cost — its real
per-template throughput is roughly 0.67 ms, against Juniper's ~7.9 ms.
Comparing a high-end discrete desktop GPU against an 8-core laptop
integrated GPU is not apples-to-apples on raw compute, so this gap is
expected; it isn't a measure of how Juniper compares to the *stock OpenCL*
build Einstein@Home ships for Mac.

Linearly projecting the 20,000-template measurement to a realistic
~50,000-template work unit gives roughly **400 seconds**, against a
**700–890 second** range recalled for the stock OpenCL Apple client on
comparable hardware — but this is an extrapolation from a synthetic
benchmark, not a real work unit measurement, and no stock build was
available to benchmark side by side on the same machine. Treat it as a
promising early signal, not a settled result.

**What's already been done**: the per-template pipeline was originally 3
separate command-buffer submissions (resampling, FFT+power-spectrum,
harmonic summing); merging resampling into the same command buffer as
FFT+power-spectrum (removing one sync point) took the 20,000-template case
from 191 s to 159 s (~17% faster), with candidate output re-verified
identical against the CUDA build afterward.

**What's not done yet**: real GPU-side profiling (Xcode's Metal System
Trace / GPU counters) to find which kernel actually dominates per-template
time — resampling, FFT, power-spectrum, or harmonic summing — rather than
further blind command-buffer restructuring.

## Validation

`test/make_synthetic.c` generates a synthetic dedispersed time series with
an injected periodic signal, a template bank, and an empty zap list — the
same harness the CUDA port's own v1.3 validation used. Both the strong-
signal (default) and weak-signal (`--amp 0.02`) cases were run through
Juniper (M1) and through `brp4-cuda-port`'s CUDA build (RTX 5070 Ti) on
identical input, and the candidate lists (`results.dat`) diffed:

| Case | Candidates (both sides) | Frequency/template/harmonic mismatches | Max relative power difference |
|---|---|---|---|
| Strong signal | 101 / 101 | 0 | 2.2 × 10⁻⁴ |
| Weak signal | 6 / 6 | 0 | ~1 × 10⁻² (see note) |

The weak-signal case's looser power tolerance isn't evidence of an
algorithmic difference: its absolute power values are ~70–100 versus ~10⁶
in the strong-signal case, so the same absolute floating-point rounding
noise (cuFFT vs. `MPSGraph`'s FFT, two independent implementations) becomes
a larger *relative* fraction of a smaller number. Every candidate's
identity — frequency bin, orbital template parameters, harmonic count —
matched exactly in both cases; only the power-derived diagnostic columns
carry expected floating-point noise.

## Why a separate repo from brp4-cuda-port

Same GPL-2+ upstream pipeline, but CUDA and Metal are different enough
platforms (driver API vs. Apple's Metal/MPSGraph, x86/ARM Linux+Windows vs.
Apple Silicon only) that keeping them as separate, focused repositories
beats one repo carrying both toolchains. `src/` here only has what a
macOS/Metal build needs — no CUDA sources, no Windows/Linux cross-build
scripts.

## Building from source

Apple Silicon Mac only.

**Prerequisites:**

- **Full Xcode**, not just Command Line Tools — the `xcrun metal` /
  `xcrun metallib` shader compiler ships only with the full IDE. After
  installing, also run `xcodebuild -downloadComponent MetalToolchain`
  (newer Xcode versions split the Metal compiler out as a separate
  ~800 MB download) and accept the license (`sudo xcodebuild -license
  accept`).
- **[Homebrew](https://brew.sh)**, then `brew install gsl fftw libxml2
  libtool`. GSL and FFTW are needed by the shared host code's threshold
  statistics regardless of GPU backend, not by the Metal kernels
  themselves; libxml2 is for the BOINC screensaver IPC channel.
- **BOINC's client libraries**, built from source — no Homebrew formula
  ships them:
  ```sh
  git clone --depth 1 https://github.com/BOINC/boinc.git
  cd boinc
  export LIBTOOLIZE=/opt/homebrew/bin/glibtoolize   # macOS's own libtool isn't GNU's
  ./_autosetup
  ./configure --prefix="$HOME/boinc-install" \
      --disable-server --disable-client --disable-manager \
      --disable-shared --enable-static
  (cd lib && make install)
  (cd api && make install)
  ```

**Build:**

```sh
cd src
make -f Makefile.macos.metal \
    METAL_CPP_DIR=../third_party/metal-cpp \
    EINSTEIN_RADIO_INSTALL="$HOME/boinc-install"
```

`METAL_CPP_DIR` and `EINSTEIN_RADIO_INSTALL` default to sensible values if
omitted (see `src/Makefile.macos.metal`); `HOMEBREW_PREFIX` and
`LIBXML2_PREFIX` can be overridden the same way if Homebrew isn't at
`/opt/homebrew`.

## Standalone usage

No BOINC needed for testing:

```sh
./einsteinbinary_BRP4_macos \
    -i input.binary -t bank.txt -l zaplist.txt \
    -o results.dat -c checkpoint.dat -W -D 0 -z
```

Generate synthetic test input with `test/make_synthetic.c`:

```sh
cc -O2 -o make_synthetic test/make_synthetic.c -lm
./make_synthetic --outdir . --samples 4194304 --templates 512
```

Note: `logMessage()` output (the interesting run-time log — device
selection, per-stage progress, candidate statistics) goes to `stderr.txt`
in the working directory once BOINC's diagnostics initialize, not to the
process's own stdout/stderr, which only shows the startup banner.

## Known limitation (same as the CUDA port)

Einstein@Home's current Ter5 `sband_dns` tasks belong to the new-generation
official application (**BRP7**, MeerKAT data), whose command-line options
(`--pb_min`, `--start_template_id`, …) and source code aren't published.
This build, like brp4-cuda-port, targets classic `-t bank`-style work and
standalone runs, and will fail on current Ter5 work if advertised via
`app_info.xml`.

## License

Upstream BRP4 source is GPLv2+; this port is distributed under the same
terms (see [LICENSE](LICENSE)). Vendored `third_party/metal-cpp` is
Apache-2.0 (Apple).
