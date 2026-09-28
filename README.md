# Juniper — Einstein@Home Binary Radio Pulsar Search (BRP4), Metal port for Apple Silicon

Unofficial Metal build of the [Einstein@Home](https://www.einsteinathome.org)
**Binary Radio Pulsar Search (BRP4)** GPU application, targeting Apple Silicon
(M-series) GPUs.

Port and builds by **Alperen Yavuz**, based on the GPL-licensed upstream BRP4
source and sharing its host-side pipeline (demodulation, harmonic summing,
candidate ranking) with the companion CUDA port,
[brp4-cuda-port](https://github.com/alplix/brp4-cuda-port). Where the CUDA
port targets every NVIDIA GPU since 2010, Juniper targets Apple's own GPU
architecture end to end: resampling, FFT, power spectrum and harmonic summing
all run as Metal compute kernels, with no CPU fallback for the search
pipeline itself.

## Status

Early development. The GPU pipeline (resampling → FFT → power spectrum →
harmonic summing) is being built and validated stage by stage against the
CUDA port's already-verified candidate output, run on identical synthetic
input, before this is trusted on real work units. See
[RELEASES.md](RELEASES.md) (once it exists) for what's actually shipped.

## Why a separate repo from brp4-cuda-port

Same GPL-2+ upstream pipeline, but CUDA and Metal are different enough
platforms (driver API vs. Apple's Metal/MPSGraph, x86/ARM Linux+Windows vs.
Apple Silicon only) that keeping them as separate, focused repositories beats
one repo carrying both toolchains. `src/` here only has what a macOS/Metal
build needs — no CUDA sources, no Windows/Linux cross-build scripts.

## Building from source

Requires full Xcode (not just Command Line Tools) for the `xcrun metal` /
`xcrun metallib` shader compiler, on an Apple Silicon Mac.

```sh
cd src
make -f Makefile.macos.metal
```

## Standalone usage

No BOINC needed for testing:

```sh
./einsteinbinary_BRP4_macos \
    -i input.binary -t bank.txt -l zaplist.txt \
    -o results.dat -c checkpoint.dat -W -D 0 -z
```

## License

Upstream BRP4 source is GPLv2+; this port is distributed under the same
terms (see [LICENSE](LICENSE)).
