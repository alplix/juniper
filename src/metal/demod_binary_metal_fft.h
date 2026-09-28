#ifndef DEMOD_BINARY_METAL_FFT_H
#define DEMOD_BINARY_METAL_FFT_H

#include <stdint.h>

// Objective-C++ bridge (demod_binary_metal_fft.mm) exposing MPSGraph's
// real-to-Hermitean FFT to the plain-C++ rest of the Metal backend, which
// uses metal-cpp and can't call Objective-C APIs directly. Every pointer
// here is a metal-cpp object pointer (MTL::Device*, MTL::CommandQueue*,
// MTL::Buffer*, MTL::ComputePipelineState*) passed as void* -- the .mm file
// bridges them to their Objective-C id<...> counterparts (metal-cpp objects
// are ABI-identical to the underlying Objective-C object pointer, see
// NS::Object's definition in Foundation/NSObject.hpp).
//
// FFT + power-spectrum are encoded and committed together, in one function,
// entirely inside this bridge -- deliberately NOT split across the C-ABI
// boundary as "encode FFT" then "encode power-spectrum on the same still-
// open command buffer", because MPSGraph's encodeToCommandBuffer: may
// internally call commitAndContinue (per Apple's own docs), which can swap
// out the underlying MTLCommandBuffer from under the caller; handing a
// possibly-stale command buffer pointer back across the ABI boundary would
// be fragile to get right. Keeping the whole FFT+power-spectrum stage
// inside one Objective-C++ scope sidesteps that risk entirely -- it's
// exactly the structure verified working in the project's Phase 1 smoke
// test (fft_smoketest.mm: FFT via encodeToCommandBuffer, then a plain
// compute kernel via the resulting mpsCmd.rootCommandBuffer, then commit
// +wait, all in one function).
//
// Net effect versus the CUDA port's zero-round-trip design: resampling
// commits+waits on its own command buffer (see demod_binary_metal.cpp's
// run_resampling), then this FFT+power-spectrum stage is a second, separate
// command buffer with its own commit+wait -- one extra sync point per
// template versus CUDA's fully chained default-stream design, in exchange
// for a bridge implementation whose command-buffer lifetime is easy to
// reason about. Harmonic summing (demod_binary_hs_metal.cpp) already needs
// its own sync point regardless, so this adds one sync point total to the
// per-template budget, not a whole new class of overhead.

#ifdef __cplusplus
extern "C" {
#endif

typedef void *MPSFFTHandle;

// Builds an MPSGraph performing a real-to-Hermitean FFT of length nsamples
// (unscaled, matching cuFFT's CUFFT_R2C default). Output is nsamples/2+1
// interleaved-complex-float32 bins, confirmed against cuFFT's cufftComplex
// layout by the project's Phase 1 smoke test -- kernelPowerspectrum's
// existing input[2*i]/input[2*i+1] indexing needs no changes.
MPSFFTHandle mps_fft_create(void *mtlDevice, uint32_t nsamples);

// Encodes the FFT (inputBuffer, real float32, nsamples elements ->
// fftScratchBuffer, interleaved-complex float32, nsamples/2+1 elements) and
// the power-spectrum kernel (fftScratchBuffer -> powerspectrumOutputBuffer,
// float32, fftSizePadded elements, using the already-created
// powerspectrumPipeline and normFactor -- same kernel, same arguments as
// demod_binary_metal.metal's kernelPowerspectrum) onto mtlCommandBuffer --
// an MTL::CommandBuffer* the caller already obtained from its command queue
// and already used to encode the resampling kernels (see
// demod_binary_metal.cpp's run_resampling/run_fft), left deliberately
// uncommitted so this call can continue encoding onto it rather than
// starting a second command buffer / second sync point. Commits it and
// waits for completion. Returns 0 on success.
int mps_fft_and_powerspectrum_encode(MPSFFTHandle handle, void *mtlCommandBuffer,
                                     void *inputBuffer, void *fftScratchBuffer,
                                     void *powerspectrumPipeline, void *powerspectrumOutputBuffer,
                                     float normFactor, uint32_t fftSizePadded);

void mps_fft_destroy(MPSFFTHandle handle);

#ifdef __cplusplus
}
#endif

#endif
