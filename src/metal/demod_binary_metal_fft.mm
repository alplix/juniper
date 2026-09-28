// Objective-C++ bridge: MPSGraph real-to-Hermitean FFT + power-spectrum
// dispatch for the Juniper Metal backend. See demod_binary_metal_fft.h for
// the plain-C++ interface this exposes to demod_binary_metal.cpp, and why
// FFT and power-spectrum are encoded together in one bridge call.
//
// Verified empirically (project plan Phase 1, fft_smoketest.mm, run on a
// real M1) before this file was written:
//   - realToHermiteanFFTWithTensor:axes:descriptor:name: exists on this
//     Xcode/macOS (API available since macOS 14.0) and matches cuFFT's
//     CUFFT_R2C semantics bin-for-bin (delta and cosine test signals
//     verified against a naive O(N^2) reference DFT).
//   - Output layout is interleaved complex float32 pairs (re, im), i.e.
//     identical to cuFFT's cufftComplex -- confirmed by reading the raw
//     output buffer bytes directly, not just trusting the header docs.
//   - Arbitrary (non-power-of-two) lengths work (tested N=4097, N=6000).
//   - MPSGraph can be encoded into an application-owned MTLCommandBuffer
//     (via MPSCommandBuffer commandBufferWithCommandBuffer:) that also
//     carries an ordinary compute-kernel dispatch encoded afterwards on
//     the same underlying command buffer -- confirmed the second kernel
//     correctly saw the FFT's output with no host round trip in between.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "demod_binary_metal_fft.h"

namespace {

struct FFTPlan {
  MPSGraph *graph;
  MPSGraphTensor *inputTensor;
  MPSGraphTensor *outputTensor;
  uint32_t nsamples;
  uint32_t halfBins;
};

// metal-cpp objects are ABI-identical to the underlying Objective-C object
// pointer (NS::Object wraps an objc_object*, see Foundation/NSObject.hpp),
// so a plain bridge cast is the correct, standard bridge -- this is the
// same pattern Apple's own metal-cpp samples use.
template <typename T>
T bridge(void *p) {
  return (__bridge T)p;
}

}  // namespace

extern "C" MPSFFTHandle mps_fft_create(void *mtlDevice, uint32_t nsamples) {
  @autoreleasepool {
    (void)mtlDevice;  // MPSGraph doesn't need the device at graph-build time

    FFTPlan *plan = new FFTPlan();
    plan->nsamples = nsamples;
    plan->halfBins = nsamples / 2 + 1;

    plan->graph = [[MPSGraph alloc] init];
    plan->inputTensor = [plan->graph placeholderWithShape:@[ @(nsamples) ]
                                                  dataType:MPSDataTypeFloat32
                                                      name:@"resampledTimeSeries"];

    MPSGraphFFTDescriptor *desc = [MPSGraphFFTDescriptor descriptor];
    desc.scalingMode = MPSGraphFFTScalingModeNone;  // matches cuFFT's unscaled CUFFT_R2C
    plan->outputTensor = [plan->graph realToHermiteanFFTWithTensor:plan->inputTensor
                                                              axes:@[ @0 ]
                                                        descriptor:desc
                                                              name:@"powerSpectrumFFT"];

    // retain the graph/tensors for the plan's lifetime (ARC would otherwise
    // release them once this autoreleasepool drains)
    CFBridgingRetain(plan->graph);
    CFBridgingRetain(plan->inputTensor);
    CFBridgingRetain(plan->outputTensor);

    return (MPSFFTHandle)plan;
  }
}

extern "C" int mps_fft_and_powerspectrum_encode(MPSFFTHandle handle, void *mtlCommandBuffer,
                                                void *inputBuffer, void *fftScratchBuffer,
                                                void *powerspectrumPipeline,
                                                void *powerspectrumOutputBuffer, float normFactor,
                                                uint32_t fftSizePadded) {
  @autoreleasepool {
    FFTPlan *plan = (FFTPlan *)handle;
    id<MTLCommandBuffer> rawCmd = bridge<id<MTLCommandBuffer>>(mtlCommandBuffer);
    id<MTLBuffer> inBuf = bridge<id<MTLBuffer>>(inputBuffer);
    id<MTLBuffer> fftBuf = bridge<id<MTLBuffer>>(fftScratchBuffer);
    id<MTLComputePipelineState> psPipeline = bridge<id<MTLComputePipelineState>>(powerspectrumPipeline);
    id<MTLBuffer> psOutBuf = bridge<id<MTLBuffer>>(powerspectrumOutputBuffer);

    MPSGraphTensorData *inData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:@[ @(plan->nsamples) ]
                                             dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *outData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:fftBuf
                                                shape:@[ @(plan->halfBins) ]
                                             dataType:MPSDataTypeComplexFloat32];

    NSDictionary *feeds = @{plan->inputTensor : inData};
    NSDictionary *results = @{plan->outputTensor : outData};

    // rawCmd already has the resampling kernels encoded on it (see
    // demod_binary_metal.cpp's run_resampling) and was deliberately left
    // uncommitted for exactly this -- wrapping and continuing here keeps
    // resampling + FFT + power-spectrum as one command buffer / one sync
    // point for the whole per-template pipeline, matching the CUDA port's
    // fully-chained default-stream design.
    MPSCommandBuffer *mpsCmd = [MPSCommandBuffer commandBufferWithCommandBuffer:rawCmd];

    [plan->graph encodeToCommandBuffer:mpsCmd
                                  feeds:feeds
                       targetOperations:nil
                      resultsDictionary:results
                    executionDescriptor:nil];

    // power-spectrum kernel, chained on the SAME (possibly commitAndContinue'd)
    // root command buffer -- see the header comment for why this stays inside
    // one Objective-C++ scope instead of crossing back into C++.
    id<MTLComputeCommandEncoder> enc = [mpsCmd.rootCommandBuffer computeCommandEncoder];
    [enc setComputePipelineState:psPipeline];
    [enc setBuffer:fftBuf offset:0 atIndex:0];
    [enc setBuffer:psOutBuf offset:0 atIndex:1];
    [enc setBytes:&normFactor length:sizeof(normFactor) atIndex:2];
    NSUInteger tg = psPipeline.maxTotalThreadsPerThreadgroup;
    if (tg > fftSizePadded) tg = fftSizePadded;
    [enc dispatchThreads:MTLSizeMake(fftSizePadded, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];

    [mpsCmd.rootCommandBuffer commit];
    [mpsCmd.rootCommandBuffer waitUntilCompleted];

    if (mpsCmd.rootCommandBuffer.status == MTLCommandBufferStatusError) {
      NSLog(@"Juniper: FFT+power-spectrum command buffer failed: %@",
           mpsCmd.rootCommandBuffer.error);
      return 1;
    }
    return 0;
  }
}

extern "C" void mps_fft_destroy(MPSFFTHandle handle) {
  @autoreleasepool {
    FFTPlan *plan = (FFTPlan *)handle;
    if (!plan) return;
    CFRelease((__bridge CFTypeRef)plan->graph);
    CFRelease((__bridge CFTypeRef)plan->inputTensor);
    CFRelease((__bridge CFTypeRef)plan->outputTensor);
    delete plan;
  }
}
