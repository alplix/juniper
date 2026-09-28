/***************************************************************************
 *   Copyright (C) 2023 by Oliver Behnke                                   *
 *   oliver.behnke[AT]aei.mpg.de                                           *
 *   Copyright (C) 2026 by Alperen Yavuz (rewrite: device-chained          *
 *   resampling, dropped VkFFT/GSL, shared device/queue/library globals)   *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Demodulates dedispersed time series using a bank of orbital           *
 *   parameters. After this step, an FFT of the resampled time series is   *
 *   searched for pulsed, periodic signals by harmonic summing (see        *
 *   demod_binary_hs_metal.cpp).                                           *
 *                                                                         *
 *   FFT: real-to-Hermitean via MPSGraph, bridged through                  *
 *   demod_binary_metal_fft.mm (plain C++ here can't call the Objective-C  *
 *   MPSGraph API directly). Verified against the project plan's Phase 1   *
 *   feasibility gate on a real M1 before being wired in here -- see that  *
 *   file's header comment for what was confirmed and how.                 *
 *                                                                         *
 *   Einstein@Home is free software: you can redistribute it and/or modify *
 *   it under the terms of the GNU General Public License as published     *
 *   by the Free Software Foundation, version 2 of the License.            *
 *                                                                         *
 *   Einstein@Home is distributed in the hope that it will be useful,      *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with Einstein@Home. If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 ***************************************************************************/

#include "demod_binary_metal.h"

#include <stdlib.h>
#include <string.h>

#ifndef NS_PRIVATE_IMPLEMENTATION
#define NS_PRIVATE_IMPLEMENTATION
#endif
#ifndef CA_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#endif
#ifndef MTL_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#endif
// combined embedded .metallib (resampling/FFT/power-spectrum +
// harmonic-summing kernels, one library -- see Makefile.macos.metal)
#include <default.metallib.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "../demod_binary.h"
#include "../erp_utilities.h"
#include "demod_binary_metal_fft.h"
#include "demod_binary_metal_shared.h"

// matches the CUDA port's CUDA_FFT_BLOCKDIM_X (demod_binary_cuda.cuh) --
// kept the same value so the power-spectrum kernel's grid granularity (and
// therefore fft_size_padded, and the CPU-side toplist indexing that assumes
// it) stays identical between backends.
#define METAL_FFT_BLOCKDIM_X 256
#define PADDED_FFT_SIZE(fftsize) \
  (METAL_FFT_BLOCKDIM_X * ((unsigned int)(fftsize) + METAL_FFT_BLOCKDIM_X - 1) / METAL_FFT_BLOCKDIM_X)

// definitions for the shared globals declared in demod_binary_metal_shared.h
// -- this is the ONE translation unit in the program that defines the
// metal-cpp *_PRIVATE_IMPLEMENTATION macros above, so it's also the natural
// home for the one set of shared handles.
MTL::Device *g_metalDevice = NULL;
MTL::CommandQueue *g_metalQueue = NULL;
MTL::Library *g_metalLibrary = NULL;

namespace {

MTL::Function *fnModulation = NULL;
MTL::Function *fnLengthModulated = NULL;
MTL::Function *fnResampling = NULL;
MTL::Function *fnMeanReduction = NULL;
MTL::Function *fnPadding = NULL;
MTL::Function *fnPowerspectrum = NULL;

MTL::ComputePipelineState *pipeModulation = NULL;
MTL::ComputePipelineState *pipeLengthModulated = NULL;
MTL::ComputePipelineState *pipeResampling = NULL;
MTL::ComputePipelineState *pipeMeanReduction = NULL;
MTL::ComputePipelineState *pipePadding = NULL;
MTL::ComputePipelineState *pipePowerspectrum = NULL;

MTL::Buffer *originalTimeSeriesBuf = NULL;
MTL::Buffer *sinLUTBuf = NULL;
MTL::Buffer *cosLUTBuf = NULL;
MTL::Buffer *modTimeOffsetsBuf = NULL;
MTL::Buffer *timeSeriesLengthBuf = NULL;  // single uint, device-side only (no host round trip)
MTL::Buffer *timeSeriesMeanBuf = NULL;    // single atomic_float, device-side only
MTL::Buffer *resampledTimeSeriesBuf = NULL;

// resampling's command buffer, encoded but deliberately left uncommitted --
// run_fft continues encoding FFT + power-spectrum onto this SAME buffer and
// commits it, so the whole per-template GPU pipeline (resampling through
// power-spectrum) is one command buffer / one sync point, matching the CUDA
// port's fully-chained default-stream design instead of the two separate
// submissions an earlier version of this file used (see the project memory
// for the perf numbers that motivated this -- ~9.5ms/template before this
// change, empirically dominated by the extra sync between the two stages).
MTL::CommandBuffer *pendingCommandBuffer = NULL;

MPSFFTHandle fftHandle = NULL;
MTL::Buffer *fftScratchBuf = NULL;         // interleaved-complex float32, nsamples/2+1 elements
MTL::Buffer *powerspectrumOutputBuf = NULL;  // float32, fft_size_padded elements

unsigned int gNsamplesUnpadded = 0;
unsigned int gNsamples = 0;

}  // namespace

int initialize_metal(int metalDeviceIdGiven, int *metalDeviceIdPtr) {
  if (!metalDeviceIdGiven || *metalDeviceIdPtr < 0) {
    logMessage(debug, true, "No (valid) Metal device ID given. Using default device...\n");
    g_metalDevice = MTL::CreateSystemDefaultDevice();
  }
  else {
    NS::Array *devices = MTL::CopyAllDevices();
    if (*metalDeviceIdPtr < (int)devices->count()) {
      g_metalDevice = (MTL::Device *)devices->object((NS::UInteger)*metalDeviceIdPtr);
      logMessage(debug, true, "Selected Metal device #%i as requested...\n", *metalDeviceIdPtr);
    }
  }
  if (!g_metalDevice) {
    logMessage(error, true, "No suitable Metal device available!\n");
    return (RADPUL_METAL_DEVICE_FIND);
  }

  logMessage(info, true, "Using Metal device \"%s\"\n",
             g_metalDevice->name()->cString(NS::UTF8StringEncoding));

  g_metalQueue = g_metalDevice->newCommandQueue();
  if (!g_metalQueue) {
    logMessage(error, true, "Couldn't create Metal command queue!\n");
    return (RADPUL_METAL_CMDQUEUE_CREATE);
  }

  NS::Error *nsError = NULL;
  dispatch_data_t libraryData = dispatch_data_create(&default_metallib[0], default_metallib_len, NULL, NULL);
  g_metalLibrary = g_metalDevice->newLibrary(libraryData, &nsError);
  if (!g_metalLibrary) {
    logMessage(error, true, "Couldn't load embedded Metal kernel library!\n");
    return (RADPUL_METAL_LIBRARY_CREATE);
  }

  struct {
    MTL::Function **fn;
    const char *name;
  } fns[] = {
      {&fnModulation, "kernelTimeSeriesModulation"},
      {&fnLengthModulated, "kernelTimeSeriesLengthModulated"},
      {&fnResampling, "kernelTimeSeriesResampling"},
      {&fnMeanReduction, "kernelTimeSeriesMeanReduction"},
      {&fnPadding, "kernelTimeSeriesPadding"},
      {&fnPowerspectrum, "kernelPowerspectrum"},
  };
  for (size_t k = 0; k < sizeof(fns) / sizeof(fns[0]); k++) {
    *fns[k].fn = g_metalLibrary->newFunction(NS::String::string(fns[k].name, NS::UTF8StringEncoding));
    if (!*fns[k].fn) {
      logMessage(error, true, "Couldn't find Metal function \"%s\"!\n", fns[k].name);
      return (RADPUL_METAL_KERNEL_CREATE);
    }
  }

  return 0;
}

int set_up_resampling(DIfloatPtr input_dip,
                      DIfloatPtr *output_dip,
                      const RESAMP_PARAMS *const params,
                      float *sinLUTsamples,
                      float *cosLUTsamples) {
  NS::Error *nsError = NULL;

  struct {
    MTL::ComputePipelineState **pipe;
    MTL::Function *fn;
    const char *what;
  } pipes[] = {
      {&pipeModulation, fnModulation, "TSM"},
      {&pipeLengthModulated, fnLengthModulated, "TSLM"},
      {&pipeResampling, fnResampling, "TSR"},
      {&pipeMeanReduction, fnMeanReduction, "TSMR"},
      {&pipePadding, fnPadding, "TSP"},
  };
  for (size_t k = 0; k < sizeof(pipes) / sizeof(pipes[0]); k++) {
    *pipes[k].pipe = g_metalDevice->newComputePipelineState(pipes[k].fn, &nsError);
    if (!*pipes[k].pipe) {
      logMessage(error, true, "Couldn't create %s compute pipeline!\n", pipes[k].what);
      return (RADPUL_METAL_PIPELINE_CREATE);
    }
  }

  gNsamplesUnpadded = params->nsamples_unpadded;
  gNsamples = params->nsamples;

  // Original time series and the two LUTs are host data we need on the GPU;
  // use the COPYING newBuffer overload (newBufferWithBytes:length:options:)
  // rather than the no-copy variant the 2023 skeleton used -- the no-copy
  // path requires page-aligned host memory, which a plain malloc'd float*
  // isn't guaranteed to be, so copying is the safe choice here.
  originalTimeSeriesBuf = g_metalDevice->newBuffer(
      input_dip.host_ptr, sizeof(float) * params->nsamples_unpadded, MTL::ResourceStorageModeShared);
  sinLUTBuf = g_metalDevice->newBuffer(sinLUTsamples, ERP_SINCOS_LUT_SIZE * sizeof(float),
                                       MTL::ResourceStorageModeShared);
  cosLUTBuf = g_metalDevice->newBuffer(cosLUTsamples, ERP_SINCOS_LUT_SIZE * sizeof(float),
                                       MTL::ResourceStorageModeShared);
  if (!originalTimeSeriesBuf || !sinLUTBuf || !cosLUTBuf) {
    logMessage(error, true, "Error allocating resampling input device memory!\n");
    return (RADPUL_METAL_MEM_ALLOC_DEVICE);
  }

  modTimeOffsetsBuf =
      g_metalDevice->newBuffer(sizeof(float) * params->nsamples_unpadded, MTL::ResourceStorageModeShared);
  timeSeriesLengthBuf = g_metalDevice->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
  timeSeriesMeanBuf = g_metalDevice->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
  // Sized to comfortably hold the resampled real time series; FFT buffer
  // sizing/layout is revisited once Phase 1 (MPSGraph feasibility) lands --
  // see the PENDING note on set_up_fft below.
  resampledTimeSeriesBuf =
      g_metalDevice->newBuffer(sizeof(float) * params->nsamples, MTL::ResourceStorageModeShared);

  if (!modTimeOffsetsBuf || !timeSeriesLengthBuf || !timeSeriesMeanBuf || !resampledTimeSeriesBuf) {
    logMessage(error, true, "Error allocating resampling working device memory!\n");
    return (RADPUL_METAL_MEM_ALLOC_DEVICE);
  }

  output_dip->device_ptr = (void *)resampledTimeSeriesBuf;

  return 0;
}

int run_resampling(DIfloatPtr input_dip, DIfloatPtr output_dip, const RESAMP_PARAMS *const params) {
  (void)input_dip;
  (void)output_dip;  // == resampledTimeSeriesBuf, already known to this TU

  MTL::CommandBuffer *cmd = g_metalQueue->commandBuffer();

  // zero the per-template scratch (mean accumulator) before this template's
  // dispatches -- everything else below is written unconditionally by the
  // kernels that follow, so it needs no reset.
  MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
  blit->fillBuffer(timeSeriesMeanBuf, NS::Range(0, sizeof(float)), 0);
  blit->endEncoding();

  MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();

  float tau = params->tau, Omega = params->Omega, Psi0 = params->Psi0;
  float dt = params->dt, step_inv = params->step_inv, S0 = params->S0;
  uint32_t nsamplesUnpadded = params->nsamples_unpadded;
  uint32_t nsamples = params->nsamples;

  // --- time series modulation ---
  enc->setComputePipelineState(pipeModulation);
  enc->setBuffer(sinLUTBuf, 0, 0);
  enc->setBuffer(cosLUTBuf, 0, 1);
  enc->setBytes(&tau, sizeof(tau), 2);
  enc->setBytes(&Omega, sizeof(Omega), 3);
  enc->setBytes(&Psi0, sizeof(Psi0), 4);
  enc->setBytes(&dt, sizeof(dt), 5);
  enc->setBytes(&step_inv, sizeof(step_inv), 6);
  enc->setBytes(&S0, sizeof(S0), 7);
  enc->setBuffer(modTimeOffsetsBuf, 0, 8);
  enc->setBytes(&nsamplesUnpadded, sizeof(nsamplesUnpadded), 9);
  {
    NS::UInteger tg = pipeModulation->maxTotalThreadsPerThreadgroup();
    if (tg > nsamplesUnpadded) tg = nsamplesUnpadded;
    enc->dispatchThreads(MTL::Size(nsamplesUnpadded, 1, 1), MTL::Size(tg, 1, 1));
  }

  // --- modulated length (single threadgroup, block-parallel backward scan) ---
  enc->setComputePipelineState(pipeLengthModulated);
  enc->setBytes(&nsamplesUnpadded, sizeof(nsamplesUnpadded), 0);
  enc->setBuffer(modTimeOffsetsBuf, 0, 1);
  enc->setBuffer(timeSeriesLengthBuf, 0, 2);
  {
    NS::UInteger tg = pipeLengthModulated->maxTotalThreadsPerThreadgroup();
    if (tg > 1024) tg = 1024;
    if (tg > nsamplesUnpadded) tg = nsamplesUnpadded;
    enc->setThreadgroupMemoryLength(sizeof(int), 0);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(tg, 1, 1));
  }

  // --- resampling (reads *timeSeriesLengthBuf on-device, no host round trip) ---
  enc->setComputePipelineState(pipeResampling);
  enc->setBuffer(originalTimeSeriesBuf, 0, 0);
  enc->setBuffer(modTimeOffsetsBuf, 0, 1);
  enc->setBuffer(timeSeriesLengthBuf, 0, 2);
  enc->setBuffer(resampledTimeSeriesBuf, 0, 3);
  enc->setBytes(&nsamples, sizeof(nsamples), 4);
  {
    NS::UInteger tg = pipeResampling->maxTotalThreadsPerThreadgroup();
    if (tg > nsamples) tg = nsamples;
    enc->dispatchThreads(MTL::Size(nsamples, 1, 1), MTL::Size(tg, 1, 1));
  }

  // --- mean reduction ---
  NS::UInteger tgMR = pipeMeanReduction->maxTotalThreadsPerThreadgroup();
  if (tgMR > nsamplesUnpadded) tgMR = nsamplesUnpadded;
  NS::UInteger simdSize = pipeMeanReduction->threadExecutionWidth();
  NS::UInteger simdGroups = (tgMR + simdSize - 1) / simdSize;

  enc->setComputePipelineState(pipeMeanReduction);
  enc->setBuffer(resampledTimeSeriesBuf, 0, 0);
  enc->setBuffer(timeSeriesMeanBuf, 0, 1);
  enc->setBytes(&nsamplesUnpadded, sizeof(nsamplesUnpadded), 2);
  {
    uint32_t simdGroups32 = (uint32_t)simdGroups;
    enc->setBytes(&simdGroups32, sizeof(simdGroups32), 3);
  }
  enc->setThreadgroupMemoryLength(sizeof(float) * simdGroups, 0);
  enc->dispatchThreads(MTL::Size(nsamplesUnpadded, 1, 1), MTL::Size(tgMR, 1, 1));

  // --- padding (mean computed in-kernel from the raw sum + *length) ---
  enc->setComputePipelineState(pipePadding);
  enc->setBuffer(resampledTimeSeriesBuf, 0, 0);
  enc->setBuffer(timeSeriesMeanBuf, 0, 1);
  enc->setBuffer(timeSeriesLengthBuf, 0, 2);
  enc->setBytes(&nsamples, sizeof(nsamples), 3);
  {
    NS::UInteger tg = pipePadding->maxTotalThreadsPerThreadgroup();
    if (tg > nsamples) tg = nsamples;
    enc->dispatchThreads(MTL::Size(nsamples, 1, 1), MTL::Size(tg, 1, 1));
  }

  enc->endEncoding();
  // deliberately NOT committed here -- run_fft continues encoding onto this
  // same command buffer (FFT + power-spectrum) and commits/waits once for
  // the combined pipeline. Every intermediate value above (modulated
  // length, mean) already stayed device-side, so there was never a reason
  // for a host round trip in between; the only reason to had split this
  // into two submissions before was MPSGraph's command-buffer lifetime,
  // which the FFT bridge now handles by taking this buffer directly
  // instead of creating its own from the queue.
  pendingCommandBuffer = cmd;

  return 0;
}

int tear_down_resampling(DIfloatPtr output_dip) {
  (void)output_dip;

  if (originalTimeSeriesBuf) originalTimeSeriesBuf->release();
  if (sinLUTBuf) sinLUTBuf->release();
  if (cosLUTBuf) cosLUTBuf->release();
  if (modTimeOffsetsBuf) modTimeOffsetsBuf->release();
  if (timeSeriesLengthBuf) timeSeriesLengthBuf->release();
  if (timeSeriesMeanBuf) timeSeriesMeanBuf->release();
  if (resampledTimeSeriesBuf) resampledTimeSeriesBuf->release();
  originalTimeSeriesBuf = sinLUTBuf = cosLUTBuf = modTimeOffsetsBuf = NULL;
  timeSeriesLengthBuf = timeSeriesMeanBuf = resampledTimeSeriesBuf = NULL;

  for (MTL::ComputePipelineState **p :
       {&pipeModulation, &pipeLengthModulated, &pipeResampling, &pipeMeanReduction, &pipePadding}) {
    if (*p) (*p)->release();
    *p = NULL;
  }

  return 0;
}

int set_up_fft(DIfloatPtr input_dip, DIfloatPtr *output_dip, uint32_t nsamples, unsigned int fft_size) {
  (void)input_dip;  // FFT input is resampledTimeSeriesBuf, already known to this TU

  NS::Error *nsError = NULL;
  pipePowerspectrum = g_metalDevice->newComputePipelineState(fnPowerspectrum, &nsError);
  if (!pipePowerspectrum) {
    logMessage(error, true, "Couldn't create power-spectrum compute pipeline!\n");
    return (RADPUL_METAL_PIPELINE_CREATE);
  }

  fftHandle = mps_fft_create((void *)g_metalDevice, nsamples);
  if (!fftHandle) {
    logMessage(error, true, "Couldn't build MPSGraph FFT plan!\n");
    return (RADPUL_METAL_FFT_PLAN);
  }

  const unsigned int halfBins = nsamples / 2 + 1;
  fftScratchBuf =
      g_metalDevice->newBuffer(sizeof(float) * 2 * halfBins, MTL::ResourceStorageModePrivate);

  const unsigned int fftSizePadded = PADDED_FFT_SIZE(fft_size);
  powerspectrumOutputBuf =
      g_metalDevice->newBuffer(sizeof(float) * fftSizePadded, MTL::ResourceStorageModeShared);

  if (!fftScratchBuf || !powerspectrumOutputBuf) {
    logMessage(error, true, "Error allocating FFT/power-spectrum device memory!\n");
    return (RADPUL_METAL_MEM_ALLOC_DEVICE);
  }

  output_dip->device_ptr = (void *)powerspectrumOutputBuf;

  return 0;
}

int run_fft(DIfloatPtr input, DIfloatPtr output, uint32_t nsamples, unsigned int fft_size,
           float norm_factor) {
  (void)nsamples;  // fixed by the plan built in set_up_fft
  (void)output;    // == powerspectrumOutputBuf, already known to this TU

  const unsigned int fftSizePadded = PADDED_FFT_SIZE(fft_size);

  int result = mps_fft_and_powerspectrum_encode(
      fftHandle, (void *)pendingCommandBuffer, (void *)resampledTimeSeriesBuf, (void *)fftScratchBuf,
      (void *)pipePowerspectrum, (void *)powerspectrumOutputBuf, norm_factor, fftSizePadded);
  pendingCommandBuffer = NULL;  // committed by the bridge call above regardless of its result
  if (result != 0) {
    logMessage(error, true, "Metal FFT/power-spectrum command buffer failed (error: %d)\n", result);
    return (RADPUL_METAL_FFT_EXEC);
  }

  // DC bin zeroed, matching kernelPowerspectrum's own (index==0) special case being redundant
  // here is fine -- the kernel already does it; nothing further needed on the host side.

  return 0;
}

int tear_down_fft(DIfloatPtr output_dip) {
  (void)output_dip;

  if (fftHandle) mps_fft_destroy(fftHandle);
  fftHandle = NULL;
  if (fftScratchBuf) fftScratchBuf->release();
  if (powerspectrumOutputBuf) powerspectrumOutputBuf->release();
  fftScratchBuf = powerspectrumOutputBuf = NULL;
  if (pipePowerspectrum) pipePowerspectrum->release();
  pipePowerspectrum = NULL;

  return 0;
}

void printDeviceGlobalMemStatus(const ERP_LOGLEVEL logLevel, const bool followUp) {
  if (!g_metalDevice) return;

  double mib = 1024.0 * 1024.0;
  double allocated = (double)g_metalDevice->currentAllocatedSize() / mib;
  double recommendedMax = (double)g_metalDevice->recommendedMaxWorkingSetSize() / mib;

  logMessage(logLevel, !followUp,
             "Allocated %.1f MiB of GPU memory (recommended maximum: %.1f MiB)\n", allocated,
             recommendedMax);
}

int shutdown_metal() {
  for (MTL::Function **f :
       {&fnModulation, &fnLengthModulated, &fnResampling, &fnMeanReduction, &fnPadding, &fnPowerspectrum}) {
    if (*f) (*f)->release();
    *f = NULL;
  }
  if (pipePowerspectrum) pipePowerspectrum->release();
  pipePowerspectrum = NULL;

  if (g_metalLibrary) g_metalLibrary->release();
  if (g_metalQueue) g_metalQueue->release();
  if (g_metalDevice) g_metalDevice->release();
  g_metalLibrary = NULL;
  g_metalQueue = NULL;
  g_metalDevice = NULL;

  logMessage(info, true, "Metal shutdown complete!\n");
  return 0;
}
