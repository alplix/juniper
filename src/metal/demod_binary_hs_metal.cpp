/***************************************************************************
 *   Copyright (C) 2008 by Benjamin Knispel, Holger Pletsch                *
 *   benjamin.knispel[AT]aei.mpg.de                                        *
 *   Copyright (C) 2009,2010 by Oliver Bock                                *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *   Copyright (C) 2009,2010 by Heinz-Bernd Eggenstein                     *
 *   Copyright (C) 2026 by Alperen Yavuz (Metal port)                      *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Host glue for the Metal harmonic-summing kernels                     *
 *   (harmonic_summing_kernel.metal). Mirrors demod_binary_hs_cuda.cu's    *
 *   allocate-once-per-work-unit / reuse-per-template structure and its    *
 *   sync-point budget, but takes advantage of Apple Silicon's unified     *
 *   memory: buffers are allocated MTL::ResourceStorageModeShared, so the  *
 *   GPU-written sumspec/dirty data is directly host-visible after one     *
 *   command-buffer wait -- there's no separate device->host copy command  *
 *   the way CUDA needs (discrete host/device memory), so this needs only  *
 *   ONE sync point per template where the CUDA version needs two.         *
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

#include "demod_binary_hs_metal.h"

#include <stdlib.h>
#include <string.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "../demod_binary.h"
#include "../erp_utilities.h"
#include "../hs_common.h"
#include "demod_binary_metal_shared.h"

// module-scope state, set up once per work unit, reused for every template
// (mirrors demod_binary_hs_cuda.cu's module globals)

static MTL::ComputePipelineState *pipelineHarmonicSumming = NULL;
static MTL::ComputePipelineState *pipelineHarmonicSummingGaps = NULL;

static MTL::Buffer *sumspecDevBuf[5] = {NULL, NULL, NULL, NULL, NULL};  // [0] is aliased per template
static MTL::Buffer *dirtyDevBuf = NULL;  // flat 5 * nr_pages int32 buffer
static MTL::Buffer *hLutDevBuf = NULL;
static MTL::Buffer *kLutDevBuf = NULL;

static unsigned int gNrPages = 0;
static unsigned int gFundamentalIdxHi = 0;

int set_up_harmonic_summing(float **sumspec,
                            int32_t **dirty,
                            unsigned int *nr_pages_ptr,
                            unsigned int fundamental_idx_hi,
                            unsigned int harmonic_idx_hi) {
  (void)harmonic_idx_hi;

  NS::Error *nsError = NULL;

  MTL::Function *fnMain =
      g_metalLibrary->newFunction(NS::String::string("harmonic_summing_kernel", NS::UTF8StringEncoding));
  if (!fnMain) {
    logMessage(error, true, "Couldn't find harmonic_summing_kernel Metal function!\n");
    return (RADPUL_METAL_KERNEL_CREATE);
  }
  MTL::Function *fnGaps = g_metalLibrary->newFunction(
      NS::String::string("harmonic_summing_kernel_gaps", NS::UTF8StringEncoding));
  if (!fnGaps) {
    logMessage(error, true, "Couldn't find harmonic_summing_kernel_gaps Metal function!\n");
    return (RADPUL_METAL_KERNEL_CREATE);
  }

  pipelineHarmonicSumming = g_metalDevice->newComputePipelineState(fnMain, &nsError);
  if (!pipelineHarmonicSumming) {
    logMessage(error, true, "Couldn't create harmonic-summing compute pipeline!\n");
    return (RADPUL_METAL_PIPELINE_CREATE);
  }
  pipelineHarmonicSummingGaps = g_metalDevice->newComputePipelineState(fnGaps, &nsError);
  if (!pipelineHarmonicSummingGaps) {
    logMessage(error, true, "Couldn't create harmonic-summing-gaps compute pipeline!\n");
    return (RADPUL_METAL_PIPELINE_CREATE);
  }
  fnMain->release();
  fnGaps->release();

  // sumspecDevBuf[0] stays NULL here; it's aliased per template to the
  // power-spectrum buffer handed in by run_harmonic_summing, exactly like
  // sumspecDev[0] in the CUDA backend.
  for (int i = 1; i < 5; i++) {
    sumspecDevBuf[i] =
        g_metalDevice->newBuffer(sizeof(float) * fundamental_idx_hi, MTL::ResourceStorageModeShared);
    if (!sumspecDevBuf[i]) {
      logMessage(error, true, "Couldn't allocate %lu bytes of HS summing device memory!\n",
                 (unsigned long)(sizeof(float) * fundamental_idx_hi));
      return (RADPUL_METAL_MEM_ALLOC_DEVICE);
    }
    // this buffer's contents() IS the host-visible array the caller reads
    // from (unified memory) -- no separate host allocation/copy needed.
    sumspec[i] = (float *)sumspecDevBuf[i]->contents();
    memset(sumspec[i], 0, sizeof(float) * fundamental_idx_hi);
  }
  sumspec[0] = NULL;  // aliased per template in run_harmonic_summing, like CUDA's powerspectrumHost

  unsigned int nr_pages = (fundamental_idx_hi >> LOG_PS_PAGE_SIZE) + 1;
  *nr_pages_ptr = nr_pages;
  gNrPages = nr_pages;
  gFundamentalIdxHi = fundamental_idx_hi;

  dirtyDevBuf = g_metalDevice->newBuffer(sizeof(int32_t) * nr_pages * 5, MTL::ResourceStorageModeShared);
  if (!dirtyDevBuf) {
    logMessage(error, true, "Couldn't allocate %lu bytes of HS page-flag device memory!\n",
               (unsigned long)(sizeof(int32_t) * nr_pages * 5));
    return (RADPUL_METAL_MEM_ALLOC_DEVICE);
  }
  memset(dirtyDevBuf->contents(), 0, sizeof(int32_t) * nr_pages * 5);

  // dirty[h] is a pointer-arithmetic slice into the single flat dirtyDevBuf
  // (not a separate allocation -- tear_down_harmonic_summing must release
  // dirtyDevBuf as a whole, never dirty[h] individually), matching the flat
  // layout the kernels themselves index into
  // (((fundamental_idx_hi >> LOG_PS_PAGE_SIZE) + 1) * h + page).
  int32_t *dirtyBase = (int32_t *)dirtyDevBuf->contents();
  for (int i = 0; i < 5; i++) {
    dirty[i] = dirtyBase + (size_t)i * nr_pages;
  }

  hLutDevBuf = g_metalDevice->newBuffer(sizeof(int32_t) * 16, MTL::ResourceStorageModeShared);
  kLutDevBuf = g_metalDevice->newBuffer(sizeof(int32_t) * 16, MTL::ResourceStorageModeShared);
  if (!hLutDevBuf || !kLutDevBuf) {
    logMessage(error, true, "Couldn't allocate HS lookup-table device memory!\n");
    return (RADPUL_METAL_MEM_ALLOC_DEVICE);
  }
  memcpy(hLutDevBuf->contents(), h_lut, sizeof(int32_t) * 16);
  memcpy(kLutDevBuf->contents(), k_lut, sizeof(int32_t) * 16);

  logMessage(debug, true, "Metal harmonic summing set up (%u pages per harmonic)\n", nr_pages);

  return 0;
}

int tear_down_harmonic_summing(float **sumspec, int32_t **dirty) {
  (void)sumspec;
  (void)dirty;

  for (int i = 1; i < 5; i++) {
    if (sumspecDevBuf[i]) sumspecDevBuf[i]->release();
    sumspecDevBuf[i] = NULL;
  }
  if (dirtyDevBuf) dirtyDevBuf->release();
  dirtyDevBuf = NULL;
  if (hLutDevBuf) hLutDevBuf->release();
  if (kLutDevBuf) kLutDevBuf->release();
  hLutDevBuf = kLutDevBuf = NULL;

  if (pipelineHarmonicSumming) pipelineHarmonicSumming->release();
  if (pipelineHarmonicSummingGaps) pipelineHarmonicSummingGaps->release();
  pipelineHarmonicSumming = pipelineHarmonicSummingGaps = NULL;

  return 0;
}

int run_harmonic_summing(float **sumspec,
                         int32_t **dirty,
                         unsigned int nr_pages,
                         DIfloatPtr powerspectrum_dip,
                         unsigned int window_2,
                         unsigned int fundamental_idx_hi,
                         unsigned int harmonic_idx_hi,
                         float *thresholds) {
  MTL::Buffer *powerspectrumBuf = (MTL::Buffer *)powerspectrum_dip.device_ptr;

  // power spectrum acts as the first (1st harmonic) spectrum, like CUDA's
  // sumspecDev[0] = powerspectrumDev aliasing
  sumspecDevBuf[0] = powerspectrumBuf;
  sumspec[0] = (float *)powerspectrumBuf->contents();

  MTL::CommandBuffer *cmd = g_metalQueue->commandBuffer();

  // zero sumspec[1..4] and the dirty flags on the GPU (blit fill), so a
  // stale value from an earlier template can never leak through a page the
  // kernels don't touch this time -- same reason CUDA re-memsets every
  // template instead of reusing prior contents.
  MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
  for (int i = 1; i < 5; i++) {
    blit->fillBuffer(sumspecDevBuf[i], NS::Range(0, sizeof(float) * fundamental_idx_hi), 0);
  }
  blit->fillBuffer(dirtyDevBuf, NS::Range(0, sizeof(int32_t) * nr_pages * 5), 0);
  blit->endEncoding();

  MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();

  // main kernel: 16 x gridY1 threadgroups of 256 threads (exact grid math
  // from demod_binary_hs_cuda.cu's run_harmonic_summing)
  unsigned int l2Main = ((harmonic_idx_hi - 1 + 8) >> 4) + 1;
  unsigned int gridY1 = (l2Main + 255) / 256;

  enc->setComputePipelineState(pipelineHarmonicSumming);
  enc->setBuffer(sumspecDevBuf[1], 0, 0);
  enc->setBuffer(sumspecDevBuf[2], 0, 1);
  enc->setBuffer(sumspecDevBuf[3], 0, 2);
  enc->setBuffer(sumspecDevBuf[4], 0, 3);
  enc->setBuffer(dirtyDevBuf, 0, 4);
  enc->setBuffer(powerspectrumBuf, 0, 5);
  enc->setBytes(&window_2, sizeof(window_2), 6);
  enc->setBytes(&fundamental_idx_hi, sizeof(fundamental_idx_hi), 7);
  enc->setBytes(&harmonic_idx_hi, sizeof(harmonic_idx_hi), 8);
  enc->setBytes(thresholds, sizeof(float) * 5, 9);
  enc->setBuffer(hLutDevBuf, 0, 10);
  enc->setBuffer(kLutDevBuf, 0, 11);
  enc->dispatchThreadgroups(MTL::Size(16, gridY1, 1), MTL::Size(256, 1, 1));

  // gaps kernel: same buffer bindings, half the threadgroup size
  unsigned int l2Gaps = ((harmonic_idx_hi - 1 + 12) >> 4) + 1;
  unsigned int gridY2 = (l2Gaps + 255) / 256;

  enc->setComputePipelineState(pipelineHarmonicSummingGaps);
  enc->dispatchThreadgroups(MTL::Size(16, gridY2, 1), MTL::Size(128, 1, 1));

  enc->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();
  // single sync point: on return, sumspec[0..4] and dirty[0..4] (all
  // ResourceStorageModeShared, i.e. unified memory) are safe for the host
  // to read directly -- no device->host copy command needed the way CUDA's
  // discrete-memory design requires, so this backend needs one sync point
  // per template here where the CUDA one needs two.

  return 0;
}
