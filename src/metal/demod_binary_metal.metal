/***************************************************************************
 *   Copyright (C) 2023 by Oliver Behnke                                   *
 *   oliver.behnke[AT]aei.mpg.de                                           *
 *   Copyright (C) 2026 by Alperen Yavuz (device-chaining rewrite)         *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Resampling + power-spectrum Metal kernels. Adapted from upstream's    *
 *   2023 Metal port (recovered from brp4-cuda-port's git history at       *
 *   b7eb089~1, before that repo's v1.3 cleanup dropped non-CUDA code),    *
 *   with one structural change: the modulated-length and mean-reduction   *
 *   kernels' outputs are consumed by later kernels entirely on-device     *
 *   (through small single-element buffers), matching the CUDA port's      *
 *   demod_binary_cuda.cuh design -- the original 2023 kernels instead     *
 *   read those values back to the host and re-uploaded them via setBytes  *
 *   between dispatches, which is exactly the per-kernel CPU<->GPU sync    *
 *   the CUDA v1.3 rewrite eliminated (see demod_binary_metal.cpp for the  *
 *   host-side half of this fix).                                         *
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

#include <metal_atomic>
#include <metal_compute>
#include <metal_math>
#include <metal_simdgroup>

using namespace metal;

#define ERP_SINCOS_LUT_RES 64
#define ERP_TWO_PI 6.283185f

float deviceSinLUTLookup(const device float *constSinSamples,
                         const device float *constCosSamples,
                         float x)
{
  float xt;
  int i0;
  float d, d2;
  float ts, tc;

  // normalize value
  xt = fract(x / ERP_TWO_PI);  // xt in [0, 1)

  // determine LUT index
  i0 = (int)(xt * ERP_SINCOS_LUT_RES + 0.5f);
  d = d2 = ERP_TWO_PI * (xt - 1.0f / ERP_SINCOS_LUT_RES * i0);
  d2 *= 0.5f * d;

  // fetch sin/cos samples
  ts = constSinSamples[i0];
  tc = constCosSamples[i0];

  // Taylor expansion for sin around the sampled point
  return ts + d * tc - d2 * ts;
}

kernel void kernelTimeSeriesModulation(const device float *constSinSamples [[buffer(0)]],
                                       const device float *constCosSamples [[buffer(1)]],
                                       constant float &tau [[buffer(2)]],
                                       constant float &Omega [[buffer(3)]],
                                       constant float &Psi0 [[buffer(4)]],
                                       constant float &dt [[buffer(5)]],
                                       constant float &step_inv [[buffer(6)]],
                                       constant float &S0 [[buffer(7)]],
                                       device float *del_t [[buffer(8)]],
                                       constant uint &nsamples_unpadded [[buffer(9)]],
                                       uint index [[thread_position_in_grid]])
{
  if (index >= nsamples_unpadded) return;

  float t = index * dt;
  float x = Omega * t + Psi0;
  float sinX = deviceSinLUTLookup(constSinSamples, constCosSamples, x);

  del_t[index] = tau * sinX * step_inv - S0;
}

// Determines the number of resampled time steps (see
// time_series_length_modulated in demod_binary_cuda.cuh for the derivation
// -- same block-parallel backward scan, one threadgroup of
// CUDA_RESAMP_LENGTH_BLOCKDIM_X-equivalent width, ported to simd_max
// instead of atomicMax since the whole scan fits in one threadgroup and
// simdgroup reduction avoids a threadgroup-memory round trip per chunk).
kernel void kernelTimeSeriesLengthModulated(constant uint &nsamples_unpadded [[buffer(0)]],
                                            const device float *del_t [[buffer(1)]],
                                            device uint *timeSeriesLength [[buffer(2)]],
                                            threadgroup atomic_int *highestHit [[threadgroup(0)]],
                                            uint tid [[thread_position_in_threadgroup]],
                                            uint threadsPerTg [[threads_per_threadgroup]])
{
  const float limit = float(nsamples_unpadded - 1);
  int result = 0;

  for (int top = int(nsamples_unpadded) - 1; top >= 0; top -= int(threadsPerTg)) {
    if (tid == 0) {
      atomic_store_explicit(highestHit, -1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int k = top - int(tid);
    // negated comparison keeps the CUDA original's NaN behaviour (a NaN stops the walk)
    if (k >= 0 && !((float(k) - del_t[k]) >= limit)) {
      atomic_fetch_max_explicit(highestHit, k, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int hit = atomic_load_explicit(highestHit, memory_order_relaxed);
    if (hit >= 0) {
      result = hit;
      break;
    }
  }

  if (tid == 0) {
    *timeSeriesLength = uint(result);
  }
}

kernel void kernelTimeSeriesResampling(const device float *input [[buffer(0)]],
                                       const device float *del_t [[buffer(1)]],
                                       const device uint *length [[buffer(2)]],
                                       device float *output [[buffer(3)]],
                                       constant uint &nsamples [[buffer(4)]],
                                       uint index [[thread_position_in_grid]])
{
  if (index >= nsamples) return;

  if (index < *length) {
    int nearest_idx = int(float(index) - del_t[index] + 0.5f);
    output[index] = input[nearest_idx];
  }
  else {
    output[index] = 0.0f;
  }
}

// Sums nsamples_unpadded inputs (out-of-range reads treated as zero by the
// dispatch's bounds check) via simdgroup reduction + one atomic add per
// threadgroup -- matches CUDA's time_series_mean_reduction's total, just
// without CUDA's iterative ping-pong buffer (Metal's simd_sum collapses
// each 32-lane simdgroup in one step, so one pass is enough).
kernel void kernelTimeSeriesMeanReduction(const device float *input [[buffer(0)]],
                                          device atomic_float *output [[buffer(1)]],
                                          constant uint &nsamples_unpadded [[buffer(2)]],
                                          constant uint &simd_groups [[buffer(3)]],
                                          threadgroup float *group_sum [[threadgroup(0)]],
                                          uint tid [[thread_position_in_grid]],
                                          uint lid [[thread_position_in_threadgroup]],
                                          uint simd_lane_id [[thread_index_in_simdgroup]],
                                          uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  float v = (tid < nsamples_unpadded) ? input[tid] : 0.0f;
  group_sum[simd_group_id] = simd_sum(v);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float value = 0.0f;
  if (simd_group_id == 0 && simd_lane_id < simd_groups) {
    value = simd_sum(group_sum[simd_lane_id]);
  }

  if (lid == 0) atomic_fetch_add_explicit(output, value, memory_order_relaxed);
}

// Pads the resampled series (from *length up to nsamples) with its mean
// value, computed in-kernel from the raw sum + *length -- matches CUDA's
// time_series_padding, which also divides on-device rather than on the
// host (see demod_binary_cuda.cuh: `output[i] = sum[0] / (float)offset`).
kernel void kernelTimeSeriesPadding(device float *buffer [[buffer(0)]],
                                    const device float *sum [[buffer(1)]],
                                    const device uint *length [[buffer(2)]],
                                    constant uint &nsamples [[buffer(3)]],
                                    uint index [[thread_position_in_grid]])
{
  if (index >= nsamples) return;

  if (index >= *length) {
    buffer[index] = sum[0] / float(*length);
  }
}

// PENDING Phase 1 (FFT feasibility smoke test, see the project plan):
// this indexing assumes an interleaved-complex FFT output layout
// (input[2*i], input[2*i+1] = real, imag), matching cuFFT's cufftComplex
// layout that the CUDA port's fft_powerspectrum kernel relies on. Whether
// MPSGraph's real-to-complex FFT produces that same layout is unverified
// -- confirm against the Phase 1 smoke test's recorded findings before
// trusting this kernel unmodified.
kernel void kernelPowerspectrum(device const float *input [[buffer(0)]],
                                device float *output [[buffer(1)]],
                                constant float &norm_factor [[buffer(2)]],
                                uint index [[thread_position_in_grid]])
{
  // DC bin zeroed, matching CUDA's fft_powerspectrum
  const float nf = (index == 0) ? 0.0f : norm_factor;
  output[index] = nf * (pow(input[index + index], 2.0f) + pow(input[index + index + 1], 2.0f));
}
