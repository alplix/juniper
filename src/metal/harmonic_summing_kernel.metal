/***************************************************************************
 *   Copyright (C) 2008 by Benjamin Knispel, Holger Pletsch                *
 *   benjamin.knispel[AT]aei.mpg.de                                        *
 *   Copyright (C) 2009, 2010 by Oliver Bock                               *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *   Copyright (C) 2009, 2010 by Heinz-Bernd Eggenstein                    *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Metal translation of the CUDA harmonic-summing kernels               *
 *   (src/cuda/app/harmonic_summing_kernel.cuh in the companion            *
 *   brp4-cuda-port repo). Line-for-line port; see the notes below for     *
 *   the handful of places where CUDA and Metal idioms genuinely differ.   *
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

#include <metal_stdlib>
using namespace metal;

// must match hs_common.h's LOG_PS_PAGE_SIZE (10) in the companion CUDA port
#define LOG_PS_PAGE_SIZE 10

#define HS_BLOCKSIZE 256
#define HS_LOG_BLOCKSIZE 8

// CUDA's __ldg (read-only data cache hint) has no Metal equivalent because a
// `const device` pointer is already eligible for the read-only cache path;
// this is a plain read, kept as a macro only so the two kernels below stay
// visually diffable against harmonic_summing_kernel.cuh.
#define FETCH(t, i) (t[i])

// IMPORTANT: `i`/`j`/`jj` below are signed (they go negative at the left
// spectrum border, see idx_j_offset), while window_2/fundamental_idx_hi/
// harmonic_idx_hi are unsigned. Comparisons like `i < window_2` therefore
// promote `i` to unsigned first (standard C/Metal usual-arithmetic-
// conversion rules) -- a negative `i` becomes a huge unsigned value, which
// makes that comparison false but the paired `i >= harmonic_idx_hi` true,
// so the `||` still correctly excludes it. This is exactly how the CUDA
// source relies on the same conversion; keep the signed/unsigned types as
// they are (int vs. uint), do not "clean up" with casts, or left-border
// handling silently breaks.

// main kernel for harmonic summing.
// Constraint: threadgroup size must be a multiple of 16 (each sub-block of
// 16 consecutive threads is independent).
//
// Step 1: each thread looks at powerspectrum[i*k/16+0.5] for a thread
// specific i and k=1..15, computing harmonic-sum candidates into
// threadgroup memory. Step 2: 12 of every 16 threads reduce those
// candidates to a max and (if above threshold) write one of the four
// sumspec[] arrays. See harmonic_summing_kernel.cuh for the full derivation
// -- this is a faithful line-for-line port of that kernel's math.
kernel void harmonic_summing_kernel(device float *sumspec1 [[buffer(0)]],
                                    device float *sumspec2 [[buffer(1)]],
                                    device float *sumspec3 [[buffer(2)]],
                                    device float *sumspec4 [[buffer(3)]],
                                    device int *dirty [[buffer(4)]],
                                    const device float *powerspectrum [[buffer(5)]],
                                    constant uint &window_2 [[buffer(6)]],
                                    constant uint &fundamental_idx_hi [[buffer(7)]],
                                    constant uint &harmonic_idx_hi [[buffer(8)]],
                                    const device float *thrA [[buffer(9)]],
                                    const device int *d_h_lut [[buffer(10)]],
                                    const device int *d_k_lut [[buffer(11)]],
                                    uint3 tgid [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]])
{
  threadgroup float sspec_cand[4 * HS_BLOCKSIZE];

  int idx_j = (int(tgid.y) << 4) + int(tgid.x);
  int idx_j_offset = (idx_j << HS_LOG_BLOCKSIZE) + -16;  // negative index to handle left border

  int i = idx_j_offset + int(tid) + 8;
  int k;
  int h = i;
  int j, jj, len, offset, lend2, lenM1;
  float sum;
  float p;
  int i2, i4, i8;
  int iN;

  if (i < window_2 || i >= harmonic_idx_hi) {
    // no candidate contribution from this index
    sspec_cand[tid] = 0.0f;
    sspec_cand[HS_BLOCKSIZE + tid] = 0.0f;
    sspec_cand[2 * HS_BLOCKSIZE + tid] = 0.0f;
    sspec_cand[3 * HS_BLOCKSIZE + tid] = 0.0f;
  }
  else {
    p = FETCH(powerspectrum, i);
    i2 = i + i;
    i4 = i << 2;
    i8 = i4 + i4;
    iN = i8 + 8;
    if ((p > FETCH(thrA, 0)) && (i < fundamental_idx_hi)) {
      dirty[(i >> LOG_PS_PAGE_SIZE)] = 1;
    }

    p += FETCH(powerspectrum, iN >> 4);
    sspec_cand[tid] = p;

    iN = i4 + 8;
    sum = FETCH(powerspectrum, iN >> 4);
    iN += i8;
    sum += FETCH(powerspectrum, iN >> 4);
    p += sum;

    sspec_cand[HS_BLOCKSIZE + tid] = p;

    iN = i2 + 8;
    sum = FETCH(powerspectrum, iN >> 4);
    iN += i4;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i4;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i4;
    sum += FETCH(powerspectrum, iN >> 4);
    p += sum;

    sspec_cand[2 * HS_BLOCKSIZE + tid] = p;

    iN = i + 8;
    sum = FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);

    p += sum;
    sspec_cand[3 * HS_BLOCKSIZE + tid] = p;
  }

  // finished Step 1
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // selecting 12 threads out of every 16-thread sub block for step 2, one
  // for every sumspec[][] slot to fill with max candidate for that slot.
  if ((tid & 15) < 12) {
    h = FETCH(d_h_lut, tid & 15);
    k = FETCH(d_k_lut, tid & 15) + ((int(tid) >> 4) << 4);

    len = 1 << h;
    lend2 = len >> 1;
    offset = ((h - 1) << HS_LOG_BLOCKSIZE) + k;
    lenM1 = len - 1;

    sum = sspec_cand[offset];
#pragma unroll(15)
    for (j = 1; j < 16; j++) {
      sum = fmax(sum, sspec_cand[offset + (j & lenM1)]);
    }

    jj = (idx_j_offset + k + 8 + lend2);
    j = (jj >= 0) ? (jj >> h) : -1;

    if ((sum > FETCH(thrA, h)) && j >= 0 && (j < fundamental_idx_hi)) {
      // CUDA indexes sumspec[h-1][j] through a pointer array built in shared
      // memory; MSL has no clean device-address-space pointer array here,
      // so this is the one deliberate structural difference from the CUDA
      // source -- same four buffers, selected by a switch instead of an
      // array index. h is always in [1,4], so exactly one case ever runs.
      switch (h - 1) {
        case 0: sumspec1[j] = sum; break;
        case 1: sumspec2[j] = sum; break;
        case 2: sumspec3[j] = sum; break;
        case 3: sumspec4[j] = sum; break;
      }

      // mark this page of the sumspec array as dirty (plain write: CUDA's
      // original is also a plain non-atomic write here -- every thread that
      // could race on this address writes the same value 1, so there's no
      // read-modify-write and no torn-write risk for an aligned int32)
      dirty[((fundamental_idx_hi >> LOG_PS_PAGE_SIZE) + 1) * h + (j >> LOG_PS_PAGE_SIZE)] = 1;
    }
  }
}

// secondary kernel: fills the sumspec gaps the main kernel leaves at
// sub-block borders. Same shape as the main kernel, half its threadgroup
// size, different index math -- see harmonic_summing_kernel.cuh.
kernel void harmonic_summing_kernel_gaps(device float *sumspec1 [[buffer(0)]],
                                         device float *sumspec2 [[buffer(1)]],
                                         device float *sumspec3 [[buffer(2)]],
                                         device float *sumspec4 [[buffer(3)]],
                                         device int *dirty [[buffer(4)]],
                                         const device float *powerspectrum [[buffer(5)]],
                                         constant uint &window_2 [[buffer(6)]],
                                         constant uint &fundamental_idx_hi [[buffer(7)]],
                                         constant uint &harmonic_idx_hi [[buffer(8)]],
                                         const device float *thrA [[buffer(9)]],
                                         const device int *d_h_lut [[buffer(10)]],
                                         const device int *d_k_lut [[buffer(11)]],
                                         uint3 tgid [[threadgroup_position_in_grid]],
                                         uint tid [[thread_position_in_threadgroup]])
{
  threadgroup float sspec_cand[2 * HS_BLOCKSIZE];

  int idx_j = (int(tgid.y) << 4) + int(tgid.x);
  int idx_j_offset = (idx_j << HS_LOG_BLOCKSIZE);

  int idx_i_offset = int(tid);
  int i = idx_j_offset + 4 + (int(tid) & 7) + ((int(tid) >> 3) << 4);

  int k;
  int h = i;
  int j, len, offset, lend2, lenM1;
  float sum;
  float p;

  int i2, i4, i8;
  int iN;

  // for this kernel there can be overlap with the left spectrum border
  // (index 0), but i can still be lower than window_2 or higher than
  // harmonic_idx_hi -- same signed/unsigned reliance noted above.
  if (i < window_2 || i >= harmonic_idx_hi) {
    sspec_cand[idx_i_offset] = 0.0f;
    sspec_cand[HS_BLOCKSIZE / 2 + idx_i_offset] = 0.0f;
    sspec_cand[2 * (HS_BLOCKSIZE / 2) + idx_i_offset] = 0.0f;
    sspec_cand[3 * (HS_BLOCKSIZE / 2) + idx_i_offset] = 0.0f;
  }
  else {
    p = FETCH(powerspectrum, i);
    i2 = i + i;
    i4 = i << 2;
    i8 = i4 + i4;
    iN = i8 + 8;
    if ((p > FETCH(thrA, 0)) && (i < fundamental_idx_hi)) {
      dirty[(i >> LOG_PS_PAGE_SIZE)] = 1;
    }
    p += FETCH(powerspectrum, iN >> 4);
    sspec_cand[idx_i_offset] = p;

    iN = i4 + 8;
    sum = FETCH(powerspectrum, iN >> 4);
    iN += i8;
    sum += FETCH(powerspectrum, iN >> 4);
    p += sum;

    sspec_cand[(HS_BLOCKSIZE / 2) + idx_i_offset] = p;

    iN = i2 + 8;
    sum = FETCH(powerspectrum, iN >> 4);
    iN += i4;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i4;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i4;
    sum += FETCH(powerspectrum, iN >> 4);
    p += sum;

    sspec_cand[2 * (HS_BLOCKSIZE / 2) + idx_i_offset] = p;

    iN = i + 8;
    sum = FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);
    iN += i2;
    sum += FETCH(powerspectrum, iN >> 4);

    p += sum;
    sspec_cand[3 * (HS_BLOCKSIZE / 2) + idx_i_offset] = p;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2 sub-blocks of 3 active threads for each 16 (2*8) sub-block in step 1
  if ((tid < 4 * (HS_BLOCKSIZE / 16)) && ((tid & 3) != 3)) {
    // for this kernel the h_lut lookup value is directly computable from
    // the thread index, so (unlike the main kernel) there's no d_h_lut read
    h = 3 - (int(tid) & 3);
    k = FETCH(d_k_lut, (int(tid) & 3) + 13) + ((int(tid) >> 2) << 3);
    len = 1 << h;
    lend2 = len >> 1;
    offset = ((h - 1) << (HS_LOG_BLOCKSIZE - 1)) + k;
    lenM1 = len - 1;

    sum = sspec_cand[offset];

    // for this kernel the maximum number of candidates to max is 8
#pragma unroll(7)
    for (j = 1; j < 8; j++) {
      sum = fmax(sum, sspec_cand[offset + (j & lenM1)]);
    }

    j = ((idx_j_offset + k + ((int(tid) >> 2) << 3) + 4 + lend2) >> h);

    // j is provably >= 0 here (idx_j_offset, k, the shift term, 4 and
    // lend2 are all >= 0 for this kernel's index ranges), unlike the main
    // kernel there's no explicit j >= 0 guard in the CUDA source either.
    if ((sum > FETCH(thrA, h)) && (j < fundamental_idx_hi)) {
      switch (h - 1) {
        case 0: sumspec1[j] = sum; break;
        case 1: sumspec2[j] = sum; break;
        case 2: sumspec3[j] = sum; break;
        case 3: sumspec4[j] = sum; break;
      }

      dirty[((fundamental_idx_hi >> LOG_PS_PAGE_SIZE) + 1) * h + (j >> LOG_PS_PAGE_SIZE)] = 1;
    }
  }
}
