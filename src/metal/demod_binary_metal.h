/***************************************************************************
 *   Copyright (C) 2023 by Oliver Behnke                                   *
 *   oliver.behnke[AT]aei.mpg.de                                           *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
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

#ifndef DEMOD_BINARY_METAL_H
#define DEMOD_BINARY_METAL_H

#include <stdbool.h>
#include <stdint.h>

#include "../diptr.h"
#include "../erp_utilities.h"
#include "../structs.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int initialize_metal(int metalDeviceIdGiven, int *metalDeviceId);

// Metal equivalent of the CUDA backend's printDeviceGlobalMemStatus (see
// cuda_utilities.h/.c in the companion brp4-cuda-port repo) -- logs current
// GPU memory usage via MTL::Device's currentAllocatedSize()/
// recommendedMaxWorkingSetSize(), used by demod_binary.c's debug logging.
extern void printDeviceGlobalMemStatus(const ERP_LOGLEVEL logLevel, const bool followUp);

extern int set_up_resampling(DIfloatPtr input,
                             DIfloatPtr *output,
                             const RESAMP_PARAMS *const params,
                             float *sinLUTsamples,
                             float *cosLUTsamples);
extern int run_resampling(DIfloatPtr input, DIfloatPtr output, const RESAMP_PARAMS *const params);
extern int tear_down_resampling(DIfloatPtr output);

extern int set_up_fft(DIfloatPtr input,
                      DIfloatPtr *output,
                      uint32_t nsamples,
                      unsigned int fft_size);
extern int run_fft(DIfloatPtr input,
                   DIfloatPtr output,
                   uint32_t nsamples,
                   unsigned int fft_size,
                   float norm_factor);
extern int tear_down_fft(DIfloatPtr output);

extern int shutdown_metal();

#ifdef __cplusplus
}
#endif

#endif
