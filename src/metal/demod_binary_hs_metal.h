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
 *   Metal backend for harmonic summing (2nd ... 16th harmonic) of the    *
 *   power spectrum. Interface mirrors demod_binary_hs_cuda.cu in the     *
 *   companion brp4-cuda-port repo exactly, so demod_binary.c can dispatch *
 *   to either backend without caring which one it's calling.              *
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

#ifndef DEMOD_BINARY_HS_METAL_H
#define DEMOD_BINARY_HS_METAL_H

#include <stdint.h>

#include "../diptr.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int set_up_harmonic_summing(float **sumspec,
                                   int32_t **dirty,
                                   unsigned int *nr_pages_ptr,
                                   unsigned int fundamental_idx_hi,
                                   unsigned int harmonic_idx_hi);

extern int run_harmonic_summing(float **sumspec,
                                int32_t **dirty,
                                unsigned int nr_pages,
                                DIfloatPtr powerspectrum_dip,
                                unsigned int window_2,
                                unsigned int fundamental_idx_hi,
                                unsigned int harmonic_idx_hi,
                                float *thresholds);

extern int tear_down_harmonic_summing(float **sumspec, int32_t **dirty);

#ifdef __cplusplus
}
#endif

#endif
