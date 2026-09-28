###########################################################################
#   Copyright (C) 2008-2023 by Oliver Behnke                              #
#   Copyright (C) 2026 by Alperen Yavuz (Metal-only rework for Juniper)   #
#                                                                         #
#   This file is part of Einstein@Home (Radio Pulsar Edition).            #
#                                                                         #
#   Einstein@Home is free software: you can redistribute it and/or modify #
#   it under the terms of the GNU General Public License as published     #
#   by the Free Software Foundation, version 2 of the License.            #
#                                                                         #
#   Einstein@Home is distributed in the hope that it will be useful,      #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of        #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
#   GNU General Public License for more details.                          #
#                                                                         #
#   You should have received a copy of the GNU General Public License     #
#   along with Einstein@Home. If not, see <http://www.gnu.org/licenses/>. #
#                                                                         #
###########################################################################
#
# Prerequisites on the build Mac (Apple Silicon only):
#   - full Xcode (not just Command Line Tools) for `xcrun metal`/`metallib`
#   - GSL, FFTW (single precision), libxml2 dev headers+libs -- these are
#     needed by the shared demod_binary.c/erp_boinc_ipc.cpp regardless of
#     GPU backend (statistics + BOINC screensaver IPC), not by the Metal
#     kernels themselves. Point EINSTEIN_RADIO_INSTALL at wherever these
#     (and BOINC's own dev libraries, which still need to be built from
#     https://github.com/BOINC/boinc source -- no Homebrew formula ships
#     them) are installed, e.g. a Homebrew prefix plus a manual BOINC build.
#
# path settings
EINSTEIN_RADIO_SRC?=$(PWD)
EINSTEIN_RADIO_INSTALL?=$(PWD)
METAL_CPP_DIR?=$(EINSTEIN_RADIO_SRC)/../third_party/metal-cpp

# config values
CXX ?= clang++
ERP_VERSION ?= v0.1-dev

# variables
LIBS += -L$(EINSTEIN_RADIO_INSTALL)/lib
LIBS += -lgsl -lgslcblas -lfftw3f -lxml2
LIBS += -lboinc_api -lboinc
LIBS += -lpthread -lz

LDFLAGS += -framework Foundation -framework QuartzCore -framework Metal
LDFLAGS += -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

CXXFLAGS += -I$(METAL_CPP_DIR)
CXXFLAGS += -I$(EINSTEIN_RADIO_INSTALL)/include
CXXFLAGS += -I$(EINSTEIN_RADIO_INSTALL)/include/boinc
CXXFLAGS += -I/usr/include/libxml2
CXXFLAGS += -DHAVE_INLINE -DBOINCIFIED
CXXFLAGS += -DUSE_METAL
CXXFLAGS += -std=c++17

MTLFLAGS += -std=metal3.0 -fno-fast-math -Werror

DEPS = Makefile.macos.metal
OBJS = demod_binary.o demod_binary_metal.o demod_binary_metal_fft.o demod_binary_hs_metal.o hs_common.o rngmed.o \
       erp_boinc_ipc.o erp_getopt.o erp_getopt1.o erp_utilities.o
EINSTEINBINARY_TARGET ?= einsteinbinary_BRP4_macos
TARGET = $(EINSTEINBINARY_TARGET)
METALLIB = default.metallib

default: release
debug: erp_git_version.h $(TARGET)
release: clean erp_git_version.h $(TARGET)

debug: CXXFLAGS += -DLOGLEVEL=debug -pg -ggdb3 -O0 -Wall
release: CXXFLAGS += -DNDEBUG -DLOGLEVEL=info -ggdb3 -O3 -Wall

$(TARGET): $(DEPS) $(EINSTEIN_RADIO_SRC)/erp_boinc_wrapper.cpp $(OBJS)
	$(CXX) -g $(CXXFLAGS) $(LDFLAGS) $(EINSTEIN_RADIO_SRC)/erp_boinc_wrapper.cpp -o $(TARGET) $(OBJS) $(LIBS)

demod_binary.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/demod_binary.c $(EINSTEIN_RADIO_SRC)/demod_binary.h
	$(CXX) -g $(CXXFLAGS) -c $(EINSTEIN_RADIO_SRC)/demod_binary.c

# both kernel files are compiled into ONE embedded .metallib -- see
# demod_binary_metal.cpp / demod_binary_hs_metal.cpp, which both look their
# kernels up from the same g_metalLibrary
$(METALLIB).h: $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal.metal $(EINSTEIN_RADIO_SRC)/metal/harmonic_summing_kernel.metal
	xcrun metal $(MTLFLAGS) -c $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal.metal -o demod_binary_metal.ir
	xcrun metal $(MTLFLAGS) -c $(EINSTEIN_RADIO_SRC)/metal/harmonic_summing_kernel.metal -o harmonic_summing_kernel.ir
	xcrun metallib demod_binary_metal.ir harmonic_summing_kernel.ir -o $(METALLIB)
	xxd -i $(METALLIB) > $(METALLIB).h

# -x objective-c++: metal-cpp's <Foundation/Foundation.hpp>-style includes
# only resolve to metal-cpp's own headers (instead of the real Foundation
# framework's identically-shaped path triggering Clang's Darwin framework
# header lookup and shadowing them) when compiled in Objective-C++ mode --
# confirmed empirically on real hardware; plain C++ mode picks the wrong
# header with the same -I flags. Harmless for these files: no Objective-C
# syntax is used, this only changes header search behavior.
demod_binary_metal.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal.cpp $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal.h $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal_fft.h $(METALLIB).h
	$(CXX) -g $(CXXFLAGS) -x objective-c++ -I. -c $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal.cpp

# Objective-C++: bridges MPSGraph (Objective-C only, no metal-cpp bindings)
# to the plain-C++ rest of the backend -- see demod_binary_metal_fft.h.
demod_binary_metal_fft.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal_fft.mm $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal_fft.h
	$(CXX) -g $(CXXFLAGS) -fobjc-arc -x objective-c++ -c $(EINSTEIN_RADIO_SRC)/metal/demod_binary_metal_fft.mm -o demod_binary_metal_fft.o

# -x objective-c++ -I.: see the comment on the demod_binary_metal.o rule
# above -- empirically, -I. must be present alongside -x objective-c++ for
# the metal-cpp Foundation/Metal/QuartzCore headers to resolve correctly
# here too (confirmed on real hardware; dropping either one regresses to
# the same wrong-header error), even though this file has no
# default.metallib.h dependency of its own.
demod_binary_hs_metal.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/metal/demod_binary_hs_metal.cpp $(EINSTEIN_RADIO_SRC)/metal/demod_binary_hs_metal.h
	$(CXX) -g $(CXXFLAGS) -x objective-c++ -I. -c $(EINSTEIN_RADIO_SRC)/metal/demod_binary_hs_metal.cpp

hs_common.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/hs_common.c $(EINSTEIN_RADIO_SRC)/hs_common.h
	$(CXX) -g $(CXXFLAGS) -c $(EINSTEIN_RADIO_SRC)/hs_common.c

rngmed.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/rngmed.c $(EINSTEIN_RADIO_SRC)/rngmed.h
	$(CXX) -g $(CXXFLAGS) -c $(EINSTEIN_RADIO_SRC)/rngmed.c

erp_boinc_ipc.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/erp_boinc_ipc.cpp $(EINSTEIN_RADIO_SRC)/erp_boinc_ipc.h
	$(CXX) -g $(CXXFLAGS) -c $(EINSTEIN_RADIO_SRC)/erp_boinc_ipc.cpp

erp_getopt.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/erp_getopt.c $(EINSTEIN_RADIO_SRC)/erp_getopt.h $(EINSTEIN_RADIO_SRC)/erp_getopt_int.h
	$(CXX) -g $(CXXFLAGS) -c $(EINSTEIN_RADIO_SRC)/erp_getopt.c

erp_getopt1.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/erp_getopt1.c $(EINSTEIN_RADIO_SRC)/erp_getopt.h $(EINSTEIN_RADIO_SRC)/erp_getopt_int.h
	$(CXX) -g $(CXXFLAGS) -c $(EINSTEIN_RADIO_SRC)/erp_getopt1.c

erp_utilities.o: $(DEPS) $(EINSTEIN_RADIO_SRC)/erp_utilities.cpp $(EINSTEIN_RADIO_SRC)/erp_utilities.h
	$(CXX) -g $(CXXFLAGS) -c $(EINSTEIN_RADIO_SRC)/erp_utilities.cpp

erp_git_version.h:
	@echo "#ifndef ERP_GIT_VERSION_H" > $@
	@echo "#define ERP_GIT_VERSION_H" >> $@
	@echo "#define ERP_GIT_VERSION \"$(ERP_VERSION)\"" >> $@

install:
	mkdir -p $(EINSTEIN_RADIO_INSTALL)/../dist
	cp $(TARGET) $(EINSTEIN_RADIO_INSTALL)/../dist

clean:
	rm -f $(OBJS) $(TARGET) $(METALLIB) $(METALLIB).h *.ir erp_git_version.h
