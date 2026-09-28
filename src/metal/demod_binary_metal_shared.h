#ifndef DEMOD_BINARY_METAL_SHARED_H
#define DEMOD_BINARY_METAL_SHARED_H

// Metal device/queue/compiled-kernel-library handles, set up once by
// initialize_metal() in demod_binary_metal.cpp and shared with the
// harmonic-summing backend (demod_binary_hs_metal.cpp) -- both kernel
// files are compiled into one embedded .metallib (see Makefile.macos.metal),
// so both look kernels up from the same MTL::Library. The CUDA port doesn't
// need an equivalent of this because its context is implicit/thread-local
// across CUDA driver API calls; Metal has no such implicit context, so this
// is explicit here.

namespace MTL {
class Device;
class CommandQueue;
class Library;
}  // namespace MTL

extern MTL::Device *g_metalDevice;
extern MTL::CommandQueue *g_metalQueue;
extern MTL::Library *g_metalLibrary;

#endif
