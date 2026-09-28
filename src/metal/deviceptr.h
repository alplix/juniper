#ifndef __DEVICEPTR_H__
#define __DEVICEPTR_H__

#ifdef __cplusplus
extern "C" {
#endif

// opaque device-side handle; the Metal backend casts this to/from MTL::Buffer*
// (a real pointer, not a driver-API integer handle like CUDA's CUdeviceptr, but
// the same fixed-width slot in DIfloatPtr fits it without change)
typedef void* device_ptr_t;

#ifdef __cplusplus
}
#endif

#endif
