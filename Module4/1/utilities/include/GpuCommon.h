#pragma once

#include <iostream>

#if defined(USEHIP)
	#include <hip/hip_runtime.h>
	#include <hip/hip_runtime_api.h>

	#define __GPU_API__ "HIP"

    #define gpuDeviceProp_t hipDeviceProp_t
    #define gpuError_t hipError_t
    #define gpuFree hipFree
    #define gpuGetDevice hipGetDevice
    #define gpuGetDeviceProperties hipGetDeviceProperties
    #define gpuGetErrorString hipGetErrorString    
    #define gpuGetLastError hipGetLastError
    #define gpuMalloc hipMalloc
    #define gpuMemcpy hipMemcpy
    #define gpuSuccess hipSuccess
    


#elif defined(USECUDA)
	#include <cuda_runtime.h>
	#include <cuda_runtime_api.h>
	#include <device_launch_parameters.h>

	#define __GPU_API__ "CUDA"

    #define gpuDeviceProp_t cudaDeviceProp
    #define gpuError_t cudaError_t
    #define gpuFree cudaFree
    #define gpuGetDevice cudaGetDevice
    #define gpuGetDeviceProperties cudaGetDeviceProperties
    #define gpuGetErrorString cudaGetErrorString
    #define gpuGetLastError cudaGetLastError
    #define gpuMalloc cudaMalloc
    #define gpuMemcpy cudaMemcpy
    #define gpuSuccess cudaSuccess
    

#endif


#define gpuCheckErrors(msg)                                     \
    do                                                          \
    {                                                           \
        gpuError_t __err = gpuGetLastError();                   \    
        if (__err != gpuSuccess)                                \
        {                                                       \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
            msg, gpuGetErrorString(__err),                      \
                __FILE__, __LINE__);                            \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)                                                 \

#define gpuReportDevice()                                       \
    {                                                           \
            int device;                                         \
            gpuDeviceProp_t devProp;                            \
            gpuGetDevice(&device);                              \
            gpuGetDeviceProperties(&devprop, device);           \
            std::cout << "[@" << __func__ << " L" << __LINE__   \
                << "] : Using" << __GPU_API__                   \
                << " Device " << device << ": "                 \
                << devprop.name << std::endl;                   \
    }                                                           \