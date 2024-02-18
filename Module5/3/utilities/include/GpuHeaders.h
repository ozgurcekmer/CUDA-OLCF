#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define cudaCheckErrors(msg)                                     \
do { \
    cudaError_t __err = cudaGetLastError(); \
    __err != cudaSuccess ? \
    fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n*** FAILED - ABORTING\n", \
        msg, cudaGetErrorString(__err), __FILE__, __LINE__), exit(1) : \
    void(0); \
} while (0)

#define cudaReportDevice()                                       \
    {                                                           \
            int device;                                         \
            cudaDeviceProp_t devProp;                            \
            cudaGetDevice(&device);                              \
            cudaGetDeviceProperties(&devprop, device);           \
            std::cout << "[@" << __func__ << " L" << __LINE__   \
                << "] : Using Device: "                    \
                << device << ": "                 \
                << devprop.name << std::endl;                   \
    }                                                           \
