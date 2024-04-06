#include "../include/BlasGPU.h"

using std::cout;
using std::endl;
using std::vector;

template<typename T>
void BlasGPU<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, SIZE_A);
    gpuMalloc(&dB, SIZE_B);
    gpuMalloc(&dC, SIZE_C);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void BlasGPU<T>::copyH2D()
{
    gpuMemcpy(dA, this->a.data(), SIZE_A, gpuMemcpyHostToDevice);
    gpuMemcpy(dB, this->b.data(), SIZE_B, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void BlasGPU<T>::copyD2H()
{
    gpuMemcpy(this->c.data(), dC, SIZE_C, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
BlasGPU<T>::~BlasGPU()
{
    // Deallocate device vectors
    gpuFree(dA);
    gpuFree(dB);
    gpuFree(dC);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void BlasGPU<T>::matrixMult()
{
    deviceAllocations();
    copyH2D();
    // cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), cudaFuncCachePreferL1);

    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);
    gpublasHandle_t handle;
    gpublasCreate(&handle);
    
    if (sizeof(T) == sizeof(float))
    {
        const float ALPHA = 1.0;
        const float BETA = 0.0;
        gpublasSgemm(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, N, M, K, &ALPHA, (const float*)dB, N, (const float*)dA, K, &BETA, (float*)dC, N);
    }
    else
    {
        const double ALPHA = 1.0;
        const double BETA = 0.0;
        gpublasDgemm(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, N, M, K, &ALPHA, (const double*)dB, N, (const double*)dA, K, &BETA, (double*)dC, N);
    }

    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
    gpublasDestroy(handle);
}

template void BlasGPU<float>::matrixMult();
template void BlasGPU<double>::matrixMult();
template void BlasGPU<float>::deviceAllocations();
template void BlasGPU<double>::deviceAllocations();
template void BlasGPU<float>::copyH2D();
template void BlasGPU<double>::copyH2D();
template BlasGPU<float>::~BlasGPU();
template BlasGPU<double>::~BlasGPU();
