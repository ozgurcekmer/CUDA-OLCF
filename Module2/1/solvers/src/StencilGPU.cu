#include "../include/StencilGPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuStencil(T *in, T *out)
{
    __shared__ 
    T temp[BLOCK_SIZE + 2*RADIUS];

    int iGlobal = threadIdx.x + blockIdx.x * blockDim.x;
    int iLocal = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[iLocal] = in[iGlobal];
    if (threadIdx.x < RADIUS)
    {
        temp[iLocal - RADIUS] = in[iGlobal - RADIUS];
        temp[iLocal + BLOCK_SIZE] = in[iGlobal + BLOCK_SIZE];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply stencil
    T result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; ++offset)
    {
        result += temp[iLocal + offset];
    }

    // Store the result
    out[iGlobal] = result;

}

template<typename T>
void StencilGPU<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dIn, SIZE);
    gpuMalloc(&dOut, SIZE);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void StencilGPU<T>::copyH2D()
{
    gpuMemcpy(dIn, this->in.data(), SIZE, gpuMemcpyHostToDevice);
    //gpuMemcpy(dOut, this->out.data(), SIZE, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void StencilGPU<T>::copyD2H()
{
    gpuMemcpy(this->out.data(), dOut, SIZE, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
StencilGPU<T>::~StencilGPU()
{
    // Deallocate device vectors
    gpuFree(dIn);
    gpuFree(dOut);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void StencilGPU<T>::stencil()
{
    deviceAllocations();
    copyH2D();
    cout << "Block size: " << BLOCK_SIZE << endl;
    cout << "Grid size : " << GRID_SIZE << endl;

    // cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), cudaFuncCachePreferL1);

    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);
    
    gpuStencil <<< GRID_SIZE, BLOCK_SIZE >>> (dIn + RADIUS, dOut + RADIUS);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
}

template void StencilGPU<float>::stencil();
template void StencilGPU<double>::stencil();
template void StencilGPU<float>::deviceAllocations();
template void StencilGPU<double>::deviceAllocations();
template void StencilGPU<float>::copyH2D();
template void StencilGPU<double>::copyH2D();
template void StencilGPU<float>::copyD2H();
template void StencilGPU<double>::copyD2H();
template StencilGPU<float>::~StencilGPU();
template StencilGPU<double>::~StencilGPU();
