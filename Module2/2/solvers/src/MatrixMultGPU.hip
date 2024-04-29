#include "../include/MatrixMultGPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuMatrixMult(T* a, T* b, T* c)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    T cTemp{ 0.0 };

    for (auto k = 0; k < K; ++k)
    {
        cTemp += a[i * K + k] * b[k * N + j];
    }
    c[i * N + j] = cTemp;

}

template<typename T>
void MatrixMultGPU<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, SIZE_A);
    gpuMalloc(&dB, SIZE_B);
    gpuMalloc(&dC, SIZE_C);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void MatrixMultGPU<T>::copyH2D()
{
    gpuMemcpy(dA, this->a.data(), SIZE_A, gpuMemcpyHostToDevice);
    gpuMemcpy(dB, this->b.data(), SIZE_B, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void MatrixMultGPU<T>::copyD2H()
{
    gpuMemcpy(this->c.data(), dC, SIZE_C, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
MatrixMultGPU<T>::~MatrixMultGPU()
{
    // Deallocate device vectors
    gpuFree(dA);
    gpuFree(dB);
    gpuFree(dC);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void MatrixMultGPU<T>::matrixMult()
{
    deviceAllocations();
    copyH2D();
    const dim3 BLOCK_SIZE(blockSize, blockSize);
    const dim3 GRID_SIZE(N / BLOCK_SIZE.x, M / BLOCK_SIZE.y);
    cout << "Block size: (" << BLOCK_SIZE.x << ", " << BLOCK_SIZE.x << ")" << endl;
    cout << "Grid size : (" << GRID_SIZE.x << ", " << GRID_SIZE.y << ")" << endl;

    // cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), cudaFuncCachePreferL1);

    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);
    
    gpuMatrixMult <<< GRID_SIZE, BLOCK_SIZE >>> (dA, dB, dC);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
}

template void MatrixMultGPU<float>::matrixMult();
template void MatrixMultGPU<double>::matrixMult();
template void MatrixMultGPU<float>::deviceAllocations();
template void MatrixMultGPU<double>::deviceAllocations();
template void MatrixMultGPU<float>::copyH2D();
template void MatrixMultGPU<double>::copyH2D();
template MatrixMultGPU<float>::~MatrixMultGPU();
template MatrixMultGPU<double>::~MatrixMultGPU();
