#include "../include/MatrixMultShared.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void sharedMatrixMult(T* a, T* b, T* c)
{
    __shared__ T aShared[blockSize * blockSize];
    __shared__ T bShared[blockSize * blockSize];

    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    size_t iLocal = threadIdx.y;
    size_t jLocal = threadIdx.x;


    if (i < M && j < N)
    {
        T cTemp{ 0.0 };
        for (int k0 = 0; k0 < (K / blockSize); ++k0)
        {
            aShared[iLocal * blockSize + jLocal] = a[i * K + (k0 * blockSize + jLocal)];
            bShared[iLocal * blockSize + jLocal] = b[(k0 * blockSize + iLocal) * N + j];

            __syncthreads();

            for (int k = 0; k < blockSize; ++k)
            {
                cTemp += aShared[iLocal * blockSize + k] * bShared[k * blockSize + jLocal];
            }
            __syncthreads();

        }
        c[i * N + j] = cTemp;
    }
}

template<typename T>
void MatrixMultShared<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, SIZE_A);
    gpuMalloc(&dB, SIZE_B);
    gpuMalloc(&dC, SIZE_C);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void MatrixMultShared<T>::copyH2D()
{
    gpuMemcpy(dA, this->a.data(), SIZE_A, gpuMemcpyHostToDevice);
    gpuMemcpy(dB, this->b.data(), SIZE_B, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void MatrixMultShared<T>::copyD2H()
{
    gpuMemcpy(this->c.data(), dC, SIZE_C, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
MatrixMultShared<T>::~MatrixMultShared()
{
    // Deallocate device vectors
    gpuFree(dA);
    gpuFree(dB);
    gpuFree(dC);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void MatrixMultShared<T>::matrixMult()
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
    
    sharedMatrixMult <<< GRID_SIZE, BLOCK_SIZE >>> (dA, dB, dC);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
}

template void MatrixMultShared<float>::matrixMult();
template void MatrixMultShared<double>::matrixMult();
template void MatrixMultShared<float>::deviceAllocations();
template void MatrixMultShared<double>::deviceAllocations();
template void MatrixMultShared<float>::copyH2D();
template void MatrixMultShared<double>::copyH2D();
template void MatrixMultShared<float>::copyD2H();
template void MatrixMultShared<double>::copyD2H();
template MatrixMultShared<float>::~MatrixMultShared();
template MatrixMultShared<double>::~MatrixMultShared();
