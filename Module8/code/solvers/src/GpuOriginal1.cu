#include "../include/GpuOriginal1.h"

#ifdef KERNELTIME
#include <omp.h>
#endif

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuOriginal1(T* __restrict__ a, T* __restrict__ b)
{
    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;
    //int width = gridDim.x * TILE_DIM;
    int width = N;

    if (row < N && col < N)
    {
        b[INDX(row, col, width)] = a[INDX(col, row, N)];
    }

}

template<typename T>
void GpuOriginal1<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, BYTES);
    gpuMalloc(&dB, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuOriginal1<T>::copyH2D()
{
    gpuMemcpy(dA, this->A.data(), BYTES, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuOriginal1<T>::copyD2H()
{
    gpuMemcpy(this->B.data(), dB, BYTES, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuOriginal1<T>::~GpuOriginal1()
{
    gpuFree(dA);
    gpuFree(dB);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuOriginal1<T>::solver()
{

    deviceAllocations();

    copyH2D();
    dim3 threads(TILE_DIM, TILE_DIM, 1);
    dim3 blocks(N / TILE_DIM, N / TILE_DIM, 1);
#ifdef KERNELTIME
    auto t0 = omp_get_wtime();
    gpuOriginal1<T> << < blocks, threads >> > (dA, dB);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    auto t1 = omp_get_wtime();
    cout << "Kernel runtime: " << (t1 - t0) * 1000.0 << " ms." << endl;
#else
    gpuOriginal1<T> << < blocks, threads >> > (dA, dB);
    gpuCheckErrors("gpu kernel launch failure");
#endif
    copyD2H();

}

template void GpuOriginal1<float>::solver();
template void GpuOriginal1<double>::solver();
template void GpuOriginal1<float>::deviceAllocations();
template void GpuOriginal1<double>::deviceAllocations();
template void GpuOriginal1<float>::copyH2D();
template void GpuOriginal1<double>::copyH2D();
template GpuOriginal1<float>::~GpuOriginal1();
template GpuOriginal1<double>::~GpuOriginal1();
