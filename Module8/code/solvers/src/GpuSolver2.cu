#include "../include/GpuSolver2.h"

#ifdef KERNELTIME
#include <omp.h>
#endif

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSolver2(T* __restrict__ a, T* __restrict__ b)
{
    __shared__ T tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;
    //int width = gridDim.x * TILE_DIM;
    int width = N;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = a[INDX(col + j, row, N)];
    }

    __syncthreads();

    row = blockIdx.y * TILE_DIM + threadIdx.x;
    col = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        b[INDX(col + j, row, width)] = tile[threadIdx.x][threadIdx.y + j];
    }
    

}

template<typename T>
void GpuSolver2<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, BYTES);
    gpuMalloc(&dB, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSolver2<T>::copyH2D()
{
    gpuMemcpy(dA, this->A.data(), BYTES, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSolver2<T>::copyD2H()
{
    gpuMemcpy(this->B.data(), dB, BYTES, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSolver2<T>::~GpuSolver2()
{
    gpuFree(dA);
    gpuFree(dB);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSolver2<T>::solver()
{

    deviceAllocations();

    copyH2D();
    dim3 threads(TILE_DIM, BLOCK_ROWS, 1);
    dim3 blocks(N / TILE_DIM, N / TILE_DIM, 1);
#ifdef KERNELTIME
    auto t0 = omp_get_wtime();
    gpuSolver2<T> << < blocks, threads >> > (dA, dB);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    auto t1 = omp_get_wtime();
    cout << "Kernel runtime: " << (t1 - t0) * 1000.0 << " ms." << endl;
#else
    gpuSolver2<T> << < blocks, threads >> > (dA, dB);
    gpuCheckErrors("gpu kernel launch failure");
#endif
    copyD2H();

}

template void GpuSolver2<float>::solver();
template void GpuSolver2<double>::solver();
template void GpuSolver2<float>::deviceAllocations();
template void GpuSolver2<double>::deviceAllocations();
template void GpuSolver2<float>::copyH2D();
template void GpuSolver2<double>::copyH2D();
template GpuSolver2<float>::~GpuSolver2();
template GpuSolver2<double>::~GpuSolver2();
