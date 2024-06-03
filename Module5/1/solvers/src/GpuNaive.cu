#include "../include/GpuNaive.h"
#include <iostream>

/*
// Matrix column-sum kernel
template<typename T>
__global__
void sumsKernel(const T* A, T* out, size_t ds)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < ds)
	{
		atomicAdd(out, A[idx]);
	}
	
}
*/

__global__
void reduceAtomicNaive(const float* A, float* out, size_t ds)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < ds)
	{
		atomicAdd(out, A[idx]);
	}
}

template<typename T>
inline void GpuNaive<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE_A);
	gpuMalloc(&dSum, SIZE_SUMS);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuNaive<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE_A, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dSum, this->Sum.data(), SIZE_SUMS, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuNaive<T>::copyD2H()
{
	gpuMemcpy(this->Sum.data(), this->dSum, SIZE_SUMS, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuNaive<T>::~GpuNaive()
{
	gpuFree(dA);
	gpuFree(dSum);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void GpuNaive<T>::solver()
{
	deviceAllocations();
	copyH2D();
	reduceAtomicNaive << < GRID_SIZE, BLOCK_SIZE >> > ((float*)dA, (float*)dSum, N);
	copyD2H();
}

template void GpuNaive<float>::deviceAllocations();
template void GpuNaive<double>::deviceAllocations();
template void GpuNaive<float>::copyH2D();
template void GpuNaive<double>::copyH2D();
template void GpuNaive<float>::copyD2H();
template void GpuNaive<double>::copyD2H();
template void GpuNaive<float>::solver();
template void GpuNaive<double>::solver();
template GpuNaive<float>::~GpuNaive();
template GpuNaive<double>::~GpuNaive();







