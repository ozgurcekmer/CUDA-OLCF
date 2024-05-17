#include "../include/GpuSolver.h"

// Matrix row-sum kernel
template<typename T>
__global__
void row_sums(T* A, T* sums, size_t ds)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < ds)
	{
		T sum = 0.0;
		for (size_t i = 0; i < ds; ++i)
		{
			sum += A[idx * ds + i];
		}
		sums[idx] = sum;
	}
}

// Matrix column-sum kernel
template<typename T>
__global__
void column_sums(T* A, T* sums, size_t ds)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < ds)
	{
		T sum = 0.0;
		for (size_t i = 0; i < ds; ++i)
		{
			sum += A[i * ds + idx];
		}
		sums[idx] = sum;
	}
}

template<typename T>
inline void GpuSolver<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE_A);
	gpuMalloc(&dRowSums, SIZE_SUMS);
	gpuMalloc(&dColSums, SIZE_SUMS);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSolver<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE_A, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dRowSums, this->RowSums.data(), SIZE_SUMS, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dColSums, this->ColSums.data(), SIZE_SUMS, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSolver<T>::copyD2H()
{
	gpuMemcpy(this->RowSums.data(), this->dRowSums, SIZE_SUMS, gpuMemcpyDeviceToHost);
	gpuMemcpy(this->ColSums.data(), this->dColSums, SIZE_SUMS, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSolver<T>::~GpuSolver()
{
	gpuFree(dA);
	gpuFree(dRowSums);
	gpuFree(dColSums);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void GpuSolver<T>::rowSums()
{
	row_sums <<< GRID_SIZE, BLOCK_SIZE >>> (dA, dRowSums, DSIZE);
}

template<typename T>
void GpuSolver<T>::colSums()
{
	column_sums <<< GRID_SIZE, BLOCK_SIZE >>>(dA, dColSums, DSIZE);
}

template<typename T>
void GpuSolver<T>::solver()
{
	deviceAllocations();
	copyH2D();
	rowSums();
	colSums();
	copyD2H();
}


template void GpuSolver<float>::deviceAllocations();
template void GpuSolver<double>::deviceAllocations();
template void GpuSolver<float>::copyH2D();
template void GpuSolver<double>::copyH2D();
template void GpuSolver<float>::copyD2H();
template void GpuSolver<double>::copyD2H();
template void GpuSolver<float>::rowSums();
template void GpuSolver<double>::rowSums();
template void GpuSolver<float>::colSums();
template void GpuSolver<double>::colSums();
template void GpuSolver<float>::solver();
template void GpuSolver<double>::solver();
template GpuSolver<float>::~GpuSolver();
template GpuSolver<double>::~GpuSolver();







