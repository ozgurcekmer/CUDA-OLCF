#include "../include/GpuSolver.h"

// Matrix row-sum kernel
template<typename T>
__global__
void row_sums(const T* A, T* sums, size_t ds)
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
void column_sums(const T* A, T* sums, size_t ds)
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
	cudaMalloc(&dA, SIZE_A);
	cudaMalloc(&dRowSums, SIZE_SUMS);
	cudaMalloc(&dColSums, SIZE_SUMS);
	cudaCheckErrors("cudaMalloc failure");
}

template<typename T>
void GpuSolver<T>::copyH2D()
{
	cudaMemcpy(this->dA, this->A.data(), SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(this->dRowSums, this->RowSums.data(), SIZE_SUMS, cudaMemcpyHostToDevice);
	cudaMemcpy(this->dColSums, this->ColSums.data(), SIZE_SUMS, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy H2D failure");
}

template<typename T>
void GpuSolver<T>::copyD2H()
{
	cudaMemcpy(this->RowSums.data(), this->dRowSums, SIZE_SUMS, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->ColSums.data(), this->dColSums, SIZE_SUMS, cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy D2H failure");
}

template<typename T>
GpuSolver<T>::~GpuSolver()
{
	cudaFree(dA);
	cudaFree(dRowSums);
	cudaFree(dColSums);
	cudaCheckErrors("cudaFree failure");
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







