#include "../include/GpuSolver.h"

// Matrix column-sum kernel
template<typename T>
__global__
void sumsKernel(const T* A, T* Sum, size_t ds)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < ds)
	{
		// To be written
	}
}

template<typename T>
inline void GpuSolver<T>::deviceAllocations()
{
	cudaMalloc(&dA, SIZE_A);
	cudaMalloc(&dSum, SIZE_SUMS);
	cudaCheckErrors("cudaMalloc failure");
}

template<typename T>
void GpuSolver<T>::copyH2D()
{
	cudaMemcpy(this->dA, this->A.data(), SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(this->dSum, this->Sum.data(), SIZE_SUMS, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy H2D failure");
}

template<typename T>
void GpuSolver<T>::copyD2H()
{
	cudaMemcpy(this->Sum.data(), this->dSum, SIZE_SUMS, cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy D2H failure");
}

template<typename T>
GpuSolver<T>::~GpuSolver()
{
	cudaFree(dA);
	cudaFree(dSum);
	cudaCheckErrors("cudaFree failure");
}

template<typename T>
void GpuSolver<T>::solver()
{
	deviceAllocations();
	copyH2D();
	sumsKernel << < GRID_SIZE, BLOCK_SIZE >> > (dA, dSum, N);
	copyD2H();
}

template void GpuSolver<float>::deviceAllocations();
template void GpuSolver<double>::deviceAllocations();
template void GpuSolver<float>::copyH2D();
template void GpuSolver<double>::copyH2D();
template void GpuSolver<float>::copyD2H();
template void GpuSolver<double>::copyD2H();
template void GpuSolver<float>::solver();
template void GpuSolver<double>::solver();
template GpuSolver<float>::~GpuSolver();
template GpuSolver<double>::~GpuSolver();







