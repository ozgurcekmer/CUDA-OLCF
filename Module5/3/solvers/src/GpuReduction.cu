#include "../include/GpuReduction.h"

// Matrix row-sum kernel
template<typename T>
__global__
void row_sums_Red(const T* A, T* sums, size_t ds)
{
	// extern __shared__ T sA[];
	// Dynamic shared memory is not available with templates currently

	__shared__ T sA[BLOCK_SIZE];
	int iLoc = threadIdx.x; // Local ID of the thread
	int iBlock = blockIdx.x; // Block ID
	int iTemp = iLoc;
	sA[iLoc] = 0.0;
	
	// block-stride loop to load data
	while (iTemp < ds) 
	{
		sA[iLoc] += A[iBlock * ds + iTemp];
		iTemp += blockDim.x;
	}

	// Parallel sweep
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		__syncthreads();
		if (iLoc < s)
		{
			sA[iLoc] += sA[iLoc + s];
		}
	}

	// Store the result per block
	if (iLoc == 0)
	{
		sums[iBlock] = sA[0];
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

// EXTRA: Parallel reduction in column-sum kernel
template<typename T>
__global__
void column_sums_Red(const T* A, T* sums, size_t ds)
{
	__shared__ T sA[BLOCK_SIZE];
	int iLoc = threadIdx.x; // Local ID of the thread
	int iBlock = blockIdx.x; // Block ID
	int iTemp = iLoc;
	sA[iLoc] = 0.0;

	// block-stride loop to load data
	while (iTemp < ds)
	{
		sA[iLoc] += A[iTemp * ds + iBlock];
		iTemp += blockDim.x;
	}

	// Parallel sweep
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		__syncthreads();
		if (iLoc < s)
		{
			sA[iLoc] += sA[iLoc + s];
		}
	}

	// Store the result per block
	if (iLoc == 0)
	{
		sums[iBlock] = sA[0];
	}
}

template<typename T>
inline void GpuReduction<T>::deviceAllocations()
{
	cudaMalloc(&dA, SIZE_A);
	cudaMalloc(&dRowSums, SIZE_SUMS);
	cudaMalloc(&dColSums, SIZE_SUMS);
	cudaCheckErrors("cudaMalloc failure");
}

template<typename T>
void GpuReduction<T>::copyH2D()
{
	cudaMemcpy(this->dA, this->A.data(), SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(this->dRowSums, this->RowSums.data(), SIZE_SUMS, cudaMemcpyHostToDevice);
	cudaMemcpy(this->dColSums, this->ColSums.data(), SIZE_SUMS, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy H2D failure");
}

template<typename T>
void GpuReduction<T>::copyD2H()
{
	cudaMemcpy(this->RowSums.data(), this->dRowSums, SIZE_SUMS, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->ColSums.data(), this->dColSums, SIZE_SUMS, cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy D2H failure");
}

template<typename T>
GpuReduction<T>::~GpuReduction()
{
	cudaFree(dA);
	cudaFree(dRowSums);
	cudaFree(dColSums);
	cudaCheckErrors("cudaFree failure");
}

template<typename T>
void GpuReduction<T>::rowSums()
{
	// SIZE_SHARED = BLOCK_SIZE * sizeof(T);
	// row_sums_Red << < DSIZE, BLOCK_SIZE, SIZE_SHARED >> > (dA, dRowSums, DSIZE);
	// Dynamic shared memory is not available with templates currently
	row_sums_Red << < DSIZE, BLOCK_SIZE >> > (dA, dRowSums, DSIZE);
}

template<typename T>
void GpuReduction<T>::colSums()
{
	column_sums_Red << < GRID_SIZE, BLOCK_SIZE >> > (dA, dColSums, DSIZE);
}

template<typename T>
void GpuReduction<T>::solver()
{
	deviceAllocations();
	copyH2D();
	rowSums();
	colSums();
	copyD2H();
}

template void GpuReduction<float>::deviceAllocations();
template void GpuReduction<double>::deviceAllocations();
template void GpuReduction<float>::copyH2D();
template void GpuReduction<double>::copyH2D();
template void GpuReduction<float>::copyD2H();
template void GpuReduction<double>::copyD2H();
template void GpuReduction<float>::rowSums();
template void GpuReduction<double>::rowSums();
template void GpuReduction<float>::colSums();
template void GpuReduction<double>::colSums();
template void GpuReduction<float>::solver();
template void GpuReduction<double>::solver();
template GpuReduction<float>::~GpuReduction();
template GpuReduction<double>::~GpuReduction();
