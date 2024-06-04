#include "../include/GpuReduce.h"
#include <iostream>

// Matrix column-sum kernel
__global__
void reduce2Steps(const float* A, float* out, size_t ds)
{
	__shared__ float Tile[BLOCK_SIZE];
	size_t tID = threadIdx.x;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	Tile[tID] = 0.0;
	const size_t STRIDE = gridDim.x * blockDim.x;

	while (idx < ds)
	{
		Tile[tID] = (Tile[tID] > A[idx]) ? Tile[tID] : A[idx];
		idx += STRIDE;
	}

	for (auto s = blockDim.x / 2; s > 0; s /= 2)
	{
		__syncthreads();
		if (tID < s)
		{
			Tile[tID] = (Tile[tID] > Tile[tID + s]) ? Tile[tID] : Tile[tID + s];
		}
	}

	if (tID == 0)
	{
		out[blockIdx.x] = Tile[0];
	}

}

template<typename T>
inline void GpuReduce<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE_A);
	gpuMalloc(&dMax, SIZE_RED);
	gpuMalloc(&dMax0, SIZE_RED0);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuReduce<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE_A, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dMax, this->Max.data(), SIZE_RED, gpuMemcpyHostToDevice);
	gpuMemset(this->dMax0, 0, SIZE_RED0);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuReduce<T>::copyD2H()
{
	gpuMemcpy(this->Max.data(), this->dMax, SIZE_RED, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
void GpuReduce<T>::launchSetup()
{
	auto blocksPerSM = 2048 / BLOCK_SIZE;
	int devID;
	int numSMs;
	gpuGetDevice(&devID);

	gpuDeviceGetAttribute(&numSMs, gpuDevAttrMultiProcessorCount, devID);
	std::cout << "There are " << numSMs << " SMs in this device." << std::endl;
	std::cout << "Blocks per SM: " << blocksPerSM << std::endl;

	gridSize = blocksPerSM * numSMs;
	std::cout << "Grid Size: " << gridSize << std::endl;
	std::cout << "Block Size: " << BLOCK_SIZE << std::endl;
}

template<typename T>
GpuReduce<T>::~GpuReduce()
{
	gpuFree(dA);
	gpuFree(dMax);
	gpuFree(dMax0);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void GpuReduce<T>::solver()
{
	deviceAllocations();
	copyH2D();
	launchSetup();
	reduce2Steps << < gridSize, BLOCK_SIZE >> > ((float*)dA, (float*)dMax0, N);
	reduce2Steps << < 1, BLOCK_SIZE >> > ((float*)dMax0, (float*)dMax, gridSize);
	copyD2H();
}

template void GpuReduce<float>::deviceAllocations();
template void GpuReduce<double>::deviceAllocations();
template void GpuReduce<float>::copyH2D();
template void GpuReduce<double>::copyH2D();
template void GpuReduce<float>::copyD2H();
template void GpuReduce<double>::copyD2H();
template void GpuReduce<float>::solver();
template void GpuReduce<double>::solver();
template GpuReduce<float>::~GpuReduce();
template GpuReduce<double>::~GpuReduce();







