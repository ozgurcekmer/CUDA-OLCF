#include "../include/GpuReduceA.h"
#include <iostream>

// Matrix column-sum kernel
__global__
void reduceAtomic(const float* A, float* out, size_t ds)
{
	__shared__ float Tile[BLOCK_SIZE];
	size_t tID = threadIdx.x;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	Tile[tID] = 0.0;
	const size_t STRIDE = gridDim.x * blockDim.x;

	while (idx < ds)
	{
		Tile[tID] += A[idx];
		idx += STRIDE;
	}

	for (auto s = BLOCK_SIZE / 2; s > 0; s /= 2)
	{
		__syncthreads();
		if (tID < s)
		{
			Tile[tID] += Tile[tID + s];
		}
	}

	if (tID == 0)
	{
		atomicAdd(out, Tile[0]);
	}
}

template<typename T>
inline void GpuReduceA<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE_A);
	gpuMalloc(&dSum, SIZE_SUMS);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuReduceA<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE_A, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dSum, this->Sum.data(), SIZE_SUMS, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuReduceA<T>::copyD2H()
{
	gpuMemcpy(this->Sum.data(), this->dSum, SIZE_SUMS, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
void GpuReduceA<T>::launchSetup()
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
GpuReduceA<T>::~GpuReduceA()
{
	gpuFree(dA);
	gpuFree(dSum);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void GpuReduceA<T>::solver()
{
	deviceAllocations();
	copyH2D();
	launchSetup();
	reduceAtomic << < gridSize, BLOCK_SIZE >> > ((float*)dA, (float*)dSum, N);
	copyD2H();
}

template void GpuReduceA<float>::deviceAllocations();
template void GpuReduceA<double>::deviceAllocations();
template void GpuReduceA<float>::copyH2D();
template void GpuReduceA<double>::copyH2D();
template void GpuReduceA<float>::copyD2H();
template void GpuReduceA<double>::copyD2H();
template void GpuReduceA<float>::solver();
template void GpuReduceA<double>::solver();
template GpuReduceA<float>::~GpuReduceA();
template GpuReduceA<double>::~GpuReduceA();







