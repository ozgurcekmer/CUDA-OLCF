#include "../include/GpuWarpShuffle.h"
#include <iostream>

// Matrix column-sum kernel
__global__
void reduceWS(const float* A, float* out, size_t N)
{
	__shared__ float Tile[32];
	size_t tID = threadIdx.x;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	float val = 0.0;
	unsigned mask = 0xFFFFFFFFU;

	size_t lane = threadIdx.x % warpSize; // lane is the thread ID in a warp
	size_t warpID = threadIdx.x / warpSize;

	const size_t STRIDE = gridDim.x * blockDim.x;

	// Grid-stride loop to load
	while (idx < N)
	{
		val += A[idx];
		idx += STRIDE;
	}

	// 1st Warp-Shuffle Reduction
	for (auto offset = warpSize / 2; offset > 0; offset /= 2)
	{
		val += __shfl_down_sync(mask, val, offset);
	}

	if (lane == 0)
	{
		Tile[warpID] = val;
	}
	__syncthreads();

	// Only warp 0 works from here
	if (warpID == 0)
	{
		val = (tID < blockDim.x / warpSize) ? Tile[lane] : 0;

		for (auto offset = warpSize / 2; offset > 0; offset /= 2)
		{
			val += __shfl_down_sync(mask, val, offset);
		}

		if (tID == 0)
		{
			atomicAdd(out, val);
		}
	}

}

template<typename T>
inline void GpuWarpShuffle<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE_A);
	gpuMalloc(&dSum, SIZE_SUMS);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuWarpShuffle<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE_A, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dSum, this->Sum.data(), SIZE_SUMS, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuWarpShuffle<T>::copyD2H()
{
	gpuMemcpy(this->Sum.data(), this->dSum, SIZE_SUMS, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
void GpuWarpShuffle<T>::launchSetup()
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
GpuWarpShuffle<T>::~GpuWarpShuffle()
{
	gpuFree(dA);
	gpuFree(dSum);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void GpuWarpShuffle<T>::solver()
{
	deviceAllocations();
	copyH2D();
	launchSetup();
	reduceWS << < gridSize, BLOCK_SIZE >> > ((float*)dA, (float*)dSum, N);
	copyD2H();
}

template void GpuWarpShuffle<float>::deviceAllocations();
template void GpuWarpShuffle<double>::deviceAllocations();
template void GpuWarpShuffle<float>::copyH2D();
template void GpuWarpShuffle<double>::copyH2D();
template void GpuWarpShuffle<float>::copyD2H();
template void GpuWarpShuffle<double>::copyD2H();
template void GpuWarpShuffle<float>::solver();
template void GpuWarpShuffle<double>::solver();
template GpuWarpShuffle<float>::~GpuWarpShuffle();
template GpuWarpShuffle<double>::~GpuWarpShuffle();

