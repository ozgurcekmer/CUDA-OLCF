#include "../include/WarmupGPU.h"

using std::vector;
using std::cout;
using std::endl;

__global__
void reduceWarmup(const float* A, float* out, size_t N)
{
	__shared__ float TILE[BLOCK_SIZE];
	size_t tID = threadIdx.x;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	TILE[tID] = 0.0;

	const size_t STRIDE = blockDim.x * gridDim.x;

	while (idx < N)
	{
		TILE[tID] += A[idx];
		idx += STRIDE;
	}

	for (auto s = blockDim.x / 2; s > 0; s /= 2)
	{
		__syncthreads();
		if (tID < s)
		{
			TILE[tID] += TILE[tID + s];
		}
	}

	if (tID == 0)
	{
		atomicAdd(out, TILE[0]);
	}
}

void WarmupGPU::warmup() const
{
	vector<float> A(PROBLEM_SIZE, 1.0);
	vector<float> Sum(1, 0.0);
	vector<float> SumResult(1, static_cast<float>(PROBLEM_SIZE));
	
	const size_t SIZE_A = PROBLEM_SIZE * sizeof(float);
	const size_t SIZE_SUM = sizeof(float);

	float* dA;
	float* dSum;
	
	gpuMalloc(&dA, SIZE_A);
	gpuMalloc(&dSum, SIZE_SUM);
	
	gpuMemcpy(dA, A.data(), SIZE_A, gpuMemcpyHostToDevice);
	gpuMemcpy(dSum, Sum.data(), SIZE_SUM, gpuMemcpyHostToDevice);

	const int GRID_SIZE = static_cast<int>(ceil(static_cast<float>(PROBLEM_SIZE) / BLOCK_SIZE));
	
	reduceWarmup << <GRID_SIZE, BLOCK_SIZE >> > (dA, dSum, PROBLEM_SIZE);

	gpuMemcpy(Sum.data(), dSum, SIZE_SUM, gpuMemcpyDeviceToHost);

	MaxError<float> maxError;
	cout << "Verifying Warmup code" << endl;
	maxError.maxError(Sum, SumResult);

	gpuFree(dA);
	gpuFree(dSum);
}

void WarmupGPU::setup(bool& refGPU, bool& testGPU)
{
	std::string patternGpu("gpu");

	// GPU solver names should have the letters "gpu"
	patternGpu = "[[:alpha:]]*" + patternGpu + "[[:alpha:]]*";
	std::regex rGpu(patternGpu);
	std::smatch resultsGpu;

	refGPU = std::regex_search(refSolverName, resultsGpu, rGpu);
	testGPU = std::regex_search(testSolverName, resultsGpu, rGpu);
}
