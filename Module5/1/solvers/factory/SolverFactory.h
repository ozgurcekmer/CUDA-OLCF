#pragma once

#include <memory>

#include "../interface/ISolver.h"
#include "../include/CpuSolver.h"
#include "../include/CpuOmp.h"
#include "../include/GpuNaive.h"
#include "../include/GpuReduce.h"
#include "../include/GpuReduceA.h"
#include "../include/GpuWarpShuffle.h"

template <typename T>
class SolverFactory
{
private:
	const std::vector<T>& A;
	std::vector<T>& Sum;
	
	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(const std::vector<T>& A,
		std::vector<T>& Sum) : A{ A }, Sum{ Sum } {}

	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(A, Sum);
		}
		else if (solverType == "cpuOmp")
		{
			solverSelect = std::make_shared<CpuOmp<T>>(A, Sum);
		}
		else if (solverType == "gpuNaive")
		{
			solverSelect = std::make_shared<GpuNaive<T>>(A, Sum);
		}
		else if (solverType == "gpuReduce")
		{
			solverSelect = std::make_shared<GpuReduce<T>>(A, Sum);
		}
		else if (solverType == "gpuReduceA")
		{
			solverSelect = std::make_shared<GpuReduceA<T>>(A, Sum);
		}
		else if (solverType == "gpuWarpShuffle")
		{
			solverSelect = std::make_shared<GpuWarpShuffle<T>>(A, Sum);
		}
		return solverSelect;
	}
};