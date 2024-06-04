#pragma once

#include <memory>

#include "../interface/ISolver.h"
#include "../include/CpuSolver.h"
#include "../include/CpuOmp.h"
#include "../include/GpuReduce.h"

template <typename T>
class SolverFactory
{
private:
	const std::vector<T>& A;
	std::vector<T>& Max;
	
	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(const std::vector<T>& A,
		std::vector<T>& Max) : A{ A }, Max{ Max } {}

	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(A, Max);
		}
		else if (solverType == "cpuOmp")
		{
			solverSelect = std::make_shared<CpuOmp<T>>(A, Max);
		}
		else if (solverType == "gpuReduce")
		{
			solverSelect = std::make_shared<GpuReduce<T>>(A, Max);
		}
		return solverSelect;
	}
};