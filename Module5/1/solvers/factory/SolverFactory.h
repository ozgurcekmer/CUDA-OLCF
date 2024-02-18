#pragma once

#include <memory>

#include "../interface/ISolver.h"
#include "../include/CpuSolver.h"
#include "../include/GpuSolver.h"

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
		else if (solverType == "gpu")
		{
			solverSelect = std::make_shared<GpuSolver<T>>(A, Sum);
		}

		return solverSelect;
	}

};