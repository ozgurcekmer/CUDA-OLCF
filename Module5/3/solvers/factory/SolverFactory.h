#pragma once

#include <memory>

#include "../interface/ISolver.h"
#include "../include/CpuSolver.h"
#include "../include/GpuSolver.h"
#include "../include/GpuReduction.h"

template <typename T>
class SolverFactory
{
private:
	const std::vector<T>& A;
	std::vector<T>& RowSums;
	std::vector<T>& ColSums;

	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(const std::vector<T>& A,
		std::vector<T>& RowSums,
		std::vector<T>& ColSums) : A{ A }, RowSums{ RowSums }, ColSums{ ColSums } {}

	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(A, RowSums, ColSums);
		}
		else if (solverType == "gpu")
		{
			solverSelect = std::make_shared<GpuSolver<T>>(A, RowSums, ColSums);
		}
		else if (solverType == "gpuRed")
		{
			solverSelect = std::make_shared<GpuReduction<T>>(A, RowSums, ColSums);
		}
		return solverSelect;
	}

};