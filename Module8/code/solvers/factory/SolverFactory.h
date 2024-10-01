#pragma once

#include <memory>
#include <string>

#include "../interface/ISolver.h"
#include "../include/GpuSolver1.h"
#include "../include/GpuSolver2.h"
#include "../include/GpuSolver3.h"
#include "../include/GpuOriginal1.h"
#include "../include/GpuOriginal2.h"
#include "../include/GpuOriginal3.h"
#include "../include/CpuSolver.h"
#include "../../utilities/include/vectors/PinnedVector.h"

template <typename T>
class SolverFactory
{
private:
	std::vector<T>& A;
	std::vector<T>& B;
	
	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(std::vector<T>& A, std::vector<T>& B) : A{ A }, B{ B } {}
	
	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "gpuSolver1")
		{
			solverSelect = std::make_shared<GpuSolver1<T>>(A, B);
		}
		else if (solverType == "gpuSolver2")
		{
			solverSelect = std::make_shared<GpuSolver2<T>>(A, B);
		}
		else if (solverType == "gpuSolver3")
		{
			solverSelect = std::make_shared<GpuSolver3<T>>(A, B);
		}
		else if (solverType == "gpuOriginal1")
		{
			solverSelect = std::make_shared<GpuOriginal1<T>>(A, B);
		}
		else if (solverType == "gpuOriginal2")
		{
			solverSelect = std::make_shared<GpuOriginal2<T>>(A, B);
		}
		else if (solverType == "gpuOriginal3")
		{
			solverSelect = std::make_shared<GpuOriginal3<T>>(A, B);
		}
		else if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(A, B);
		}
		return solverSelect;
	}

};
