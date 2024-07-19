#pragma once

#include <memory>
#include <string>

#include "../interface/ISolver.h"
#include "../include/GpuSequential.h"
#include "../include/GpuSequentialEvents.h"
#include "../include/GpuOverlap.h"
#include "../include/GpuOverlapEvents.h"
#include "../include/CpuSolver.h"
#include "../../utilities/include/vectors/PinnedVector.h"

template <typename T>
class SolverFactory
{
private:
	Vector::pinnedVector<T>& x;
	Vector::pinnedVector<T>& y;

	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(Vector::pinnedVector<T>& x, Vector::pinnedVector<T>& y) : x{ x }, y{ y } {}
	
	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "gpuSequential")
		{
			solverSelect = std::make_shared<GpuSequential<T>>(x, y);
		}
		else if (solverType == "gpuSequentialEvents")
		{
			solverSelect = std::make_shared<GpuSequentialEvents<T>>(x, y);
		}
		else if (solverType == "gpuOverlap")
		{
			solverSelect = std::make_shared<GpuOverlap<T>>(x, y);
		}
		else if (solverType == "gpuOverlapEvents")
		{
			solverSelect = std::make_shared<GpuOverlapEvents<T>>(x, y);
		}
		else if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(x, y);
		}
		
		return solverSelect;
	}

};
