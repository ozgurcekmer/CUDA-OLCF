#pragma once

#include <memory>
#include <string>

#include "../interface/IMatrixMult.h"
#include "../include/MatrixMultCPU.h"
#include "../include/BlasGPU.h"
#include "../include/ReorderedCPU.h"
#include "../include/OuterCPU.h"
#include "../include/BlockedCPU.h"
#include "../include/OmpCPU.h"
#include "../include/MatrixMultBlas.h"
#include "../include/MatrixMultGPU.h"
#include "../include/MatrixMultShared.h"

template <typename T>
class MatrixMultFactory
{
private:
	const std::vector<T>& a;
	const std::vector<T>& b;
	std::vector<T>& c;

	std::shared_ptr<IMatrixMult<T>> solverSelect;

public:
	MatrixMultFactory(const std::vector<T>& a,
		const std::vector<T>& b,
		std::vector<T>& c) : a{ a }, b{ b }, c{ c } {}
	
	std::shared_ptr<IMatrixMult<T>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<MatrixMultCPU<T>>(a, b, c);
		}
		else if (solverType == "gpu")
		{
			solverSelect = std::make_shared<MatrixMultGPU<T>>(a, b, c);
		}
		else if (solverType == "reordered")
		{
			solverSelect = std::make_shared<ReorderedCPU<T>>(a, b, c);
		}
		else if (solverType == "outer")
		{
			solverSelect = std::make_shared<OuterCPU<T>>(a, b, c);
		}
		else if (solverType == "omp")
		{
			solverSelect = std::make_shared<OmpCPU<T>>(a, b, c);
		}
		else if (solverType == "blocked")
		{
			solverSelect = std::make_shared<BlockedCPU<T>>(a, b, c);
		}
		else if (solverType == "shared")
		{
			solverSelect = std::make_shared<MatrixMultShared<T>>(a, b, c);
		}
		else if (solverType == "blas")
		{
			solverSelect = std::make_shared<MatrixMultBlas<T>>(a, b, c);
		}
		else if (solverType == "gpuBlas")
		{
			solverSelect = std::make_shared<BlasGPU<T>>(a, b, c);
		}
		return solverSelect;
	}

};
