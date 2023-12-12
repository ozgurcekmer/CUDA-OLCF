#pragma once

#include <memory>
#include <string>

#include "../interface/IMatrixMult.h"
#include "../include/MatrixMultCPU.h"
#include "../include/MatrixMultGPU.h"
// #include "../include/MatrixMultBlas.h"

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
/*		else if (solverType == "blas")
		{
			solverSelect = std::make_shared<MatrixMultBlas<T>>(a, b, c);
		}
*/		return solverSelect;
	}

};
