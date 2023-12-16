#pragma once

#include <memory>
#include <string>

#include "../interface/IStencil.h"
#include "../include/StencilCPU.h"
#include "../include/StencilOmp.h"
#include "../include/StencilGPU.h"

template <typename T>
class StencilFactory
{
private:
	const std::vector<T>& in;
    std::vector<T>& out;

	std::shared_ptr<IStencil<T>> solverSelect;

public:
	StencilFactory(const std::vector<T>& in,
		std::vector<T>& out) : in{ in }, out{ out } {}
	
	std::shared_ptr<IStencil<T>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<StencilCPU<T>>(in, out);
		}
		else if (solverType == "gpu")
		{
			solverSelect = std::make_shared<StencilGPU<T>>(in, out);
		}
		else if (solverType == "omp")
		{
			solverSelect = std::make_shared<StencilOmp<T>>(in, out);
		}
		return solverSelect;
	}

};
