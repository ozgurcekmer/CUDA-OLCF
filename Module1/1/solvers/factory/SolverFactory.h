#pragma once

#include <memory>
#include <string>

#include "../interface/IHello.h"
#include "../include/GpuHello.h"
#include "../include/OmpHello.h"
#include "../include/SerialHello.h"

class SolverFactory
{
private:
	std::shared_ptr<IHello> solverSelect;

public:
	
	std::shared_ptr<IHello> getSolver(std::string solverType)
	{
		if (solverType == "gpu")
		{
			solverSelect = std::make_shared<GpuHello>();
		}
		else if (solverType == "omp")
		{
			solverSelect = std::make_shared<OmpHello>();
		}
		else if (solverType == "serial")
		{
			solverSelect = std::make_shared<SerialHello>();
		}

		return solverSelect;
	}

};
