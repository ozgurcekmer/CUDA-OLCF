#pragma once

#include "../Parameters.h"

static bool refGPU = false;
static bool testGPU = false;

void warmupSetup()
{
    if (refSolverName == "gpu" 
        || refSolverName == "shared"
        || refSolverName == "gpuBlas")
    {
        refGPU = true;
    }
    if (testSolverName == "gpu" 
        || testSolverName == "shared"
        || testSolverName == "gpuBlas")
    {
        testGPU = true;
    }
}
