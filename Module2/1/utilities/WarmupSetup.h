#pragma once

#include "../Parameters.h"

static bool refGPU = false;
static bool testGPU = false;

void warmupSetup()
{
    if (refSolverName != "cpu")
    {
        refGPU = true;
    }
    if (testSolverName != "cpu")
    {
        testGPU = true;
    }
}
