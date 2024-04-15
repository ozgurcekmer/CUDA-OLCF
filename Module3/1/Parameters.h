#pragma once

/* N:
    1<<25  : 34M
    1<<24  : 17M
    1<<20  :  1M
    1<<14  : 16k
*/

#include <complex>
#include <cmath>

typedef float Real;

static const int N = 1<<25; //(17 M)

static const int BLOCK_SIZE = 1024;

// Solver selection
static const std::string refSolverName = "cpu";
static const std::string testSolverName = "gpu";

/*
    SOLVERS:
    CPU Solvers:
    - cpu

    GPU Solvers:
    - gpu

    WARNING: All GPU solvers need to have the letters "gpu"
    (in this order & lower case) in their names
*/

static bool refGPU = false;
static bool testGPU = false;
