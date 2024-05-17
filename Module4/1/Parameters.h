#pragma once

/* DSIZE:
    1<<25  : 34M
    1<<24  : 17M
    1<<20  :  1M
    1<<14  : 16k
*/


#include <string>
#include <cmath>

// single/double precision
typedef float Real;
//typedef double Real;

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

// Matrix dimension
const size_t DSIZE = 16384; 
//const size_t DSIZE = 32;

const int BLOCK_SIZE = 256;
const int GRID_SIZE = static_cast<int>(ceil(static_cast<float>(DSIZE) / BLOCK_SIZE));