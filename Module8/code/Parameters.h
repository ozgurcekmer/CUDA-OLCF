#pragma once

#include <complex>
#include <cmath>
#include <vector>


/* VECTOR SIZE:
    1<<26  : 68M
    1<<25  : 34M
    1<<24  : 17M
    1<<23  :  8M
    1<<22  :  4M
    1<<21  :  2M
    1<<20  :  1M
    1<<14  : 16k
    1<<13  : 8k
    1<<12  : 4k
*/

// Vector dimension
const size_t N = 1<<12;

const size_t TILE_DIM = 32;
const size_t BLOCK_ROWS = 8;

// Other parameters
typedef float Real;
//typedef double Real;

// Solver selection
static const std::string refSolverName = "gpuOriginal3";
static const std::string testSolverName = "gpuSolver3";

/*
    SOLVERS:
    CPU Solvers:
    - cpu: A CPU solver with OpenMP threads
    
    GPU Solvers:
    - gpuOriginal1: A naive GPU solver
    - gpuOriginal2: A coalesced GPU solver with shared memory
    - gpuOriginal3: A coalesced GPU solver with shared memory - bank conflicts prevented
    - gpuSolver1: A naive GPU solver - fewer threads than data
    - gpuSolver2: A coalesced GPU solver with shared memory - fewer threads than data
    - gpuSolver3: A coalesced GPU solver with shared memory - bank conflicts prevented - fewer threads than data

    WARNING: All GPU solvers need to have the letters "gpu"
    (in this order & lower case) in their names
*/

static bool refGPU = false;
static bool testGPU = false;

#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )