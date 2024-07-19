#pragma once

#include <complex>
#include <cmath>
#include <vector>

/* VECTOR SIZE:
    1<<25  : 34M
    1<<24  : 17M
    1<<23  :  8M
    1<<22  :  4M
    1<<21  :  2M
    1<<20  :  1M
    1<<14  : 16k
*/

// Vector dimension
const size_t N = 1 << 26;

// Kernel launch parameters
static const size_t BLOCK_SIZE = 256;
static const size_t GRID_SIZE = static_cast<size_t>(std::ceil(static_cast<float>(N) / BLOCK_SIZE));

// CUDA streams parameters
static const int NUM_STREAMS = 8;
const int CHUNKS = 16;

// Other parameters
typedef float Real;
//typedef double Real;
const int COUNT = 22;

// Solver selection
static const std::string refSolverName = "gpuSequentialEvents";
static const std::string testSolverName = "gpuOverlapEvents";
/*
    SOLVERS:
    CPU Solvers:
    - cpu: a CPU solver using OpenMP threads
  
    GPU Solvers:
    - gpuSequential: Classic data transfer
    - gpuOverlap: Data transfer & kernel overlap with multi-streams

    WARNING: All GPU solvers need to have the letters "gpu"
    (in this order & lower case) in their names
*/

static bool refGPU = false;
static bool testGPU = false;
