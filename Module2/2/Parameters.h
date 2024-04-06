#pragma once

#include <string>

static bool refGPU = false;
static bool testGPU = false;

typedef float Real;
//typedef double Real;
///*
static const int M = 1024;
static const int K = 1024;
static const int N = 1024;
// */

/*
static const int M = 8;
static const int K = 8;
static const int N = 8;
*/

static const size_t blockSize = 32;

static const size_t cacheBlockI = 32;
static const size_t cacheBlockJ = 32;
static const size_t cacheBlockK = 32;

// Solver selection
static const std::string refSolverName = "gpu";
static const std::string testSolverName = "gpuBlas";

/*
	SOLVERS:

	CPU Solvers:
	- cpu
	- omp
	- blas
	- outer
	- blocked
	- reordered

	GPU Solvers:
	- gpu
	- gpuBlas
	- gpuShared

	WARNING: All GPU solvers need to have the letters "gpu"
	(in this order & lower case) in their names
*/