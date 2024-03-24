#pragma once

#include <string>

static bool refGPU = false;
static bool testGPU = false;

typedef double Real;

//static const int N = 4096;
static const int N = 4096 * 4096;
//static const int BLOCK_SIZE = 16;
static const int BLOCK_SIZE = 256;
static const int GRID_SIZE = N / BLOCK_SIZE;
static const int RADIUS = 3;

// Solver selection
static const std::string refSolverName = "omp";
static const std::string testSolverName = "gpu";

/*
	SOLVERS:

	CPU Solvers:
	- cpu
	- omp

	GPU Solvers:
	- gpu

	WARNING: All GPU solvers need to have the letters "gpu"
	(in this order & lower case) in their names
*/
