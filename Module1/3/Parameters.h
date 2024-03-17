#pragma once

#include <string>

static bool refGPU = false;
static bool testGPU = false;

typedef double Real;

static const int M = 1024;
static const int K = 512;
static const int N = 256;
 

static const size_t blockSize = 32;

// Solver selection
static const std::string refSolverName = "blas";
static const std::string testSolverName = "gpu";

/*
	SOLVERS:
	
	CPU Solvers:
	- cpu 
	- blas

	GPU Solvers:
	- gpu

	WARNING: All GPU solvers need to have the letters "gpu"
	(in this order & lower case) in their names
*/
