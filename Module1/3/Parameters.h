#pragma once

#include <string>

typedef double Real;

static const int M = 1024;
static const int K = 512;
static const int N = 256;
 

static const size_t blockSize = 32;

// Solver selection
static const std::string refSolverName = "cpu";
static const std::string testSolverName = "gpu";

/*
SOLVERS:
cpu 
gpu
blas
*/
