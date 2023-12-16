#pragma once

#include <string>

typedef float Real;

//static const int N = 4096;
static const int N = 4096*4096;
//static const int BLOCK_SIZE = 16;
static const int BLOCK_SIZE = 256;
static const int GRID_SIZE = N / BLOCK_SIZE;
static const int RADIUS = 3;

// Solver selection
static const std::string refSolverName = "gpu";
static const std::string testSolverName = "gpu";

/*
SOLVERS:
cpu 
gpu
omp
*/
