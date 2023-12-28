#pragma once

/* N:
    1<<24  : 17M
    1<<20  :  1M
    1<<14  : 16k
*/

#include <complex>
#include <cmath>

typedef float Real;

static const int N = 1<<24; //(17 M)

static const size_t BLOCK_SIZE = 1024;
static const size_t GRID_SIZE = static_cast<size_t>(std::ceil(static_cast<float>(N) / BLOCK_SIZE));

// Solver selection
static const std::string refSolverName = "cpu";
static const std::string testSolverName = "gpu";

/*
SOLVERS:
cpu 
gpu
*/
