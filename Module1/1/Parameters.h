#pragma once

#include <string>

// CUDA Parameters
static const int BLOCK_SIZE = 3;
static const int GRID_SIZE = 2;

static const std::string solverName = "omp";

/*
    Options:
    - gpu
    - omp
    - serial
*/
