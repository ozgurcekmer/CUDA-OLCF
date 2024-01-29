#pragma once

#include <string>

// single/double precision
typedef float Real;
//typedef double Real;

// Solver selection
static const std::string refSolverName = "cpu";
static const std::string testSolverName = "gpu";
/*
	Solver names:
	- cpu: classical cpu solver
	- gpu: classical gpu solver
*/


// Matrix dimension
const size_t DSIZE = 16384; 
//const size_t DSIZE = 32;

const int BLOCK_SIZE = 256;
const int GRID_SIZE = static_cast<int>(ceil(static_cast<float>(DSIZE) / BLOCK_SIZE));