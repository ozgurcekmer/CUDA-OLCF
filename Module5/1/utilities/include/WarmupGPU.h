#pragma once

#include "../../Parameters.h"

// Utilities
#include "MaxError.h"

// CUDA libs
#include "GpuCommon.h"

// Standard libs
#include <vector>
#include <iostream>
#include <string>
#include <regex>

/* 
    1<<25  : 34M
    1<<24  : 17M
    1<<23  :  8M
    1<<22  :  4M
    1<<21  :  2M
    1<<20  :  1M
    1<<14  : 16k
*/

class WarmupGPU
{
private:
	const size_t PROBLEM_SIZE = 1 << 25;

public:
	void warmup() const;
	void setup(bool& refGPU, bool& testGPU);
};
