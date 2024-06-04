#pragma once

#include "../interface/ISolver.h"
#include <omp.h>
#include <iostream>

template <typename T>
class CpuOmp : public ISolver<T>
{
private:
	const int NTHREADS = omp_get_max_threads();

public:
	CpuOmp(const std::vector<T>& A,
		std::vector<T>& Sum) : ISolver<T>(A, Sum) {}

	virtual ~CpuOmp() {}
	void solver() override;
};