#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuHeaders.h"

template <typename T>
class GpuSolver : public ISolver<T>
{
private:
	T* dA;
	T* dSum;

	const size_t SIZE_A =  N * sizeof(T);
	const size_t SIZE_SUMS = sizeof(T);

	void deviceAllocations();
	void copyH2D();
	void copyD2H();

public:
	GpuSolver(const std::vector<T>& A,
		std::vector<T>& Sum) : ISolver<T>(A, Sum) {}

	virtual ~GpuSolver();
	void solver() override;
};