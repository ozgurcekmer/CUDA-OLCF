#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

template <typename T>
class GpuNaive : public ISolver<T>
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
	GpuNaive(const std::vector<T>& A,
		std::vector<T>& Sum) : ISolver<T>(A, Sum) {}

	virtual ~GpuNaive();
	void solver() override;
};