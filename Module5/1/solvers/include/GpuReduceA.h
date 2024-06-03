#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

template <typename T>
class GpuReduceA : public ISolver<T>
{
private:
	T* dA;
	T* dSum;

	size_t gridSize;

	const size_t SIZE_A = N * sizeof(T);
	const size_t SIZE_SUMS = sizeof(T);

	void deviceAllocations();
	void copyH2D();
	void copyD2H();
	void launchSetup();

public:
	GpuReduceA(const std::vector<T>& A,
		std::vector<T>& Sum) : ISolver<T>(A, Sum) {}

	virtual ~GpuReduceA();
	void solver() override;
};