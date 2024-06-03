#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

template <typename T>
class GpuReduce : public ISolver<T>
{
private:
	T* dA;
	T* dSum;
	T* dSum0;

	size_t gridSize;

	const size_t SIZE_A = N * sizeof(T);
	const size_t SIZE_SUMS = sizeof(T);
	const size_t SIZE_SUMS0 = BLOCK_SIZE * sizeof(T);
	
	void deviceAllocations();
	void copyH2D();
	void copyD2H();
	void launchSetup();

public:
	GpuReduce(const std::vector<T>& A,
		std::vector<T>& Sum) : ISolver<T>(A, Sum) {}

	virtual ~GpuReduce();
	void solver() override;
};