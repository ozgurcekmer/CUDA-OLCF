#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

template <typename T>
class GpuReduce : public ISolver<T>
{
private:
	T* dA;
	T* dMax;
	T* dMax0;

	size_t gridSize;

	const size_t SIZE_A = N * sizeof(T);
	const size_t SIZE_RED = sizeof(T);
	const size_t SIZE_RED0 = BLOCK_SIZE * sizeof(T);
	
	void deviceAllocations();
	void copyH2D();
	void copyD2H();
	void launchSetup();

public:
	GpuReduce(const std::vector<T>& A,
		std::vector<T>& Max) : ISolver<T>(A, Max) {}

	virtual ~GpuReduce();
	void solver() override;
};