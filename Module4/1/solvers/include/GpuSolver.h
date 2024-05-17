#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

template <typename T>
class GpuSolver : public ISolver<T>
{
private:
	T* dA;
	T* dRowSums;
	T* dColSums;

	const size_t SIZE_A = DSIZE * DSIZE * sizeof(T);
	const size_t SIZE_SUMS = DSIZE * sizeof(T);

	void deviceAllocations();
	void copyH2D();
	void copyD2H();

public:
	GpuSolver(const std::vector<T>& A,
		std::vector<T>& RowSums,
		std::vector<T>& ColSums) : ISolver<T>(A, RowSums, ColSums) {}

	virtual ~GpuSolver();
	void rowSums() override;
	void colSums() override;
	void solver() override;
};