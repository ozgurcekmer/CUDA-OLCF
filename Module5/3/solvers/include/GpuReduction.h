#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuHeaders.h"

template <typename T>
class GpuReduction : public ISolver<T>
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
	GpuReduction(const std::vector<T>& A,
		std::vector<T>& RowSums,
		std::vector<T>& ColSums) : ISolver<T>(A, RowSums, ColSums) {}

	virtual ~GpuReduction();
	void rowSums() override;
	void colSums() override;
	void solver() override;
};