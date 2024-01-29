#pragma once

#include "../interface/ISolver.h"

template <typename T>
class CpuSolver : public ISolver<T>
{
private:

public:
	CpuSolver(const std::vector<T>& A,
		std::vector<T>& RowSums,
		std::vector<T>& ColSums) : ISolver<T>(A, RowSums, ColSums) {}

	virtual ~CpuSolver() {}
	void rowSums() override;
	void colSums() override;
	void solver() override;
};