#pragma once

#include "../interface/ISolver.h"

template <typename T>
class CpuSolver : public ISolver<T>
{
private:

public:
	CpuSolver(const std::vector<T>& A,
		std::vector<T>& Sum) : ISolver<T>(A, Sum) {}

	virtual ~CpuSolver() {}
	void solver() override;
};