#pragma once

#include "../interface/ISolver.h"

template <typename T>
class CpuSolver : public ISolver<T>
{
private:

public:
	CpuSolver(const std::vector<T>& A,
		std::vector<T>& Max) : ISolver<T>(A, Max) {}

	virtual ~CpuSolver() {}
	void solver() override;
};