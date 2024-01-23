#pragma once

#include "../../Parameters.h"

#include <vector>

template <typename T>
class ISolver
{
protected:
	const std::vector<T>& A;
	std::vector<T>& RowSums;
	std::vector<T>& ColSums;

public:
	ISolver(const std::vector<T>& A,
		std::vector<T>& RowSums,
		std::vector<T>& ColSums) : A{A}, RowSums{RowSums}, ColSums{ ColSums } {}
	virtual ~ISolver() {}
	virtual void rowSums() = 0;
	virtual void colSums() = 0;
};