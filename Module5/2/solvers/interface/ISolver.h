#pragma once

#include "../../Parameters.h"

#include <vector>

template <typename T>
class ISolver
{
protected:
	const std::vector<T>& A;
	std::vector<T>& Max;

public:
	ISolver(const std::vector<T>& A,
		std::vector<T>& Max) : A{A}, Max{Max} {}
	virtual ~ISolver() {}
	virtual void solver() = 0;
};