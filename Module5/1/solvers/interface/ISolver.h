#pragma once

#include "../../Parameters.h"

#include <vector>

template <typename T>
class ISolver
{
protected:
	const std::vector<T>& A;
	std::vector<T>& Sum;

public:
	ISolver(const std::vector<T>& A,
		std::vector<T>& Sum) : A{A}, Sum{Sum} {}
	virtual ~ISolver() {}
	virtual void solver() = 0;
};