#pragma once

#include <vector>
#include <iostream>

#include "../../Parameters.h"

template <typename T>
class Validate
{
private:

public:
	bool validate(const std::vector<T>& v);
};