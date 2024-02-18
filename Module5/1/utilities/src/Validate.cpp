#include "../include/Validate.h"

using std::cout;
using std::endl;

template<typename T>
inline bool Validate<T>::validate(const std::vector<T>& v)
{
	if (v[0] != static_cast<T>(N))
	{
		return false;
	}
	return true;
}


template bool Validate<float>::validate(const std::vector<float>& v);
template bool Validate<double>::validate(const std::vector<double>& v);
