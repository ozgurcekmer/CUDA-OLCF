#include "../include/Validate.h"

using std::cout;
using std::endl;

template<typename T>
inline bool Validate<T>::validate(const std::vector<T>& v)
{
	for (auto i = 0; i < v.size(); ++i)
	{
		if (v[i] != static_cast<T>(N))
		{
			cout << "Results mismatch at "
				<< i << ", was: " << v[i] << " should be: "
				<< static_cast<T>(N) << endl;
			return false;
		}
	}
	return true;
}


template bool Validate<float>::validate(const std::vector<float>& v);
template bool Validate<double>::validate(const std::vector<double>& v);
