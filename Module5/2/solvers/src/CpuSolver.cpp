#include "../include/CpuSolver.h"

template <typename T>
void CpuSolver<T>::solver()
{
	T temp = 0.0;
	for (const auto &i : A)
	{
		temp = (temp > i) ? temp : i;
	}
	Max[0] = temp;
}

template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();