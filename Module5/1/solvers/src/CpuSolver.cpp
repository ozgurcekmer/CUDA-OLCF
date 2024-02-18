#include "../include/CpuSolver.h"

template <typename T>
void CpuSolver<T>::solver()
{
	T temp = 0.0;
	for (auto i : A)
	{
		temp += i;
	}
	Sum[0] = temp;
}

template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();