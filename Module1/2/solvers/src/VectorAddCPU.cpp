#include "../include/VectorAddCPU.h"

template <typename T>
void VectorAddCPU<T>::vectorAdd()
{
	for (auto i = 0; i < N; ++i)
	{
		c[i] = a[i] + b[i];
	}
}

template void VectorAddCPU<float>::vectorAdd();
template void VectorAddCPU<double>::vectorAdd();