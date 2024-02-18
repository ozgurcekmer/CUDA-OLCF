#include "../include/CpuSolver.h"

template <typename T>
void CpuSolver<T>::rowSums()
{
	for (auto i = 0; i < DSIZE; ++i)
	{
		T sum = 0.0;
		for (auto j = 0; j < DSIZE; ++j)
		{
			sum += A[i * DSIZE + j];
		}
		RowSums[i] = sum;
	}
}

template <typename T>
void CpuSolver<T>::colSums()
{
	for (auto j = 0; j < DSIZE; ++j)
	{
		T sum = 0.0;
		for (auto i = 0; i < DSIZE; ++i)
		{
			sum += A[i * DSIZE + j];
		}
		ColSums[j] = sum;
	}
}

template <typename T>
void CpuSolver<T>::solver()
{
	rowSums();
	colSums();
}

template void CpuSolver<float>::rowSums();
template void CpuSolver<double>::rowSums();
template void CpuSolver<float>::colSums();
template void CpuSolver<double>::colSums();
template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();