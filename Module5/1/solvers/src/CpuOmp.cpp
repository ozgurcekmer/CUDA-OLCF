#include "../include/CpuOmp.h"

using std::cout;
using std::endl;

template <typename T>
void CpuOmp<T>::solver()
{
	omp_set_num_threads(NTHREADS);
	cout << "Working with " << NTHREADS << " OpenMP threads." << endl;
	T tempSum = 0.0;

#pragma omp parallel
	{
		T threadSum = 0.0;

#pragma omp for schedule(static)
		for (auto i = 0; i < N; ++i)
		{
			threadSum += A[i];
		}

#pragma omp critical
		tempSum += threadSum;
	}

	Sum[0] = tempSum;
}

template void CpuOmp<float>::solver();
template void CpuOmp<double>::solver();