#include "../include/CpuOmp.h"

using std::cout;
using std::endl;

template <typename T>
void CpuOmp<T>::solver()
{
	omp_set_num_threads(NTHREADS);
	cout << "Working with " << NTHREADS << " OpenMP threads." << endl;
	T tempMax = 0.0;

#pragma omp parallel
	{
		T threadMax = 0.0;

#pragma omp for schedule(static)
		for (auto i = 0; i < N; ++i)
		{
			threadMax = (threadMax > A[i]) ? threadMax : A[i];
		}

#pragma omp critical
		tempMax = (threadMax > tempMax) ? threadMax : tempMax;
	}

	Max[0] = tempMax;
}

template void CpuOmp<float>::solver();
template void CpuOmp<double>::solver();