#include "../include/CpuSolver.h"

using std::cout;
using std::endl;
using std::vector;

template<typename T>
T CpuSolver<T>::gpdf(T val)
{
    return exp(-0.5 * val * val) / (SIGMA * 2.5066282747946493232942230134974);
}

template <typename T>
void CpuSolver<T>::solver()
{
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP threads." << endl;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (auto i = 0; i < N; ++i)
        {
            T in = x[i] - (COUNT / 2) * 0.01;
            T out = 0.0;
            for (auto k = 0; k < COUNT; ++k)
            {
                T temp = (in - MEAN) / SIGMA;
                out += gpdf(temp);
                in += 0.01;
            }
            y[i] = out / COUNT;
        }
    }
}

template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();