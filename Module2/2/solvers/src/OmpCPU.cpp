#include "../include/OmpCPU.h"

using std::cout;
using std::endl;

template <typename T>
void OmpCPU<T>::matrixMult()
{
    int nThreads, tID;
    nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "OpenMP with " << nThreads << " threads." << endl;
    #pragma omp parallel for
    for (auto ii = 0; ii < M; ii += cacheBlockI)
    {
        for (auto jj = 0; jj < N; jj += cacheBlockJ)
        {
            for (auto kk = 0; kk < K; kk += cacheBlockK)
            {
                for (auto i = ii; i < ii + cacheBlockI; ++i)
                {
                    for (auto j = jj; j < jj + cacheBlockJ; ++j)
                    {
                        T tmp = 0.0;
                        for (auto k = kk; k < kk + cacheBlockK; ++k)
                        {
                            tmp += this->a[i * K + k] * this->b[k * N + j];
                        }
                        this->c[i * N + j] += tmp;
                    }
                }
            }
        }
    }
}

template void OmpCPU<float>::matrixMult();
template void OmpCPU<double>::matrixMult();