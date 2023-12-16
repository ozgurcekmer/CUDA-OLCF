#include "../include/ReorderedCPU.h"

template <typename T>
void ReorderedCPU<T>::matrixMult()
{
    for (auto i = 0; i < M; ++i)
    {
        for (auto k = 0; k < K; ++k)
        {
            for (auto j = 0; j < N; ++j)
            {
                this->c[i * N + j] += this->a[i * K + k] * this->b[k * N + j];
            }
        }
    }
}

template void ReorderedCPU<float>::matrixMult();
template void ReorderedCPU<double>::matrixMult();