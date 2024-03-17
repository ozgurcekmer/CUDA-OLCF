#include "../include/MatrixMultCPU.h"

template <typename T>
void MatrixMultCPU<T>::matrixMult()
{
    for (auto i = 0; i < M; ++i)
    {
        for (auto j = 0; j < N; ++j)
        {
            for (auto k = 0; k < K; ++k)
            {
                c[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
}

template void MatrixMultCPU<float>::matrixMult();
template void MatrixMultCPU<double>::matrixMult();