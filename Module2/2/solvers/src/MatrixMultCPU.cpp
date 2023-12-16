#include "../include/MatrixMultCPU.h"

template <typename T>
void MatrixMultCPU<T>::matrixMult()
{
    for (auto i = 0; i < M; ++i)
    {
        for (auto j = 0; j < N; ++j)
        {
            T tmp = 0.0;
            for (auto k = 0; k < K; ++k)
            {
                tmp += this->a[i * K + k] * this->b[k * N + j];
            }
            this->c[i * N + j] = tmp;
        }
    }
}

template void MatrixMultCPU<float>::matrixMult();
template void MatrixMultCPU<double>::matrixMult();