#include "../include/OuterCPU.h"

using std::vector;

template <typename T>
void OuterCPU<T>::matrixMult()
{
    vector<T> Temp(M * N, 0.0);
    for (auto k = 0; k < K; ++k)
    {
        for (auto i = 0; i < M; ++i)
        {
            for (auto j = 0; j < N; ++j)
            {
                Temp[i * N + j] += this->a[i * K + k] * this->b[k * N + j];
            }
        }
    }

    for (auto i = 0; i < M; ++i)
    {
        for (auto j = 0; j < N; ++j)
        {
            this->c[i * N + j] = Temp[i * N + j];
        }
    }


}

template void OuterCPU<float>::matrixMult();
template void OuterCPU<double>::matrixMult();