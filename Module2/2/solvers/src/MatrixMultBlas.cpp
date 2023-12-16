#include "../include/MatrixMultBlas.h"

template <typename T>
void MatrixMultBlas<T>::matrixMult()
{
    if (sizeof(T) == sizeof(float))
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, (const float*)a.data(), K, (const float*)b.data(), N, BETA, (float*)c.data(), N);
    }
    else
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, (const double*)a.data(), K, (const double*)b.data(), N, BETA, (double*)c.data(), N);
    }
}

template void MatrixMultBlas<float>::matrixMult();
template void MatrixMultBlas<double>::matrixMult();