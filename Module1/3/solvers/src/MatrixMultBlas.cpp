#include "../include/MatrixMultBlas.h"

template <typename T>
void MatrixMultBlas<T>::matrixMult()
{
    if (sizeof(T) == sizeof(float))
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, (const float*)this->a.data(), K, (const float*)this->b.data(), N, BETA, (float*)this->c.data(), N);
    }
    else
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, (const double*)this->a.data(), K, (const double*)this->b.data(), N, BETA, (double*)this->c.data(), N);
    }
}

template void MatrixMultBlas<float>::matrixMult();
template void MatrixMultBlas<double>::matrixMult();