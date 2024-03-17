#pragma once

#include "../interface/IMatrixMult.h"

//#include <cblas.h> // In Setonix
#include <mkl_cblas.h>
#include <vector>

template <typename T>
class MatrixMultBlas : public IMatrixMult<T>
{
private:
    const T ALPHA = 1.0;
    const T BETA = 0.0;

public:
    MatrixMultBlas(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IMatrixMult<T>(a, b, c) {}

    virtual ~MatrixMultBlas() {}

    void matrixMult() override;
};
