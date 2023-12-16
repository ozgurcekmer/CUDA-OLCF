#pragma once

#include "../interface/IMatrixMult.h"
#include <omp.h>
#include <iostream>

#include <vector>

template <typename T>
class OmpCPU : public IMatrixMult<T>
{
private:

public:
    OmpCPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IMatrixMult<T>(a, b, c) {}

    virtual ~OmpCPU() {}

    void matrixMult() override;
};
