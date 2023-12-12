#pragma once

#include "../interface/IMatrixMult.h"

#include <vector>

template <typename T>
class MatrixMultCPU : public IMatrixMult<T>
{
private:
    
public:
    MatrixMultCPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IMatrixMult<T>(a, b, c) {}
    
    virtual ~MatrixMultCPU() {}

    void matrixMult() override;
};
