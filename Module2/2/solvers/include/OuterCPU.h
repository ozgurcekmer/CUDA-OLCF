#pragma once

#include "../interface/IMatrixMult.h"

#include <vector>

template <typename T>
class OuterCPU : public IMatrixMult<T>
{
private:

public:
    OuterCPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IMatrixMult<T>(a, b, c) {}

    virtual ~OuterCPU() {}

    void matrixMult() override;
};
