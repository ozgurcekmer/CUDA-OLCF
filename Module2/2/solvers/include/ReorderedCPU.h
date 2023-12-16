#pragma once

#include "../interface/IMatrixMult.h"

#include <vector>

template <typename T>
class ReorderedCPU : public IMatrixMult<T>
{
private:

public:
    ReorderedCPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IMatrixMult<T>(a, b, c) {}

    virtual ~ReorderedCPU() {}

    void matrixMult() override;
};
