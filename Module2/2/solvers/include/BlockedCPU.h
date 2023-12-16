#pragma once

#include "../interface/IMatrixMult.h"

#include <vector>

template <typename T>
class BlockedCPU : public IMatrixMult<T>
{
private:

public:
    BlockedCPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IMatrixMult<T>(a, b, c) {}

    virtual ~BlockedCPU() {}

    void matrixMult() override;
};
