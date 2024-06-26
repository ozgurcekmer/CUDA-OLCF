#pragma once

#include "../interface/IMatrixMult.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class BlasGPU : public IMatrixMult<T>
{
private:
    T* dA;
    T* dB;
    T* dC;

    const size_t SIZE_A = M * K * sizeof(T);
    const size_t SIZE_B = K * N * sizeof(T);
    const size_t SIZE_C = M * N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    BlasGPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IMatrixMult<T>(a, b, c) {}

    virtual ~BlasGPU();

    void matrixMult() override;
};
