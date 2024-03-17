#pragma once

#include "../interface/IVectorAdd.h"

#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class VectorAddGPU : public IVectorAdd<T>
{
private:
    T* dA;
    T* dB;
    T* dC;

    const size_t SIZE = N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    VectorAddGPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IVectorAdd<T>(a, b, c) {}

    virtual ~VectorAddGPU();

    void vectorAdd() override;
};
