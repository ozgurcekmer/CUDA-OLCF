#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuOriginal1 : public ISolver<T>
{
private:
    T* dA;
    T* dB;

    const size_t BYTES = N * N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuOriginal1(std::vector<T>& A, std::vector<T>& B) : ISolver<T>(A, B) {}

    virtual ~GpuOriginal1();
    void solver() override;
};
