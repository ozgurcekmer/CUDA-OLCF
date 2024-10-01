#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuSolver3 : public ISolver<T>
{
private:
    T* dA;
    T* dB;

    const size_t BYTES = N * N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuSolver3(std::vector<T>& A, std::vector<T>& B) : ISolver<T>(A, B) {}

    virtual ~GpuSolver3();
    void solver() override;
};
