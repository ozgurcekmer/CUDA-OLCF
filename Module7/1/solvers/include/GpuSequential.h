#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuSequential : public ISolver<T>
{
private:
    T* dX;
    T* dY;

    const size_t BYTES = N * sizeof(T);
    
    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuSequential(Vector::pinnedVector<T>& x, Vector::pinnedVector<T>& y) : ISolver<T>(x, y) {}

    virtual ~GpuSequential();
    void solver() override;
};
