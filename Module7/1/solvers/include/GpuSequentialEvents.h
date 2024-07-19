#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuSequentialEvents : public ISolver<T>
{
private:
    T* dX;
    T* dY;

    const size_t BYTES = N * sizeof(T);
    float runtime = 0.0;

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuSequentialEvents(Vector::pinnedVector<T>& x, Vector::pinnedVector<T>& y) : ISolver<T>(x, y) {}

    virtual ~GpuSequentialEvents();
    void solver() override;
};
