#pragma once

#include "../interface/IStencil.h"
#include "../../utilities/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class StencilGPU : public IStencil<T>
{
private:
    T* dIn;
    T* dOut;

    const size_t SIZE = (N + 2 * RADIUS) * sizeof(T);
    
    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    StencilGPU(const std::vector<T>& in,
        std::vector<T>& out) : IStencil<T>(in, out) {}

    virtual ~StencilGPU();

    void stencil() override;
};
