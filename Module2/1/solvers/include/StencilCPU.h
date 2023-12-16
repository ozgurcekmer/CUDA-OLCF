#pragma once

#include "../interface/IStencil.h"

#include <vector>

template <typename T>
class StencilCPU : public IStencil<T>
{
private:
    
public:
    StencilCPU(const std::vector<T>& in,
        std::vector<T>& out) : IStencil<T>(in, out) {}
    
    virtual ~StencilCPU() {}

    void stencil() override;
};
