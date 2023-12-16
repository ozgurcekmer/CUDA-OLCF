#pragma once

#include "../interface/IStencil.h"
#include <omp.h>
#include <vector>
#include <iostream>

template <typename T>
class StencilOmp : public IStencil<T>
{
private:

public:
    StencilOmp(const std::vector<T>& in,
        std::vector<T>& out) : IStencil<T>(in, out) {}

    virtual ~StencilOmp() {}

    void stencil() override;
};
