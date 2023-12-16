#include "../include/StencilCPU.h"

template <typename T>
void StencilCPU<T>::stencil()
{
    for (auto i = RADIUS; i < N + RADIUS; ++i)
    {
        for (auto j = -RADIUS; j <= RADIUS; ++j)
        {
            this->out[i] += this->in[i + j];
        }
    }
}

template void StencilCPU<float>::stencil();
template void StencilCPU<double>::stencil();