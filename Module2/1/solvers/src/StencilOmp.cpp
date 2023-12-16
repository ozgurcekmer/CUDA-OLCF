#include "../include/StencilOmp.h"

using std::cout;
using std::endl;

template <typename T>
void StencilOmp<T>::stencil()
{
    int nThreads, tID;
    nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "OpenMP with " << nThreads << " threads." << endl;
    #pragma omp parallel for
    for (auto i = RADIUS; i < N + RADIUS; ++i)
    {
        for (auto j = -RADIUS; j <= RADIUS; ++j)
        {
            this->out[i] += this->in[i + j];
        }
    }
}

template void StencilOmp<float>::stencil();
template void StencilOmp<double>::stencil();