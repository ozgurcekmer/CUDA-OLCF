#include "../include/GpuSequential.h"

using std::cout;
using std::endl;
using std::vector;

__device__ float gpdfSeq(float val, float sigma)
{
    return expf(-0.5f * val * val) / (sigma * 2.5066282747946493232942230134974f);
}

__device__ double gpdfSeq(double val, double sigma)
{
    return expf(-0.5 * val * val) / (sigma * 2.5066282747946493232942230134974);
}

template <typename T>
__global__
void gpuSequential(T* __restrict__ x, T* __restrict__ y, const T MEAN, const T SIGMA, const size_t N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
    {
        T in = x[idx] - (COUNT / 2) * 0.01;
        T out = 0;
        for (int i = 0; i < COUNT; i++) 
        {
            T temp = (in - MEAN) / SIGMA;
            out += gpdfSeq(temp, SIGMA);
            in += 0.01;
        }
        y[idx] = out / COUNT;
    }
}

template<typename T>
void GpuSequential<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dX, BYTES);
    gpuMalloc(&dY, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSequential<T>::copyH2D()
{
    gpuMemcpy(dX, this->x.data(), BYTES, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSequential<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, BYTES, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSequential<T>::~GpuSequential()
{
    gpuFree(dX);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSequential<T>::solver()
{
    deviceAllocations();
    copyH2D();
    gpuSequential<T> << < GRID_SIZE, BLOCK_SIZE >> > (dX, dY, MEAN, SIGMA, N);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
}

template void GpuSequential<float>::solver();
template void GpuSequential<double>::solver();
template void GpuSequential<float>::deviceAllocations();
template void GpuSequential<double>::deviceAllocations();
template void GpuSequential<float>::copyH2D();
template void GpuSequential<double>::copyH2D();
template GpuSequential<float>::~GpuSequential();
template GpuSequential<double>::~GpuSequential();
