#include "../include/GpuSequentialEvents.h"

using std::cout;
using std::endl;
using std::vector;

__device__ float gpdfSeqEvents(float val, float sigma)
{
    return expf(-0.5f * val * val) / (sigma * 2.5066282747946493232942230134974f);
}

__device__ double gpdfSeqEvents(double val, double sigma)
{
    return expf(-0.5 * val * val) / (sigma * 2.5066282747946493232942230134974);
}

template <typename T>
__global__
void gpuSequentialEvents(T* __restrict__ x, T* __restrict__ y, const T MEAN, const T SIGMA, const size_t N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
    {
        T in = x[idx] - (COUNT / 2) * 0.01;
        T out = 0;
        for (int i = 0; i < COUNT; i++)
        {
            T temp = (in - MEAN) / SIGMA;
            out += gpdfSeqEvents(temp, SIGMA);
            in += 0.01;
        }
        y[idx] = out / COUNT;
    }
}

template<typename T>
void GpuSequentialEvents<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dX, BYTES);
    gpuMalloc(&dY, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSequentialEvents<T>::copyH2D()
{
    gpuMemcpy(dX, this->x.data(), BYTES, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSequentialEvents<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, BYTES, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSequentialEvents<T>::~GpuSequentialEvents()
{
    gpuFree(dX);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSequentialEvents<T>::solver()
{
    deviceAllocations();
    
    // Events setup
    gpuEvent_t startEvent, stopEvent;
    gpuEventCreate(&startEvent);
    gpuEventCreate(&stopEvent);
    gpuCheckErrors("gpu event create failure");
    
    // Time here
    gpuEventRecord(startEvent, 0);
    gpuCheckErrors("event record failure");
    copyH2D();
    gpuSequentialEvents<T> << < GRID_SIZE, BLOCK_SIZE >> > (dX, dY, MEAN, SIGMA, N);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
    
    gpuEventRecord(stopEvent, 0);
    gpuCheckErrors("event record failure");
    gpuEventSynchronize(stopEvent);
    gpuCheckErrors("event sync failure");
    gpuEventElapsedTime(&runtime, startEvent, stopEvent);
    gpuCheckErrors("event elapsed time failure");
    cout << "Sequential_Events passed time in ms: " << runtime << endl;
}

template void GpuSequentialEvents<float>::solver();
template void GpuSequentialEvents<double>::solver();
template void GpuSequentialEvents<float>::deviceAllocations();
template void GpuSequentialEvents<double>::deviceAllocations();
template void GpuSequentialEvents<float>::copyH2D();
template void GpuSequentialEvents<double>::copyH2D();
template GpuSequentialEvents<float>::~GpuSequentialEvents();
template GpuSequentialEvents<double>::~GpuSequentialEvents();
