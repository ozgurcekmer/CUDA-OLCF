#include "../include/GpuOverlap.h"

using std::cout;
using std::endl;
using std::vector;

__device__ float gpdfOverlap(float val, float sigma)
{
    return expf(-0.5f * val * val) / (sigma * 2.5066282747946493232942230134974f);
}

__device__ double gpdfOverlap(double val, double sigma)
{
    return expf(-0.5 * val * val) / (sigma * 2.5066282747946493232942230134974);
}

template <typename T>
__global__
void gpuOverlap(T* __restrict__ x, T* __restrict__ y, const T MEAN, const T SIGMA, const size_t N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
    {
        T in = x[idx] - (COUNT / 2) * 0.01;
        T out = 0;
        for (int i = 0; i < COUNT; i++) 
        {
            T temp = (in - MEAN) / SIGMA;
            out += gpdfOverlap(temp, SIGMA);
            in += 0.01;
        }
        y[idx] = out / COUNT;
    }
}

template<typename T>
void GpuOverlap<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dX, BYTES);
    gpuMalloc(&dY, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuOverlap<T>::copyH2D(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&dX[offset], &x[offset], CHUNK_BYTES, gpuMemcpyHostToDevice, stream);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuOverlap<T>::copyD2H(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&y[offset], &dY[offset], CHUNK_BYTES, gpuMemcpyDeviceToHost, stream);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuOverlap<T>::~GpuOverlap()
{
    gpuFree(dX);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuOverlap<T>::solver()
{
    deviceAllocations();

    gpuStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        gpuStreamCreate(&streams[i]);
    }
    gpuCheckErrors("gpu stream create failure");

    for (int i = 0; i < CHUNKS; ++i)
    {
        int offset = i * CHUNK_SIZE;
        copyH2D(offset, streams[i % NUM_STREAMS]);
        gpuOverlap << <GRID_SIZE / CHUNKS, BLOCK_SIZE, 0, streams[i % NUM_STREAMS] >> > (dX + offset, dY + offset, MEAN, SIGMA, CHUNK_SIZE);
        copyD2H(offset, streams[i % NUM_STREAMS]);
        gpuStreamQuery(streams[i % NUM_STREAMS]);
    }
    gpuDeviceSynchronize();
    gpuCheckErrors("device sync failure");

    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        gpuStreamDestroy(streams[i]);
    }
    gpuCheckErrors("stream destroy failure");
}

template void GpuOverlap<float>::solver();
template void GpuOverlap<double>::solver();
template void GpuOverlap<float>::deviceAllocations();
template void GpuOverlap<double>::deviceAllocations();
template GpuOverlap<float>::~GpuOverlap();
template GpuOverlap<double>::~GpuOverlap();
