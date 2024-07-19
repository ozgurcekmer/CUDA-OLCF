#include "../include/GpuOverlapEvents.h"

using std::cout;
using std::endl;
using std::vector;

__device__ float gpdfOverlapEvents(float val, float sigma)
{
    return expf(-0.5f * val * val) / (sigma * 2.5066282747946493232942230134974f);
}

__device__ double gpdfOverlapEvents(double val, double sigma)
{
    return expf(-0.5 * val * val) / (sigma * 2.5066282747946493232942230134974);
}

template <typename T>
__global__
void gpuOverlapEvents(T* __restrict__ x, T* __restrict__ y, const T MEAN, const T SIGMA, const size_t N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
    {
        T in = x[idx] - (COUNT / 2) * 0.01;
        T out = 0;
        for (int i = 0; i < COUNT; i++)
        {
            T temp = (in - MEAN) / SIGMA;
            out += gpdfOverlapEvents(temp, SIGMA);
            in += 0.01;
        }
        y[idx] = out / COUNT;
    }
}

template<typename T>
void GpuOverlapEvents<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dX, BYTES);
    gpuMalloc(&dY, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuOverlapEvents<T>::copyH2D(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&dX[offset], &x[offset], CHUNK_BYTES, gpuMemcpyHostToDevice, stream);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuOverlapEvents<T>::copyD2H(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&y[offset], &dY[offset], CHUNK_BYTES, gpuMemcpyDeviceToHost, stream);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuOverlapEvents<T>::~GpuOverlapEvents()
{
    gpuFree(dX);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuOverlapEvents<T>::solver()
{
    deviceAllocations();

    // Streams setup
    gpuStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        gpuStreamCreate(&streams[i]);
    }
    gpuCheckErrors("gpu stream create failure");

    // Events setup
    gpuEvent_t startEvent, stopEvent;
    gpuEventCreate(&startEvent);
    gpuEventCreate(&stopEvent);
    gpuCheckErrors("gpu event create failure");

    // Time here
    gpuEventRecord(startEvent, 0);
    gpuCheckErrors("event record failure");
    for (int i = 0; i < CHUNKS; ++i)
    {
        int offset = i * CHUNK_SIZE;
        copyH2D(offset, streams[i % NUM_STREAMS]);
        gpuOverlapEvents << <GRID_SIZE / CHUNKS, BLOCK_SIZE, 0, streams[i % NUM_STREAMS] >> > (dX + offset, dY + offset, MEAN, SIGMA, CHUNK_SIZE);
        copyD2H(offset, streams[i % NUM_STREAMS]);
        gpuStreamQuery(streams[i % NUM_STREAMS]);
    }
    gpuEventRecord(stopEvent, 0);
    gpuCheckErrors("event record failure");
    gpuEventSynchronize(stopEvent);
    gpuCheckErrors("event sync failure");
    gpuEventElapsedTime(&runtime, startEvent, stopEvent);
    gpuCheckErrors("event elapsed time failure");
    cout << "Version1_Events passed time in ms: " << runtime << endl;

    // Cleanup
    gpuEventDestroy(startEvent);
    gpuEventDestroy(stopEvent);
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        gpuStreamDestroy(streams[i]);
    }
    gpuCheckErrors("stream destroy failure");
}

template void GpuOverlapEvents<float>::solver();
template void GpuOverlapEvents<double>::solver();
template void GpuOverlapEvents<float>::deviceAllocations();
template void GpuOverlapEvents<double>::deviceAllocations();
template GpuOverlapEvents<float>::~GpuOverlapEvents();
template GpuOverlapEvents<double>::~GpuOverlapEvents();
