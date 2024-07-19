#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuOverlap : public ISolver<T>
{
private:
    T* dX;
    T* dY;

    const size_t BYTES = N * sizeof(T);
    const size_t STREAM_BYTES = BYTES / NUM_STREAMS;
    const size_t CHUNK_BYTES = BYTES / CHUNKS;
    const size_t STREAM_SIZE = N / NUM_STREAMS;
    const size_t CHUNK_SIZE = N / CHUNKS;

    void deviceAllocations();
    void copyH2D(size_t offset, gpuStream_t stream);
    void copyD2H(size_t offset, gpuStream_t stream);

public:
    GpuOverlap(Vector::pinnedVector<T>& x, Vector::pinnedVector<T>& y) : ISolver<T>(x, y) {}
    
    virtual ~GpuOverlap();
    void solver() override;
};
