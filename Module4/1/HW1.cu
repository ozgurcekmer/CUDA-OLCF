#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

// Error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const size_t DSIZE = 16384; // Matrix side dimension
const int BLOCK_SIZE = 256; // CUDA maximum is 1024

// Matrix row-sum kernel
__global__
void row_sums(const float* A, float* sums, size_t ds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ds)
    {
        float sum = 0.0;
        for (size_t i = 0; i < ds; ++i)
        {
            sum += A[idx * ds + i];
        }
        sums[idx] = sum;
    }
}

// Matrix column-sum kernel
__global__
void column_sums(const float* A, float* sums, size_t ds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ds)
    {
        float sum = 0.0;
        for (size_t i = 0; i < ds; ++i)
        {
            sum += A[i * ds + idx];
        }
        sums[idx] = sum;
    }
}

bool validate(vector<float> data, size_t sz)
{
    for (size_t i = 0; i < sz; ++i)
    {
        if (data[i] != static_cast<float>(sz))
        {
            cout << "Results mismatch at " << i << ", was: " << data[i] << " should be: " << static_cast<float>(sz) << endl;
            return false;
        }
        return true;
    }
}

int main()
{
    vector<float> hA(DSIZE * DSIZE, 1.0);
    vector<float> hSums(DSIZE, 0.0);

    const size_t SIZE_VECTOR = DSIZE * DSIZE * sizeof(float);
    const size_t SIZE_SUMS = DSIZE * sizeof(float);

    const int GRID_SIZE = static_cast<int>(ceil(static_cast<float>(DSIZE) / BLOCK_SIZE));
    
    float* dA;
    float* dSums;

    cudaMalloc(&dA, SIZE_VECTOR);
    cudaMalloc(&dSums, SIZE_SUMS);
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(dA, hA.data(), SIZE_VECTOR, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    row_sums << <GRID_SIZE, BLOCK_SIZE >> > (dA, dSums, DSIZE);
    cudaCheckErrors("Kernel launch failure");

    cudaMemcpy(hSums.data(), dSums, SIZE_SUMS, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    
    if (!validate(hSums, DSIZE))
    {
        return -1;
    }
    cout << "Row sums correct!" << endl;

    cudaMemset(dSums, 0.0, SIZE_SUMS);
    column_sums << <GRID_SIZE, BLOCK_SIZE >> > (dA, dSums, DSIZE);
    cudaCheckErrors("Kernel launch failure");

    cudaMemcpy(hSums.data(), dSums, SIZE_SUMS, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    if (!validate(hSums, DSIZE))
    {
        return -1;
    }
    cout << "Column sums correct!" << endl;





    cudaFree(dA);
    cudaFree(dSums);
    cudaCheckErrors("cudaFree failure");
}




