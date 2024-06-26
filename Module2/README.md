# HW 1
- A stencil operation
- 3 solvers:
    - CPU - Serial
    - CPU - OpenMP
    - GPU - Shared memory
- Problem size:
    - N = 2^24
- The results below are obtained using a personal laptop with NVIDIA GeForce RTX 2070 with Max-Q Design 

<img src="images/StencilRTX.png" alt="Stencil NVIDIA" width="400"/>
<img src="images/StencilAMD.png" alt="Stencil AMD" width="400"/>

# HW 2
- Matrix multiply (SGEMM & DGEMM)
- C = A * B
    - ***A***: a ***M x K*** matrix
    - ***B***: a ***K x N*** matrix
    - ***C***: a ***M x N*** matrix
- Benchmarking case:
    - M = N = K = 1024
- 2D threadblock/grid indexing
- 9 solvers:
    - CPU 
        - Serial
        - Blas
        - Reordered loop index
        - Outer loop index
        - Cache blocking
        - OpenMP
    - GPU 
        - Naive
        - Shared memory
        - Cublas

- To build cublas/hipblas solvers, use -lcublas/-lhipblas while compiling your code.
- SGEMM & DGEMM results with NVIDIA GeForce RTX 2070 with Max-Q Design & a GPU node on Setonix (AMD MI250X)

<img src="images/SGEMM_RTX.png" alt="SGEMM NVIDIA" width="400"/>
<img src="images/SGEMM_AMD.png" alt="SGEMM AMD" width="400"/>
<img src="images/DGEMM_RTX.png" alt="DGEMM NVIDIA" width="400"/>
<img src="images/DGEMM_AMD.png" alt="DGEMM AMD" width="400"/>

## Dynamic vs Static Shared Memory
### Static
```
const int size = 48;
__global__ void k(...)
{
  __shared__ int temp[SIZE];
  ...
}


k <<< grid, block >>> (...);
```

### Dynamic
```
__global__ void k(...)
{
  __shared__ int temp[];
  ...
}

int sharedSizeInBytes = 192;
k <<< grid, block, sharedSizeInBytes >>> (...);
```
