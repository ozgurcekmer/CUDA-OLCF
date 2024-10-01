# GPU Performance Analysis
- A personal laptop with NVIDIA GeForce RTX 2070 with Max-Q Design is used for the simulations.
- **Nsight-Compute** has been used for the performance analysis.
- The matrix size is selected 4096 x 4096 for this exercise. 
## A note about sector requests & transactions
- Data are stored in 128-byte ***lines*** in cache and in 32-byte ***sectors*** in DRAM.
- So, a cache line is 128 bytes.
- There are 32 threads in a warp (64 in AMD).
- A float is 4 bytes.
- If a warp accesses consecutive 4-byte words (floats), then the ***request*** will be a 128-byte data from ***DRAM***, which can be transferred to cache in 4 transactions (a data segment in DRAM is 32 bytes)
- Hence, the minimum number of transactions per request is 4 for ***float***s.
- If we are using ***double***s, then a request is for a ***8 bytes x 32 threads = 256 bytes***, which would be transferred in 8 transactions (***256 bytes (total bytes needed) / 32 bytes (sector size)***)
## A note about shared memory bank conflicts
- Data are stored in 4-byte-wide ***banks*** in shared memory.
- Typically, there are 32 banks in a shared memory.
- If multiple threads try to access the same bank simultaneously, a bank conflict occurs.
### How to avoid a bank conflict:
- Padding: Use [32][33] instead of [32][32].
- Interleaved access: Use a striding access to data.
- Use of ***restrict***: ***\_\_restrict\_\_*** helps the compiler optimise memory accesses by indicating that pointers do not alias.
## Codes
- Six GPU solvers have been developed for this homework:
  - gpuOriginal1: A naive GPU solver
  - gpuOriginal2: A coalesced GPU solver with shared memory
  - gpuOriginal3: A coalesced GPU solver with shared memory - bank conflicts prevented
  - gpuSolver1: A naive GPU solver - fewer threads than data
  - gpuSolver2: A coalesced GPU solver with shared memory - fewer threads than data
  - gpuSolver3: A coalesced GPU solver with shared memory - bank conflicts prevented - fewer threads than data
- The ***original***s are the ones, which belong to this homework, and the solvers are developed using Mark Harris' developer blog [post](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/#:~:text=In%20this%20post%20I%20will%20show%20some%20of).
- Here are the summarised results for double precision:

| Solver | Trans / req (Load) | Trans / req (Store) | Bank conflicts (Load) | Bank conflicts (Store) | Bandwidth (GB/s) |
| --- | :---: | :---: | :---: | :---: | :---: |
| gpuOriginal1 | 8 | 32 | N/A | N/A | 63.02 |
| gpuOriginal2 | 8 | 8 | ~16M | 0 | 58.69 |
| gpuOriginal3 | 8 | 8 | 0 | 0 | 63.28 |
| gpuSolver1 | 8 | 32 | N/A | N/A | 112.99 |
| gpuSolver2 | 8 | 8 | ~16M | 0 | 106.07 |
| gpuSolver3 | 8 | 8 | 0 | 0 | 106.93 |

- The number of transactions per request becomes 4 times higher than minimum in uncoalesced store access in naive matrix transpose.
```
 b[INDX(row, col, N)] = a[INDX(col, row, N)];
``` 
- Storing data in matrix ***b*** is where we have uncoalesced access here, which we fixed in GPU solvers 2 and 3. 
- A ***padding*** was used to fix the bank conflicts in GPU solver 3.