# Module 3 - Homework SOlutions
- A personal laptop with NVIDIA GeForce RTX 2070 with Max-Q Design is used for the simulations.
- ***Nsight-Compute*** has been used for the performance analysis.

## HW 1
- A naive ***Vector Addition*** code
- The GPU solution was tested against the CPU one.  

## HW 2
- Performance results:

| Solver | Solver Runtime (ms) | Kernel Runtime (ms) | Bandwidth (GB/s) |
| --- | --- | --- | --- |
| CPU | 343.43 | N/A | N/A 
| CUDA (1, 1) | 37049.83 | N/A | N/A
| CUDA (1, 256) | 343.56 | 232.94 | 2.39
| CUDA (1, 1024) | 168.02 | 59.08 | 7.40
| ***CUDA (288, 256)*** | 97.64 | 2.14 | 187.63
| ***CUDA (72, 1024)*** | 97.74 | 2.12 | 190.08
| CUDA (160, 1024) | 114.46 | 2.31 | 176.44
| CUDA (1152, 1024) | 107.48 | 2.15 | 189.19 
| CUDA (1152, 256) | 107.58 | 2.11 | 191.48
 
### ***NOTE:***
- All codes have been tested on a personal laptop with an NVIDIA GeForce RTX 2070 with Max-Q Design GPU
- The CPU solver is a serial solver. An OpenMP solver is needed for a healthy comparison.
- nsight-compute didn't work properly for the case with a launch configuration of (1, 1)
- The first two rows use only 1 threadblock. In each of these cases, only one SM was used
- The GPU that I used has 36 SMs. Each SM can handle 2048 threads. The following launch configurations were arranged to reach full occupancy. 
    - (288, 256)
    - (72, 1024)
- Target number of threads for the current GPU: ***2048 x 36 = 73,728***
- Optimum performance is reached with these 2 configurations
