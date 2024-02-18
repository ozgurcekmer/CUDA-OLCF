# Module 5 - Homework Solutions
- A personal laptop with NVIDIA GeForce RTX 2070 with Max-Q Design is used for the simulations.
- ***Nsight-Compute*** has been used for the performance analysis.

## HW 1
- Sum of the elements of a vector

## HW 2
- Finding the maximum and the index of the maximum in an array
 
## HW 3
- A GPU solver for HW4


### ***NOTE:***
- All codes have been tested on a personal laptop with an NVIDIA GeForce RTX 2070 with Max-Q Design GPU
- The CPU solver is a serial solver. An OpenMP solver is needed for a healthy comparison.
- nsight-compute didn't work properly for the case with a launch configuration of (1, 1)
- The first three GPU rows use only 1 threadblock. In each of these cases, only one SM was used
- The GPU that I used has 36 SMs. Each SM can handle 2048 threads. The following launch configurations were arranged to reach full occupancy. 
    - (288, 256)
    - (72, 1024)
- Target number of threads for the current GPU: ***2048 x 36 = 73,728***
- Optimum performance is reached with these 2 configurations
