# Module 3 - Homework Solutions
- The following two systems are used for the analysis:
  - HAGI: A personal laptop with NVIDIA GeForce RTX 2070 with Max-Q Design.
  - SETONIX: A single GPU node with an AMD MI250X GPU.
- ***Nsight-Compute*** and ***OmniPerf*** has been used for the performance analysis for NVIDIA and AMD, respectively.

## HW 1
- A naive ***Vector Addition*** code
- The GPU solution was verified using the CPU code.  

## HW 2
### Performance results - **HAGI**:

| Solver | Solver Runtime (ms) | Kernel Runtime (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| CPU (Serial) | 354.91 | N/A | N/A 
| GPU (1, 1) | 17,089.27 | NaN | NaN 
| GPU (1, 1024) | 160.78 | 59.09 | 7.40 
| GPU (72, 1024)* | 106.04 | 2.13 | 189.16

- **Hagi** has NVIDIA GeForce RTX 2070 with Max-Q Design, which has 36 SMs. To achieve maximum thread occupancy in Hagi SMs, we should use 36 x 2048 threads, since maximum number of threads that occupy an SM is 2048. To make the things more portable, the following lines are included to our GPU solver:
```
int devID;
int numSMs;
gpuGetDevice(&devID);
gpuDeviceGetAttribute(&numSMs, gpuDevAttrMultiProcessorCount, devID);
int blocksPerSM = 2048 / BLOCK_SIZE;
const int gridSize = blocksPerSM * numSMs;
gpuVectorAdd <<< gridSize, BLOCK_SIZE >>> (dA, dB, dC, N);
``` 
* The functions starting with ***gpu*** are defined in **utilities/GpuCommon.h**
### Performance results - **SETONIX**:

| Solver | Solver Runtime (ms) | Kernel Runtime (ms) | Bandwidth (GB/s) |
| --- | --- | --- | --- |
| CPU |  |  |  
| GPU (1, 1) |  |  | 
| GPU (1, 1024) |  |  | 
| GPU (160, 1024) |  |  | 

### ***NOTE:***

- The CPU solver is a serial solver. An OpenMP solver is needed for a healthy comparison.
- nsight-compute didn't work properly for the case with a launch configuration of (1, 1)
- The first GPU rows in both case use only 1 threadblock. In each of these cases, only one SM was used

