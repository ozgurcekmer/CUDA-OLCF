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
| Solver | Solver Runtime (ms) | Kernel Runtime (ms) | L1 (GB/s) | L2 - Read (GB/s) | L2 - Write (GB/s) |
| --- | ---: | ---: | ---: | ---: | ---: | 
| CPU | 15.09 | N/A | N/A | N/A | N/A |
| GPU (1, 1) | 6,981.72 | 6,962.53 | 0.93 | 0.04 | 0.02 | 
| GPU (1, 1024) | 35.78 | 16.60 | 24.25 | 16.17 | 7.99 |
| GPU (220, 1024)* | 19.52 | 0.36 | 1,126.99 | 751.34 | 370.96 |
* A single GPU on a Setonix GPU node has 110 SMs. To achieve the maximum occupancy (2048 threads per SM), we need to use 2 blocks per SM since we are using 1024 threads per blocks.
- Build, run & omniperf commands:
```
# build
hipcc -x hip -std=c++17 ../main.cpp ../utilities/src/*.cpp ../utilities/src/*.hip ../solvers/src/*.cpp ../solvers/src/*.hip -o game -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --offload-arch=gfx90a -fopenmp -O2 -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa -DUSEHIP

# run with omniperf profiler
srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest omniperf profile -n workload_B1T1 --roof-only -k 0 -- ./game

# profiler analysis
omniperf analyze -p workloads/workload_B1T1024/mi200/ -k 0 &> workloads/workload_B1T1024/B1T1024_Analyze.txt

```
- ***-k kernelNumber*** is used to profile only the specified kernel

### ***NOTE:***

- The CPU solver is a serial solver. An OpenMP solver is needed for a healthy comparison.
- nsight-compute didn't work properly for the case with a launch configuration of (1, 1)
- The first GPU rows in both case use only 1 threadblock. In each of these cases, only one SM was used

