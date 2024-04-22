# OO Solutions of the CUDA-OLCF Training Series HWs
- This repo was created to propose object-oriented (OO) homework solutions for CUDA training series provided by NVIDIA to OLCF and NERSC ([NVIDIA CUDA series archive](https://www.olcf.ornl.gov/cuda-training-series/)). The original homework repo by NVIDIA is public [here](https://github.com/olcf/cuda-training-series).
- Some of the codes have been prepared to be generic to work on both NVIDIA and AMD GPUs.
- All codes have been developed and tested on a personal laptop (***Hagi***) with an NVIDIA GeForce RTX 2070 with Max-Q Design GPU
- HIP codes have been tested on a GPU node of [***Setonix***](https://pawsey.org.au/systems/setonix/).
- ***Setonix*** has 192 GPU-enabled nodes
    - one 64-core AMD Trento CPU
    - 4 AMD MI250X GPU cards providing 8 logical GPUs per node
    - each MI250X GPU has 2 GCDs (Global Compute Dies)
- The following thirteen modules have been provided by NVIDIA.
## Module 1
- Introduction to CUDA C++

## Module 2
- CUDA Shared Memory

## Module 3
- Fundamental CUDA Optimization (Part 1)

## Module 4
- Fundamental CUDA Optimization (Part 2)

## Module 5
- Atomics, Reductions, and Warp Shuffle

## Module 6
- Managed Memory

## Module 7
- CUDA Concurrency

## Module 8
- GPU Performance Analysis

## Module 9
- Cooperative Groups

## Module 10
- CUDA Multithreading with Streams

## Module 11
- CUDA Multi Process Service

## Module 12
- CUDA Debugging

## Module 13
- CUDA Graphs

## General Notes
### Creating ***Symbolic Links (Symlinks)*** for AMD versions
- The following is used to create ***Symlinks*** of *.cu* files with *.hip* extensions.
```
ln -s cudaFile.cu hipFile.hip
``` 
- It is better only to work on cuda files, since their readibility is better on VIM.

### Working on Setonix GPUs
- The following should be setup:
```
module swap PrgEnv-gnu PrgEnv-cray
module load rocm craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
module load omniperf // if needed
module load omnitrace // if needed

```

