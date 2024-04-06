# HW 1
- A ***Hello World*** program
- ***Extra:*** Added an OpenMP solver

# HW 2
- Vector addition
- 2 solvers:
    - CPU - Serial
    - GPU
- The aim of this homework is writing ***syntactically*** correct GPU kernels. The CPU code is developed to test the GPU kernel. Hence, only a limited performance analysis is provided here.

| System | Solver | Runtime (ms)|
|:--|:--:|--:|
|Hagi|CPU|206.84|
|Hagi|GPU|55.14|
|Setonix|CPU|7.39|
|Setonix|GPU|9.97|


# HW 3
- Matrix multiply
- 2D threadblock/grid indexing
- 3 solvers:
    - CPU - Serial
    - CPU - Blas
    - GPU - Naive
- Detailed performance results of the solvers are given in the next chapter with the additional solvers.
- Only runtimes are provided here

| System | Solver | Runtime (ms)|
|:--|:--:|--:|
|Hagi|CPU|2204.61|
|Hagi|BLAS|9.79|
|Hagi|GPU|14.46|
|Setonix|CPU|303.54|
|Setonix|GPU|1.27|

## ***NOTE:***
- The emphasis in this problem set was on the correctness of the GPU usage (kernel development). Hence, the performance results will be analysed in the following modules.
- All codes have been developed and tested on a personal laptop (***Hagi***) with an NVIDIA GeForce RTX 2070 with Max-Q Design GPU
- HIP codes have been tested on a GPU node of [***Setonix***](https://pawsey.org.au/systems/setonix/).
- ***Setonix*** has 192 GPU-enabled nodes
    - one 64-core AMD Trento CPU
    - 4 AMD MI250X GPU cards providing 8 logical GPUs per node
    - each MI250X GPU has 2 GCDs (Global Compute Dies)