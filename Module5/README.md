# Module 5 - Homework Solutions
- ***A VERY IMPORTANT INITIAL NOTE:*** 
  - AtomicXXX cannot work with templates
  - AtomicXXX doesn't work with ***double***?
- A personal laptop with NVIDIA GeForce RTX 2070 with Max-Q Design is used for the simulations.
- ***Nsight-Compute*** has been used for the performance analysis.

## HW 1
- Sum of the elements of a vector
### Runtime in **$\mu s$**
| N | Naive Atomic | ReduceA | Warp Shuffle (WS) |
| --- | --- | --- | --- |
| 72 x 1024 (73,728) | 0.27 | 0.05 | 0.10 |
| 8M | 28.95 | 0.39 | 0.36 |
| 32M | N/A | 1.37 | 1.14 |

### Memory Bandwidth in GB/s (% of machine peak bandwidth)
| N | Naive Atomic | ReduceA | Warp Shuffle (WS) |
| --- | --- | --- | --- |
| 72 x 1024 (73,728) | 3.92 (2.17%) | 7.47 (3.93%) | 15.25 (15.21%) |
| 8M | 1.83 (1.84%) | 90.85 (26.37%) | 101.93 (29.87%) |
| 32M | N/A | 99.52 (28.47%) | 119.8 (34.29%) |

* Kernel execution time for WS is the lowest for bigger problem sizes. This may be caused by the decreased usage of shared memory.  

* The ***naive atomic solver*** doesn't work correctly when problem size is increased to 32M.

## HW 2
- Finding the maximum in an array
 
## HW 3
- A new GPU solver for HW4
### Older version:
| Kernel | Duration (ms) | Transactions <sup>1</sup> | Requests <sup>2</sup> | Transactions per Requests |
|:-----:|:-----:|-----:|-----:|-----:|
| row_sums | 24.62 | 268,225,475 | 8,388,608 | 31.95 |
| col_sums | 25.37 | 33,554,432 | 8,388,608 | 4.00 |
### New row_sums version with reduction:
| Kernel | Duration (ms) | Transactions <sup>1</sup> | Requests <sup>2</sup> | Transactions per Requests |
|:-----:|:-----:|-----:|-----:|-----:|
| row_sums | 15.55 | 33,554,432 | 8,388,608 | 4.00 |
| col_sums | 25.37 | 33,554,432 | 8,388,608 | 4.00 |

### ***NOTE:***
- All codes have been tested on a personal laptop with an NVIDIA GeForce RTX 2070 with Max-Q Design GPU
- The GPU that I used has 36 SMs. Each SM can handle 2048 threads. The following launch configuration was arranged to reach full occupancy. 
    - (72, 1024)
- Target number of threads for the current GPU for the optimum occupancy: ***2048 x 36 = 73,728***
