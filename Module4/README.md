# CUDA OPTIMIZATION - PART 2
## Global memory transactions
| Kernel | Duration (ms) | Transactions <sup>1</sup> | Requests <sup>2</sup> | Transactions per Requests |
|:-----:|:-----:|-----:|-----:|-----:|
| row_sums | 24.62 | 268,225,475 | 8,388,608 | 31.95 |
| col_sums | 25.37 | 33,554,432 | 8,388,608 | 4.00 |

1. l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    - The number of global memory load requests
    - Numerator
2. l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
    - The number of sectors requested for global loads
    - Denominator

- Dividing 1 / 2 gives the ***number of transactions per request (TPR)***
- More transactions appeared per request (less efficiency) in row_sums. 
- Although, row_sums has a higher (8 times more) TPR, the runtime is slightly less than that of col_sums. This is unexpected, since the memory coalescence (hence the efficiency) is much better in col_sums kernel.
### To do
- Try to warmup GPU before letting GPU solvers work.
- Find another system for the simulations
