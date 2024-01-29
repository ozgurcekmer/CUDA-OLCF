# CUDA OPTIMIZATION - PART 2
## Global memory transactions
| Kernel | Duration (ms) | Transactions * | Requests ** | Transactions per Requests |
|:-----:|:-----:|-----:|-----:|-----:|
| row_sums | 26.98 | 268,032,526 | 8,388,608 | 31.95 |
| col_sums | 27.80 | 33,554,432 | 8,388,608 | 4.00 |

* *(l1tex_t_sectors...)
* **(l1tex_t_requests...)