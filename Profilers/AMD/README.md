# AMD Profilers
A brief information about the following profilers, which are used in this tutorial:
- ROC-profiler (rocprof)
- Omnitrace
- Omniperf

## rocprof
### usage
```
srun rocprof --<options> appExe
```
### options used in this repo
- ***--stats*** gives durations of kernels and memory transfers
- ***--hip-trace*** gives a .json output, which can be visualised using the Chrome browser and [***perfetto***](https://ui.perfetto.dev/)

## Omnitrace
### usage
```
srun omnitrace --<options> appExe
```
### options used in this repo
- ***omnitrace -h***: help file for more information
- ***--omnitrace-instrument***
```
// Binary rewrite
omnitrace-instrument [omnitrace-options] -o <newNameOfExec> -- <CMD> <ARGS>

// EXAMPLE: Generating a new library with instrumentation built-in
omnitrace-instrument -o Jacobi_hip.inst -- ./Jacobi_hip

// Run the instrumented binary
srun <srun args> omnitrace-run -- ./Jacobi_hip.inst

```
#### Hardware Counters: List All
```
srun ... omnitrace-avail --all
```

## Omniperf
- Profile
```
omniperf profile -n workload_name [profile options] [roofline options] -- <CMD> <ARGS>
```
- Analyze
```
omniperf analyze -p <path_to_workloads_mi200>

// write the output in a text file
omniperf analyze -p <path_to_workloads_mi200> &> newAnalyze.txt

// include the analysis of only the specified kernel
omniperf analyze -p <path_to_workloads_mi200> -k 1 &> newAnalyze.txt
```

- To use a lightweight standalone GUI with CLI analyzer
```
omniperf analyze -p <path_to_workloads_mi200> --gui
```
- Database
```
omniperf database <interaction type> [connection options]
```
- More information
```
omniperf profile --help
```




