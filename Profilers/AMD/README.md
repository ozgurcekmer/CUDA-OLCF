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