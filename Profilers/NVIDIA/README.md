# NVIDIA Profilers
## Nsight-Compute
### Basic Usage
```
ncu ./myApp
```

### Profile the first N kernels
```
ncu --launch-count N ./myApp

# or

ncu -c N ./myApp
```

### Only Profile Specific Kernel Name
```
ncu --kernel-name regex:MyKernel.* ./myApp
```

### Profile Only After a Delay
```
ncu --launch-skip 3 --launch-count 2 ./myApp

# or

ncu -s 3 -c 2 ./myApp
```

### Use a Predefined Metric Set
```
ncu --set roofline ./myApp
ncu --set speedOfLight ./myApp
ncu --set full ./myApp
```

### Output to Report File (GUI Viewable)
```
ncu -o myApp_profile ./myApp
```
- Generates ***myApp_profile.ncu-rep***
- Can be opened in Nsight-Compute GUI as follows:
```
ncu-ui myApp_profile.ncu-rep
```

### Command-Line Report Summary
```
ncu --import myApp_profile.ncu-rep --csv
```