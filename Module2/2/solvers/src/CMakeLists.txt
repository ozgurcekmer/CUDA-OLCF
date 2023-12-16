# Add source to this project's executable.
add_library (SourceCodeCMake 
             MatrixMultCPU.cpp
             MatrixMultBlas.cpp
             MatrixMultGPU.cu
             MatrixMultShared.cu
             BlockedCPU.cpp
             BlasGPU.cu
             ReorderedCPU.cpp
             OmpCPU.cpp
             OuterCPU.cpp
             )
