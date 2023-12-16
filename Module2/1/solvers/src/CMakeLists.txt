# Add source to this project's executable.
add_library (SourceCodeCMake 
             StencilCPU.cpp
             StencilOmp.cpp
             StencilGPU.cu
             )
