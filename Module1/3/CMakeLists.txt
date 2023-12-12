cmake_minimum_required(VERSION 3.18)

# set the project name
project(MatrixMult
    VERSION 1.0
    LANGUAGES CXX)

# specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

macro(MatrixMult_option optname optdesc status)
        option(MATRIXMULT_${optname} "${optdesc}" "${status}")
endmacro()

# MatrixMult_option(USE_HIP "Attempt to use HIP instead of CUDA" OFF)
MatrixMult_option(USE_HIP "Attempt to use HIP instead of CUDA" OFF)
MatrixMult_option(USE_AMD "Attempt to use HIP AMD" ON)

# set the GPU language and standard
if (MATRIXMULT_USE_HIP)
    # Find hip
    find_package(HIP REQUIRED)
    # Link with HIP
    enable_language(HIP)
    if(NOT DEFINED CMAKE_HIP_STANDARD)
        set(CMAKE_HIP_STANDARD 17)
        set(CMAKE_HIP_STANDARD_REQUIRED ON)
    endif()
    add_definitions("-DUSEHIP")
    add_compile_options("-fPIE")
    target_link_libraries(hip::device)
    if (MATRIXMULT_USE_AMD)
        add_definitions("-D__HIP_PLATFORM_AMD__")
        set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
    endif()
else()
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    add_definitions("-DUSECUDA")
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    endif()
endif()
        
configure_file(MatrixMult.h.in MatrixMult.h)

# add the executable
add_executable(MatrixMult main.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
        target_link_libraries(MatrixMult PUBLIC OpenMP::OpenMP_CXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O2")
endif() 

# add the include library
add_subdirectory("utilities")
add_subdirectory("solvers/src")

target_link_libraries(MatrixMult PUBLIC UtilitiesCMake)
target_link_libraries(MatrixMult PUBLIC SourceCodeCMake)
# target_link_libraries(MatrixMult PUBLIC -lcublas -lcurand)

target_include_directories(MatrixMult PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_BINARY_DIR}/solvers/src"
                           "${PROJECT_BINARY_DIR}/utilities"
                           )
