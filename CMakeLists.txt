cmake_minimum_required(VERSION 3.18)
project(noisy_cuda LANGUAGES CXX CUDA)

# Options to choose which backend to compile
option(USE_CUDA "Build CUDA backend" OFF)
option(USE_SIMD "Build SIMD backend" OFF)
option(USE_CPP  "Build pure C++ backend" OFF)

# Common source files (main and options)
set(COMMON_SRC
    src/main.cpp
    src/options.cpp
)

# Choose backend
if(USE_CUDA)
    add_subdirectory(src/cuda)
    set(BACKEND_TARGET perlin_cuda)
elseif(USE_SIMD)
    add_subdirectory(src/simd)
    set(BACKEND_TARGET perlin_simd)
elseif(USE_CPP)
    add_subdirectory(src/cpp)
    set(BACKEND_TARGET perlin_cpp)
else()
    message(FATAL_ERROR "You must enable one backend (USE_CUDA, USE_SIMD, or USE_CPP)")
endif()

# Final executable
add_executable(noisy_cuda ${COMMON_SRC})
target_include_directories(noisy_cuda PRIVATE include)
target_link_libraries(noisy_cuda PRIVATE ${BACKEND_TARGET})
