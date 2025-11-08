# Project and cmake properties
cmake_minimum_required(VERSION 3.18)
project(noisy_cuda LANGUAGES CXX CUDA)

option(USE_CUDA "Build CUDA backend" OFF)
option(USE_SIMD "Build SIMD backend" OFF)
option(USE_CPP  "Build pure C++ backend" OFF)

set(COMMON_SRC
    src/main.cpp
    src/options.cpp
)

# Backend selection and building
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

# Define backend name and version for the executable
if(USE_CUDA)
    target_compile_definitions(noisy_cuda PRIVATE BACKEND_NAME="cuda" BACKEND_VERSION="${CUDA_VERSION}")
elseif(USE_SIMD)
    target_compile_definitions(noisy_cuda PRIVATE BACKEND_NAME="simd" BACKEND_VERSION="${SIMD_VERSION}")
elseif(USE_CPP)
    target_compile_definitions(noisy_cuda PRIVATE BACKEND_NAME="cpp" BACKEND_VERSION="${CPP_VERSION}")
endif()

# Define program version
set(PROGRAM_VERSION_MAJOR 0)
set(PROGRAM_VERSION_MINOR 0)
set(PROGRAM_VERSION_PATCH 1)

set(PROGRAM_VERSION "${PROGRAM_VERSION_MAJOR}.${PROGRAM_VERSION_MINOR}.${PROGRAM_VERSION_PATCH}")

add_definitions(-DPROGRAM_VERSION="${PROGRAM_VERSION}")

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    set(GIT_HASH "unknown")
endif()

add_definitions(-DGIT_HASH="${GIT_HASH}")
