add_library(perlin_cuda STATIC
    cuda__perlin_noise.cu
    cuda__utils.cu
)

set_target_properties(perlin_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(perlin_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)