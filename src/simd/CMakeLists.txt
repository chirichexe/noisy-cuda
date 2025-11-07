add_library(perlin_simd STATIC
    simd__perlin_noise.cpp
    simd__utils.cpp
)

target_include_directories(perlin_simd PRIVATE ${CMAKE_SOURCE_DIR}/include)
