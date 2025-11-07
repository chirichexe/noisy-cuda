add_library(perlin_cpp STATIC
    cpp__perlin_noise.cpp
)

target_include_directories(perlin_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
