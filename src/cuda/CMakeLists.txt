set(CUDA_VERSION "latest" CACHE STRING "CUDA backend version (v1, v2, ..., or latest)")

file(GLOB CUDA_VERSIONED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/cuda__perlin_noise_v*.cu")

set(CUDA_VERSIONS "")
foreach(path ${CUDA_VERSIONED_FILES})
    get_filename_component(fname ${path} NAME_WE)
    string(REGEX MATCH "v[0-9]+" version_tag ${fname})
    list(APPEND CUDA_VERSIONS ${version_tag})
endforeach()

list(SORT CUDA_VERSIONS COMPARE NATURAL)
list(GET CUDA_VERSIONS -1 CUDA_LATEST)

if(CUDA_VERSION STREQUAL "latest")
    set(CUDA_VERSION ${CUDA_LATEST})
endif()

message(STATUS "Building CUDA backend version: ${CUDA_VERSION}")

set(CUDA_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda__perlin_noise_${CUDA_VERSION}.cu"
    #"${CMAKE_CURRENT_SOURCE_DIR}/cuda__utils.cu"
)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda__perlin_noise_${CUDA_VERSION}.cu")
    message(FATAL_ERROR "CUDA source for version ${CUDA_VERSION} not found.")
endif()

add_library(perlin_cuda STATIC ${CUDA_SRC})

target_compile_options(perlin_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O0>
)

set_target_properties(perlin_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(perlin_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(perlin_cuda PRIVATE
    BACKEND_NAME=\"cuda\"
    BACKEND_VERSION=\"${CUDA_VERSION}\"
)