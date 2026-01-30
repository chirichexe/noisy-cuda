set(SIMD_VERSION "latest" CACHE STRING "SIMD backend version (v1, v2, v3, or latest)")

file(GLOB SIMD_VERSIONED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/simd__perlin_noise_v*.cpp")

set(SIMD_VERSIONS "")
foreach(path ${SIMD_VERSIONED_FILES})
    get_filename_component(fname ${path} NAME_WE)
    string(REGEX MATCH "v[0-9]+" version_tag ${fname})
    list(APPEND SIMD_VERSIONS ${version_tag})
endforeach()

list(SORT SIMD_VERSIONS COMPARE NATURAL)
list(GET SIMD_VERSIONS -1 SIMD_LATEST)

if(SIMD_VERSION STREQUAL "latest")
    set(SIMD_VERSION ${SIMD_LATEST})
endif()

message(STATUS "Building SIMD backend version: ${SIMD_VERSION}")

set(SIMD_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/simd__perlin_noise_${SIMD_VERSION}.cpp"
)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/simd__perlin_noise_${SIMD_VERSION}.cpp")
    message(FATAL_ERROR "SIMD source for version ${SIMD_VERSION} not found.")
endif()

add_library(perlin_simd STATIC ${SIMD_SRC})

if(NOT MSVC)
    target_compile_options(perlin_simd PRIVATE -mavx2 -mfma -O0)
else()
    target_compile_options(perlin_simd PRIVATE /arch:AVX2)
endif()

target_include_directories(perlin_simd PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(perlin_simd PRIVATE
    BACKEND_NAME=\"simd\"
    BACKEND_VERSION=\"${SIMD_VERSION}\"
)