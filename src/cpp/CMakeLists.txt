set(CPP_VERSION "latest" CACHE STRING "C++ backend version (v1, v2, ... or latest)")

file(GLOB CPP_VERSIONED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/cpp__perlin_noise_v*.cpp")

set(CPP_VERSIONS "")
foreach(path ${CPP_VERSIONED_FILES})
    get_filename_component(fname ${path} NAME_WE)
    string(REGEX MATCH "v[0-9]+" version_tag ${fname})
    list(APPEND CPP_VERSIONS ${version_tag})
endforeach()

list(SORT CPP_VERSIONS COMPARE NATURAL)
list(GET CPP_VERSIONS -1 CPP_LATEST)

if(CPP_VERSION STREQUAL "latest")
    set(CPP_VERSION ${CPP_LATEST})
endif()

message(STATUS "Building C++ backend version: ${CPP_VERSION}")

set(CPP_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp__perlin_noise_${CPP_VERSION}.cpp"
)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp__perlin_noise_${CPP_VERSION}.cpp")
    message(FATAL_ERROR "C++ source for version ${CPP_VERSION} not found.")
endif()

add_library(perlin_cpp STATIC ${CPP_SRC})

target_compile_options(perlin_cpp PRIVATE -O0)

target_include_directories(perlin_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(perlin_cpp PRIVATE
    BACKEND_NAME="cpp"
    BACKEND_VERSION="${CPP_VERSION}"
)

