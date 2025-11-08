cmake -B build_cuda -DUSE_CUDA=ON
cmake --build build_cuda

cmake -B build_simd_v1 -DUSE_SIMD=ON -DSIMD_VERSION=v1
cmake --build build_simd_v1
