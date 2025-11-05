/*
 * cuda_kernel.cu - specific run of a kernel CUDA implementation
 *
 */

/*
 * Copyright 2025 Davide Chirichella, Filippo Giulietti
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gpu_generate_perlin.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h>



#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


__global__ void gpu_generate_perlin_pixel(unsigned char* output, int seed, unsigned int width, unsigned int height) {

    printf("seed in kernel: %d\n", seed);
    printf("width in kernel: %d\n", width);
    printf("height in kernel: %d\n", height);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float val = curand_uniform(&state); // value in (0, 1]
    output[idx] = (unsigned char)(val * 255.0f);
}


int gpu_generate_perlin(ProgramOptions *opts, unsigned char* output) {
    unsigned char* d_output;

    // Simple kernel configuration
    int num_threads = opts->width * opts->height;
    int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;

    // Allocate device memory for output
    CHECK(cudaMalloc((void **)&d_output, sizeof(d_output)));

    // Set up grid and block dimensions
    dim3 block(255, 255);
    dim3 grid((opts->width + block.x - 1) / block.x, (opts->height + block.y - 1) / block.y);
    
    //printf("Kernel configuration: <<<%d blocks, %d threads>>>\n", grid_size, block_size);
    
    
    // Launch the kernel
    gpu_generate_perlin_pixel<<<grid_size, block_size>>>(d_output, opts->seed, opts->width, opts->height);
    // simple_print_kernel<<<grid_size, block_size>>>(num_threads);
    
    // Wait for kernel to complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    //CHECK(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyDeviceToHost));
    
    printf("CUDA kernel completed successfully!\n");
    return 0;
}