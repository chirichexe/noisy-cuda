/*
 * cuda__perlin_noise.cu - perlin noise: CUDA implementation
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

#include "perlin_noise.hpp"
#include "cuda__utils.hpp"

#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h>


__global__ void gpu_generate_perlin_pixel(unsigned char* output, int seed, unsigned int width, unsigned int height) {

    //printf("seed in kernel: %d\n", seed);
    //printf("width in kernel: %d\n", width);
    //printf("height in kernel: %d\n", height);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    size_t idx = y * width + x;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float val = curand_uniform(&state); // value in (0, 1]

    output[idx] = (unsigned char)(val * 255.0f);
}


void generate_perlin_noise(const Options& opts) {


    /* calculate gpu kernel launch parameters */
    size_t buffer_size = opts.width * opts.height * sizeof(unsigned char);
    
    dim3 block(16, 16);
    dim3 grid((opts.width + block.x - 1) / block.x, (opts.height + block.y - 1) / block.y);

    /* allocate device memory for output */
    unsigned char* d_output;
    CHECK(cudaMalloc((void **)&d_output, buffer_size));
    
    /* Launch the kernel */
    gpu_generate_perlin_pixel<<<grid, block>>>(d_output, opts.seed, opts.width, opts.height);

    /* Wait for kernel to complete */
    CHECK(cudaDeviceSynchronize());
   
    /* copy and free the result from the device to the output pointer */
    //CHECK(cudaMemcpy(output, d_output, buffer_size, cudaMemcpyDeviceToHost));
    //CHECK(cudaFree(d_output));
    
    printf("CUDA kernel completed successfully!\n");

   // return 0;
    return;
}