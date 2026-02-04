/*
 * cuda__perlin_noise_v3.cu - shared memory for lookup table
 *
 */

/*
 * Copyright 2025 Davide Chirichella, Filippo Giulietti
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * Distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "perlin_noise.hpp"
#include "utils_global.hpp"
#include "utils_cuda.hpp"

#include <algorithm>
#include <inttypes.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>


/* chunk variables, CUDA adapted */
#define BLOCK_SIZE 16

/**
 * @brief  Smoothing function for Perlin noise on CUDA device
 * * @param t 
 * @return float
 */
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

/**
 * @brief Linear interpolation function on CUDA device
 * * @param a 
 * @param b 
 * @param t 
 * @return float
 */
__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Declaring the global gradients' vectors using float2
const float2 gradients[] = {
    {1,1}, {-1,1}, {1,-1}, {-1,-1}, {1,0}, {-1,0}, {0,1}, {0,-1}
};

// Declaring the permutation table (look-up table) and gradients as constant memory on the device
__constant__ float2 d_gradients[8];
__constant__ int d_lookUpTable[512];


/**
 * @brief CUDA kernel for Perlin noise generation, equivalent to Chunk
 */
__global__ void perlin_noise_kernel(
    int width,
    int height,
    int octaves,
    float base_frequency,
    float base_amplitude,
    float lacunarity,
    float persistence,
    int offset_x,
    int offset_y,
    float* d_accumulator
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    /* loading shared memory for lookup table and gradients */
    __shared__ int shared_lookUpTable[512];
    __shared__ float2 shared_gradients[8];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    // Load lookup table
    for (int i = tid; i < 512; i += threads_per_block) {
        shared_lookUpTable[i] = d_lookUpTable[i];
    }

    // Load gradients into shared memory
    if (tid < 8) {
        shared_gradients[tid] = d_gradients[tid];
    }

    // wait until all copies are done
    __syncthreads();


    // Boundary check
    if (x >= width || y >= height) return;

    float max_dimension = fmaxf((float)width, (float)height);
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    float total_value = 0.0f;

    #pragma unroll
    for (int o = 0; o < octaves; o++) {
        
        float noise_x = ((float)(x + offset_x) / max_dimension) * frequency;
        float noise_y = ((float)(y + offset_y) / max_dimension) * frequency;
    
        float cell_left = floorf(noise_x);
        float cell_top = floorf(noise_y);
    
        float local_x = noise_x - cell_left;
        float local_y = noise_y - cell_top;
    
        int xi = (int)cell_left & 255;
        int yi = (int)cell_top  & 255;
    
        int grad_index_top_left     = shared_lookUpTable[ shared_lookUpTable[xi]     + yi] & 7;
        int grad_index_top_right    = shared_lookUpTable[ shared_lookUpTable[xi + 1] + yi] & 7;
        int grad_index_bottom_left  = shared_lookUpTable[ shared_lookUpTable[xi]     + yi + 1] & 7;
        int grad_index_bottom_right = shared_lookUpTable[ shared_lookUpTable[xi + 1] + yi + 1] & 7;
    
        // Select the gradient vectors from SHARED memory
        float2 grad_top_left     = shared_gradients[grad_index_top_left];
        float2 grad_top_right    = shared_gradients[grad_index_top_right];
        float2 grad_bottom_left  = shared_gradients[grad_index_bottom_left];
        float2 grad_bottom_right = shared_gradients[grad_index_bottom_right];
    
        float2 dist_to_top_left     = make_float2(local_x,        local_y);
        float2 dist_to_top_right    = make_float2(local_x - 1.0f, local_y);
        float2 dist_to_bottom_left  = make_float2(local_x,        local_y - 1.0f);
        float2 dist_to_bottom_right = make_float2(local_x - 1.0f, local_y - 1.0f);
    
        float influence_top_left     = grad_top_left.x * dist_to_top_left.x + grad_top_left.y * dist_to_top_left.y;
        float influence_top_right    = grad_top_right.x * dist_to_top_right.x + grad_top_right.y * dist_to_top_right.y;
        float influence_bottom_left  = grad_bottom_left.x * dist_to_bottom_left.x + grad_bottom_left.y * dist_to_bottom_left.y;
        float influence_bottom_right = grad_bottom_right.x * dist_to_bottom_right.x + grad_bottom_right.y * dist_to_bottom_right.y;
    
        float interp_top    = lerp(influence_top_left, influence_top_right, fade(local_x));
        float interp_bottom = lerp(influence_bottom_left, influence_bottom_right, fade(local_x));
        
        float pixel_noise_value = lerp(interp_top, interp_bottom, fade(local_y));
    
        total_value += pixel_noise_value * amplitude;

        frequency *= lacunarity;
        amplitude *= persistence;
    }

    d_accumulator[y * width + x] = total_value;
}

void generate_perlin_noise(const Options& opts) {
    /* initialize parameters */
    std::uint64_t seed = opts.seed;
    int width = opts.width;
    int height = opts.height;
    float base_frequency = opts.frequency;
    float base_amplitude = opts.amplitude;
    int octaves = opts.octaves;
    float lacunarity = opts.lacunarity;
    float persistence = opts.persistence;
    int offset_x = opts.offset_x;
    int offset_y = opts.offset_y;
    bool no_outputs = opts.no_outputs;
    bool benchmark = opts.benchmark;

    std::string output_filename = opts.output_filename;
    std::string output_format = opts.format;

    srand(seed);

    std::vector<int> lookUpTable(512);
    for (int i = 0; i < 256; i++) lookUpTable[i] = i;

    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(lookUpTable[i], lookUpTable[j]);
    }

    for (int i = 0; i < 256; i++) lookUpTable[256 + i] = lookUpTable[i];

    std::chrono::high_resolution_clock::time_point wall_start;
    clock_t cpu_start = 0;
    cudaEvent_t cuda_start, cuda_stop;
    
    if (benchmark) {
        wall_start = std::chrono::high_resolution_clock::now();
        cpu_start = std::clock();
        CHECK(cudaEventCreate(&cuda_start));
        CHECK(cudaEventCreate(&cuda_stop));
        CHECK(cudaEventRecord(cuda_start));
    }

    std::vector<float> accumulator(width * height, 0.0f);
    float* d_accumulator;
    
    size_t gradients_bytes = 8 * sizeof(float2);
    size_t lookUpTable_bytes = 512 * sizeof(int);
    size_t accumulator_bytes = width * height * sizeof(float);

    CHECK(cudaMalloc(&d_accumulator, accumulator_bytes ));
    CHECK(cudaMemcpy(d_accumulator, accumulator.data(), accumulator_bytes, cudaMemcpyHostToDevice));
    
    CHECK(cudaMemcpyToSymbol(d_gradients, gradients, gradients_bytes));
    CHECK(cudaMemcpyToSymbol(d_lookUpTable, lookUpTable.data(), lookUpTable_bytes));
    
    // Configure Shared Memory Carveout before launch
    //CHECK(cudaFuncSetAttribute(perlin_noise_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    perlin_noise_kernel<<<gridSize, blockSize>>>(
        width,
        height,
        octaves,
        base_frequency,
        base_amplitude,
        lacunarity,
        persistence,
        offset_x,
        offset_y,
        d_accumulator
    );
        
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(accumulator.data(), d_accumulator, accumulator_bytes, cudaMemcpyDeviceToHost));

    if (benchmark) {
        CHECK(cudaEventRecord(cuda_stop));
        CHECK(cudaEventSynchronize(cuda_stop));

        clock_t cpu_end = std::clock();
        auto wall_end = std::chrono::high_resolution_clock::now();

        float cuda_ms = 0;
        CHECK(cudaEventElapsedTime(&cuda_ms, cuda_start, cuda_stop));

        double cpu_ticks = static_cast<double>(cpu_end - cpu_start);
        double cpu_seconds = cpu_ticks / static_cast<double>(CLOCKS_PER_SEC);
        double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_end - wall_start).count();

        size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        double ms_per_pixel = (num_pixels > 0) ? (wall_ms / (double)num_pixels) : 0.0;

        std::time_t now = std::time(nullptr);
        std::cout << now << "," << width << "," << height << "," << num_pixels << ","
                  << octaves << "," << base_frequency << "," << wall_ms << ","
                  << cpu_seconds << "," << ms_per_pixel << "," << accumulator_bytes << std::endl;

        CHECK(cudaEventDestroy(cuda_start));
        CHECK(cudaEventDestroy(cuda_stop));
    }

    if (!no_outputs){
        std::vector<uint8_t> output(width * height, 0);
        for (int i = 0; i < width * height; i++) {
            float v = std::clamp( accumulator[i] , -1.0f, 1.0f);
            output[i] = static_cast<uint8_t>((v + 1.0f) * 0.5f * 255.0f);
        }
        save_output(output, width, height, 1, output_filename, output_format);
    }

    CHECK(cudaFree(d_accumulator));
}