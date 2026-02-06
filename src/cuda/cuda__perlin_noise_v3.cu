/*
 * cuda__perlin_noise_v3.cu - optimized computation with shared memory (LUT + Gradients)
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
 * 
 * @param t 
 * @return float
 */
__device__ __forceinline__ float fade(float t)
{
    // OPT: using fmaf for better performance
    return t * t * t * fmaf(t, fmaf(t, 6.0f, -15.0f), 10.0f);
}

/**
 * @brief Linear interpolation function on CUDA device
 * 
 * @param a 
 * @param b 
 * @param t 
 * @return float
 */
__device__ __forceinline__ float lerp(float a, float b, float t)
{
    // OPT: using fmaf for better performance
    return fmaf(t, b - a, a);
}



// Declaring the global gradients' vectors
float2 gradients[] = {
    {1,1}, {-1,1}, {1,-1}, {-1,-1}, {1,0}, {-1,0}, {0,1}, {0,-1}
};

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
    
    /* SHARED MEMORY for permutation table and gradients */
    __shared__ int s_lookUpTable[512];
    __shared__ float2 s_gradients[8];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    // Cooperative loading of the LUT (512 elements)
    for (int i = tid; i < 512; i += threads_per_block) {
        s_lookUpTable[i] = d_lookUpTable[i];
    }

    // Cooperative loading of the Gradients (8 elements)
    if (tid < 8) {
        s_gradients[tid] = d_gradients[tid];
    }

    // synchronization before computation
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) return;

    // Get the pixel global coordinates with:
    // - offset
    // - frequency scaling (aspect ratio to 1:1)
    float inv_max_dimension = 1.0f / fmaxf((float)width, (float)height);
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    // Accumulate noise value over octaves
    float total_value = 0.0f;

    #pragma unroll // octave loop is not definable at compile time...
    for (int o = 0; o < octaves; o++) {
        
        float noise_x = ((float)(x + offset_x) * inv_max_dimension) * frequency;
        float noise_y = ((float)(y + offset_y) * inv_max_dimension) * frequency;
    
        // Determine the integer coordinates of the cell
        // (grid square) that contains this point
        float cell_left = __float2int_rd(noise_x); // OPT: using __float2int_rd for performance
        float cell_top = __float2int_rd(noise_y);  // OPT: using __float2int_rd for performance
    
        // Get the pixel local coordinates inside the chunk 
        float local_x = noise_x - cell_left;
        float local_y = noise_y - cell_top;
    
        // Wrap cell coordinates to [0, 255] for indexing the permutation table
        // The & 255 operation is equivalent to modulo 256 but faster
        int xi = (int)cell_left & 255;
        int yi = (int)cell_top  & 255;
    
        // Use the permutation table to get pseudo-random gradient indices at the four corners of the cell
        int gi_tl = s_lookUpTable[ s_lookUpTable[xi]     + yi] & 7;
        int gi_tr = s_lookUpTable[ s_lookUpTable[xi + 1] + yi] & 7;
        int gi_bl = s_lookUpTable[ s_lookUpTable[xi]     + yi + 1] & 7;
        int gi_br = s_lookUpTable[ s_lookUpTable[xi + 1] + yi + 1] & 7;
    
        // Select the gradient vectors corresponding to these indices
        float2 grad_top_left     = s_gradients[gi_tl];
        float2 grad_top_right    = s_gradients[gi_tr];
        float2 grad_bottom_left  = s_gradients[gi_bl];
        float2 grad_bottom_right = s_gradients[gi_br];
    
        // Compute vectors from each corner of the cell to the pixel's location
        float2 dist_to_top_left     = make_float2(local_x,        local_y);
        float2 dist_to_top_right    = make_float2(local_x - 1.0f, local_y);
        float2 dist_to_bottom_left  = make_float2(local_x,        local_y - 1.0f);
        float2 dist_to_bottom_right = make_float2(local_x - 1.0f, local_y - 1.0f);
    
        // Calculate the dot product between distance vectors and gradient vectors
        // This gives the influence (contribution) of each corner on the final noise value
        float influence_top_left     = grad_top_left.x * dist_to_top_left.x + grad_top_left.y * dist_to_top_left.y;
        float influence_top_right    = grad_top_right.x * dist_to_top_right.x + grad_top_right.y * dist_to_top_right.y;
        float influence_bottom_left  = grad_bottom_left.x * dist_to_bottom_left.x + grad_bottom_left.y * dist_to_bottom_left.y;
        float influence_bottom_right = grad_bottom_right.x * dist_to_bottom_right.x + grad_bottom_right.y * dist_to_bottom_right.y;
    
        // Interpolate the influences along the x-axis using a fade function for smoothness
        float fade_local_x = fade(local_x); // OPT: calculate fade only once
        float fade_local_y = fade(local_y); // OPT: calculate fade only once
        
        float interp_top    = lerp(influence_top_left, influence_top_right, fade_local_x);
        float interp_bottom = lerp(influence_bottom_left, influence_bottom_right, fade_local_x);
        
        // Interpolate the top and bottom results along the y-axis to get the final Perlin noise value
        float pixel_noise_value = lerp(interp_top, interp_bottom, fade_local_y);
    
        // Accumulate the noise value scaled by the current amplitude
        total_value = fmaf(pixel_noise_value, amplitude, total_value);

        // controls frequency growth
        frequency *= lacunarity;
        // controls amplitude decay
        amplitude *= persistence;

    }

    // Accumulate the computed noise into the output array, scaling by the amplitude
    d_accumulator[y * width + x] = total_value;
}

void generate_perlin_noise(const Options& opts) {
    /* initialize parameters */
    // noise parameters
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
    //bool verbose = opts.verbose;
    bool benchmark = opts.benchmark;

    // output info
    std::string output_filename = opts.output_filename;
    std::string output_format = opts.format;

    /* randomize from seed */
    srand(seed);

    // build the permutation tables
    std::vector<int> lookUpTable(512);
    for (int i = 0; i < 256; i++) lookUpTable[i] = i;

    // shuffling the table with Fisher-Yates approach
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(lookUpTable[i], lookUpTable[j]);
    }

    // avoid overflows doubling the table
    for (int i = 0; i < 256; i++) lookUpTable[256 + i] = lookUpTable[i];

    /* start profiling timers */
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

    
    /* accumulator host (needed for octaves) */
    std::vector<float> accumulator(width * height, 0.0f);
    
    /* copy accumulator to device */
    float* d_accumulator;
    size_t accumulator_bytes = width * height * sizeof(float);
    
    CHECK(cudaMalloc(&d_accumulator, accumulator_bytes ));
    CHECK(cudaMemcpy(d_accumulator, accumulator.data(), accumulator_bytes, cudaMemcpyHostToDevice));
    
    /* copy lookup table and gradients to constant memory */
    size_t lookUpTable_bytes = 512 * sizeof(int);
    size_t gradients_bytes = 8 * sizeof(float2);
    CHECK(cudaMemcpyToSymbol(d_lookUpTable, lookUpTable.data(), lookUpTable_bytes));
    CHECK(cudaMemcpyToSymbol(d_gradients, gradients, gradients_bytes));

    // Configure kernel launch parameters
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    /* generate noise, the octave loop is inside */
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

    // Copy result back to host
    CHECK(cudaMemcpy(accumulator.data(), d_accumulator, accumulator_bytes, cudaMemcpyDeviceToHost));

    /* stop profiling timers and report */
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

        size_t estimated_total_alloc = accumulator_bytes;

        std::time_t now = std::time(nullptr);
        //std::cout << "timestamp,width,height,pixels,octaves,frequency,wall_ms,cpu_s,ms_per_pixel,mem_bytes\n";
        std::cout << now << ","
                  << width << ","
                  << height << ","
                  << num_pixels << ","
                  << octaves << ","
                  << base_frequency << ","
                  << wall_ms << ","
                  << cpu_seconds << ","
                  << ms_per_pixel << ","
                  << estimated_total_alloc << std::endl;

        CHECK(cudaEventDestroy(cuda_start));
        CHECK(cudaEventDestroy(cuda_stop));
    }

    /* save the generated noise image */
    if (!no_outputs){

        // create the output array
        std::vector<uint8_t> output(width * height, 0);
        for (int i = 0; i < width * height; i++) {
            // mapping the accumulator (-1, 1) to grayscaled pixels (0-255)
            float v = std::clamp( accumulator[i] , -1.0f, 1.0f);
            output[i] = static_cast<uint8_t>((v + 1.0f) * 0.5f * 255.0f);
        }

        save_output(
            output,
            width,
            height,
            1, // only 1 channel, b&w
            output_filename,
            output_format
        );
    }

    // Cleanup
    CHECK(cudaFree(d_accumulator));
}