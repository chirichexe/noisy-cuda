/*
 * cuda__perlin_noise_v1.cu - perlin noise: CUDA implementation
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

#include "utils_global.hpp"
#include "utils_cuda.hpp"
#include "options.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <inttypes.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>

#include <cstdint>  // Risolve: error: identifier "uint8_t" is undefined
#include <string>   // Necessario per std::string
#include <cctype>   // Risolve il warning/errore su std::tolower

/* device variables */
#define BLOCK_SIZE 32

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32

// Declaring the global gradients' vectors
/*
*/
__device__ __constant__ Vector2D d_gradients[8] = {
    {1,1}, {-1,1}, {1,-1}, {-1,-1}, {1,0}, {-1,0}, {0,1}, {0,-1}
};

/**
 * @brief CUDA kernel to generate Perlin noise for each pixel
 */
__global__ void gpu_generate_perlin_pixel(
    float* accumulator,
    int width,
    int height,
    const int* lookUpTable,
    float frequency,
    float amplitude,
    int offset_x,
    int offset_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) 
        return;

    // Get the pixel global coordinates with:
    // - offset
    // - frequency scaling (aspect ratio to 1:1)
    float max_dimension = (float)((width > height) ? width : height);
    float noise_x = ((float)(x + offset_x) / max_dimension) * frequency;
    float noise_y = ((float)(y + offset_y) / max_dimension) * frequency;

    // Determine the integer coordinates of the cell
    // (grid square) that contains this point
    int cell_left = (int)floorf(noise_x);
    int cell_top = (int)floorf(noise_y);

    // Get the pixel local coordinates inside the chunk 
    float local_x = noise_x - (float)cell_left;
    float local_y = noise_y - (float)cell_top;

    // Wrap cell coordinates to [0, 255] for indexing the permutation table
    // The & 255 operation is equivalent to modulo 256 but faster
    int xi = cell_left & 255;
    int yi = cell_top & 255;

    // Use the permutation table to get pseudo-random gradient indices at the four corners of the cell
    int grad_index_top_left     = lookUpTable[lookUpTable[xi]     + yi] & 7;
    int grad_index_top_right    = lookUpTable[lookUpTable[xi + 1] + yi] & 7;
    int grad_index_bottom_left  = lookUpTable[lookUpTable[xi]     + yi + 1] & 7;
    int grad_index_bottom_right = lookUpTable[lookUpTable[xi + 1] + yi + 1] & 7;

    // Select the gradient vectors corresponding to these indices
    const Vector2D grad_top_left     = d_gradients[grad_index_top_left];
    const Vector2D grad_top_right    = d_gradients[grad_index_top_right];
    const Vector2D grad_bottom_left  = d_gradients[grad_index_bottom_left];
    const Vector2D grad_bottom_right = d_gradients[grad_index_bottom_right];

    // Compute vectors from each corner of the cell to the pixel's location
    Vector2D dist_to_top_left     (local_x,        local_y);
    Vector2D dist_to_top_right    (local_x - 1.0f, local_y);
    Vector2D dist_to_bottom_left  (local_x,        local_y - 1.0f);
    Vector2D dist_to_bottom_right (local_x - 1.0f, local_y - 1.0f);

    // Calculate the dot product between distance vectors and gradient vectors
    // This gives the influence (contribution) of each corner on the final noise value
    float influence_top_left     = grad_top_left.dot(dist_to_top_left);
    float influence_top_right    = grad_top_right.dot(dist_to_top_right);
    float influence_bottom_left  = grad_bottom_left.dot(dist_to_bottom_left);
    float influence_bottom_right = grad_bottom_right.dot(dist_to_bottom_right);

    // Interpolate the influences along the x-axis using a fade function for smoothness
    float u = fade(local_x);
    float v = fade(local_y);
    
    float interp_top    = lerp(influence_top_left, influence_top_right, u);
    float interp_bottom = lerp(influence_bottom_left, influence_bottom_right, u);
    
    // Interpolate the top and bottom results along the y-axis to get the final Perlin noise value
    float pixel_noise_value = lerp(interp_top, interp_bottom, v);

    // Accumulate the computed noise into the output array, scaling by the amplitude
    accumulator[y * width + x] += pixel_noise_value * amplitude;
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
    int lacunarity = opts.lacunarity;
    float persistence = opts.persistence;
    int offset_x = opts.offset_x;
    int offset_y = opts.offset_y;
    bool no_outputs = opts.no_outputs;
    bool verbose = opts.verbose;
    bool benchmark = opts.benchmark;

    // output info
    std::string output_filename = opts.output_filename;
    std::string output_format = opts.format;

    /* randomize from seed */
    srand(seed);

    /* build the permutation tables */
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
    if (benchmark) {
        wall_start = std::chrono::high_resolution_clock::now();
        cpu_start = std::clock();
    }

    /* setting CUDA DEVICE */
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    
    /* allocate device memory */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    /* float accumulation buffer (needed for octaves) */
    // host accumulator
    std::vector<float> accumulator(width * height, 0.0f);
    
    // device accumulator
    size_t buffer_size = width * height * sizeof(float);
    float* d_accumulator;
    CHECK(cudaMalloc((void**)&d_accumulator, buffer_size));

    // initialize device accumulator to zero
    CHECK(cudaMemset(d_accumulator, 0, buffer_size));
    
    /* copy lookUpTable to device */
    int* d_lookUpTable;
    size_t lookUpTable_size = lookUpTable.size() * sizeof(int);
    CHECK(cudaMalloc((void**)&d_lookUpTable, lookUpTable_size));
    CHECK(cudaMemcpy(d_lookUpTable, lookUpTable.data(), lookUpTable_size, cudaMemcpyHostToDevice));

    /* CUDA Events for Kernel Timing */
    cudaEvent_t start_kernel, stop_kernel;
    CHECK(cudaEventCreate(&start_kernel));
    CHECK(cudaEventCreate(&stop_kernel));

    /* Start Kernel Timer */
    CHECK(cudaEventRecord(start_kernel, 0));
    
    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;
    
    //AAAAAAAAAAAAAAAA
    Vector2D h_gradients[8] = {
        {1.0f,1.0f}, {-1.0f,1.0f}, {1.0f,-1.0f}, {-1.0f,-1.0f},
        {1.0f,0.0f}, {-1.0f,0.0f}, {0.0f,1.0f}, {0.0f,-1.0f}
    };
    CHECK(cudaMemcpyToSymbol(d_gradients, h_gradients, sizeof(Vector2D) * 8));
    //AAAAAAAAAAAAAAAA

    for (int o = 0; o < octaves; o++) {
        
        /* launch the kernel */
        gpu_generate_perlin_pixel<<<grid, block>>>(
            d_accumulator, 
            width, 
            height, 
            d_lookUpTable,
            frequency, 
            amplitude, 
            offset_x, 
            offset_y
        );

        // controls frequency growth
        frequency *= lacunarity;
        // controls amplitude decay
        amplitude *= persistence;
    }

    /* Stop Kernel Timer */
    CHECK(cudaEventRecord(stop_kernel, 0));
    
    /* wait for kernel to complete */
    CHECK(cudaDeviceSynchronize());

    /* Calculate Kernel Time */
    float kernel_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
    
    /* copy result from device to host */
    CHECK(cudaMemcpy(accumulator.data(), d_accumulator, buffer_size, cudaMemcpyDeviceToHost));
    
    /* free device memory */
    CHECK(cudaFree(d_accumulator));
    CHECK(cudaFree(d_lookUpTable));

    /* Destroy CUDA Events */
    CHECK(cudaEventDestroy(start_kernel));
    CHECK(cudaEventDestroy(stop_kernel));

    /* stop profiling timers and report */
    if (benchmark) {
        clock_t cpu_end = std::clock();
        auto wall_end = std::chrono::high_resolution_clock::now();

        double cpu_ticks = static_cast<double>(cpu_end - cpu_start);
        double cpu_seconds = cpu_ticks / static_cast<double>(CLOCKS_PER_SEC);
        double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_end - wall_start).count();

        size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        double ms_per_pixel = (num_pixels > 0) ? (wall_ms / (double)num_pixels) : 0.0;

        size_t lookUpTable_bytes = lookUpTable_size;
        size_t accumulator_bytes = buffer_size;
        size_t estimated_total_alloc = lookUpTable_bytes + accumulator_bytes;

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
    }

    if (verbose) {
        size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        double ms_per_pixel = (num_pixels > 0) ? (kernel_ms / (double)num_pixels) : 0.0;
        
        int chunks_count_x = (width + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;
        int chunks_count_y = (height + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;
        int chunks_total = chunks_count_x * chunks_count_y;
        
        size_t total_memory = buffer_size + lookUpTable_size;
        
        printf("\nProfiling:\n");
        printf("  kernel time       = %.3f ms\n", kernel_ms);
        printf("  time / pixel      = %.6f ms\n", ms_per_pixel);
        printf("  grid blocks       = %dx%d (total %d)\n", 
            (width + block.x - 1) / block.x, 
            (height + block.y - 1) / block.y,
            ((width + block.x - 1) / block.x) * ((height + block.y - 1) / block.y));
        printf("  block size        = %dx%d (threads: %d)\n", block.x, block.y, block.x * block.y);
        printf("  chunks            = %dx%d (total %d)\n", chunks_count_x, chunks_count_y, chunks_total);
        printf("  mem (approx)      = %zu bytes (lookUpTable %zu + accumulator %zu)\n", 
            total_memory, lookUpTable_size, buffer_size);
        printf("  device            = %s\n", deviceProp.name);
        printf("\n");
    }

    /* save the generated noise image */
    if (!no_outputs) {

        // create the output array
        std::vector<uint8_t> output(width * height, 0);
        for (int i = 0; i < width * height; i++) {
            // mapping the accumulator (-1, 1) to grayscaled pixels (0-255)
            float v = std::clamp(accumulator[i], -1.0f, 1.0f);
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
}
