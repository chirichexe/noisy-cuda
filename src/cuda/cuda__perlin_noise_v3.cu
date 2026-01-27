/*
 * cuda__perlin_noise_v2.cu - perlin noise: CUDA second implementation
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

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32
#define BLOCK_SIZE 16

#include <cmath>

/**
 * @brief Simple 2D vector structure
 * 
 */
struct Vector2D {
    float x = 0.0f;
    float y = 0.0f;

    Vector2D() = default;

    __host__ __device__ Vector2D(float x_, float y_) : x(x_), y(y_) {}

    __host__ __device__ Vector2D operator-(const Vector2D& other) const {
        return {x - other.x, y - other.y};
    }

    __host__ __device__ float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    __host__ __device__ float length() const {
        return std::sqrt(x * x + y * y);
    }

    __host__ __device__ Vector2D normalize() const {
        float len = length();
        return len > 0 ? Vector2D(x / len, y / len) : Vector2D(0, 0);
    }
};

// Constant memory for lookup table and gradients
__constant__ int d_lookUpTable[512];
__constant__ Vector2D d_gradients[8];

__device__ __forceinline__ float fade(float t) {
    // Regola di Horner: riduce il numero di moltiplicazioni
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

/**
 * @brief CUDA kernel for Perlin noise generation
 */
/**
 * @brief CUDA kernel con Shared Memory e __syncthreads()
 */
/**
 * @brief CUDA kernel con Kernel Fusion e Shared Memory
 */
__global__ void perlin_noise_fused_kernel(
    float* output,
    int width,
    int height,
    int octaves,
    float base_frequency,
    float base_amplitude,
    float lacunarity,
    float persistence,
    int offset_x,
    int offset_y
) {
    // 1. Caricamento collaborativo in Shared Memory (come prima)
    __shared__ int s_lookUpTable[512];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int i = tid; i < 512; i += threads_per_block) {
        s_lookUpTable[i] = d_lookUpTable[i];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float max_dimension = fmaxf((float)width, (float)height);
    
    // Variabili per l'accumulo locale (registri)
    float total_noise = 0.0f;
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    // 2. CICLO OTTAVE SPOSTATO DENTRO IL KERNEL
    for (int o = 0; o < octaves; o++) {
        float noise_x = ((float)(x + offset_x) / max_dimension) * frequency;
        float noise_y = ((float)(y + offset_y) / max_dimension) * frequency;

        int cell_left = __float2int_rd(noise_x);
        int cell_top  = __float2int_rd(noise_y);

        float local_x = noise_x - (float)cell_left;
        float local_y = noise_y - (float)cell_top;

        int xi = cell_left & 255;
        int yi = cell_top  & 255;

        // Lookup degli indici dei gradienti
        int gi00 = s_lookUpTable[s_lookUpTable[xi] + yi] & 7;
        int gi01 = s_lookUpTable[s_lookUpTable[xi] + yi + 1] & 7;
        int gi10 = s_lookUpTable[s_lookUpTable[xi + 1] + yi] & 7;
        int gi11 = s_lookUpTable[s_lookUpTable[xi + 1] + yi + 1] & 7;

        // Calcolo influenze (utilizzando d_gradients in memoria costante)
        float dot00 = d_gradients[gi00].x * local_x + d_gradients[gi00].y * local_y;
        float dot10 = d_gradients[gi10].x * (local_x - 1.0f) + d_gradients[gi10].y * local_y;
        float dot01 = d_gradients[gi01].x * local_x + d_gradients[gi01].y * (local_y - 1.0f);
        float dot11 = d_gradients[gi11].x * (local_x - 1.0f) + d_gradients[gi11].y * (local_y - 1.0f);

        float u = fade(local_x);
        float v = fade(local_y);

        float interp_x1 = lerp(dot00, dot10, u);
        float interp_x2 = lerp(dot01, dot11, u);
        
        // Accumulo il rumore dell'ottava corrente
        total_noise += lerp(interp_x1, interp_x2, v) * amplitude;

        // Aggiornamento parametri per la prossima ottava
        frequency *= lacunarity;
        amplitude *= persistence;
    }

    // 3. SCRITTURA UNICA nella memoria globale
    int idx = y * width + x;
    output[idx] = total_noise;
}

void generate_perlin_noise(const Options& opts) {
    /* initialize parameters */
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

    std::string output_filename = opts.output_filename;
    std::string output_format = opts.format;

    /* randomize from seed */
    srand(seed);

    // Build permutation table
    std::vector<int> lookUpTable(512);
    for (int i = 0; i < 256; i++) lookUpTable[i] = i;

    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(lookUpTable[i], lookUpTable[j]);
    }
    for (int i = 0; i < 256; i++) lookUpTable[256 + i] = lookUpTable[i];

    // Define gradients
    const Vector2D gradients[] = {
        {1,1}, {-1,1}, {1,-1}, {-1,-1}, {1,0}, {-1,0}, {0,1}, {0,-1}
    };

    // Copy lookup table and gradients to constant memory
    CHECK(cudaMemcpyToSymbol(d_lookUpTable, lookUpTable.data(), 512 * sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_gradients, gradients, 8 * sizeof(Vector2D)));

    /* start profiling timers */
    std::chrono::high_resolution_clock::time_point wall_start;
    clock_t cpu_start = 0;
    cudaEvent_t cuda_start, cuda_stop;
    
    if (benchmark) {
        wall_start = std::chrono::high_resolution_clock::now();
        cpu_start = std::clock();
        CHECK(cudaEventCreate(&cuda_start));
        CHECK(cudaEventCreate(&cuda_stop));
    }

    // Allocate device memory
    float* d_accumulator;
    size_t accumulator_size = width * height * sizeof(float);
    CHECK(cudaMalloc(&d_accumulator, accumulator_size));
    CHECK(cudaMemset(d_accumulator, 0, accumulator_size));

    // Configure kernel launch parameters
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

                  
    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;
                  
    if (benchmark) {
        CHECK(cudaEventRecord(cuda_start));
    }
    // Un'unica chiamata al kernel per tutte le ottave
    perlin_noise_fused_kernel<<<gridSize, blockSize>>>(
        d_accumulator,
        width,
        height,
        octaves,
        base_frequency,
        base_amplitude,
        (float)lacunarity,
        persistence,
        offset_x,
        offset_y
    );
    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());

    if (benchmark) {
        CHECK(cudaEventRecord(cuda_stop));
        CHECK(cudaEventSynchronize(cuda_stop));
    }

    // Copy result back to host
    std::vector<float> accumulator(width * height);
    CHECK(cudaMemcpy(accumulator.data(), d_accumulator, accumulator_size, cudaMemcpyDeviceToHost));

    /* stop profiling timers and report */
    if (benchmark) {
        clock_t cpu_end = std::clock();
        auto wall_end = std::chrono::high_resolution_clock::now();

        float cuda_ms = 0;
        CHECK(cudaEventElapsedTime(&cuda_ms, cuda_start, cuda_stop));

        double cpu_ticks = static_cast<double>(cpu_end - cpu_start);
        double cpu_seconds = cpu_ticks / static_cast<double>(CLOCKS_PER_SEC);
        double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_end - wall_start).count();

        size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        double ms_per_pixel = (num_pixels > 0) ? (wall_ms / (double)num_pixels) : 0.0;

        size_t estimated_total_alloc = accumulator_size;

        std::time_t now = std::time(nullptr);
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
    if (!no_outputs) {
        std::vector<uint8_t> output(width * height, 0);
        for (int i = 0; i < width * height; i++) {
            float v = std::clamp(accumulator[i], -1.0f, 1.0f);
            output[i] = static_cast<uint8_t>((v + 1.0f) * 0.5f * 255.0f);
        }

        save_output(
            output,
            width,
            height,
            1,
            output_filename,
            output_format
        );
    }

    // Cleanup
    CHECK(cudaFree(d_accumulator));
}