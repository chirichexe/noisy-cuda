// ...existing code...

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

// Constant memory for lookup table and gradients
__constant__ int d_lookUpTable[512];
__constant__ Vector2D d_gradients[8];

/**
 * @brief CUDA kernel for Perlin noise generation
 */
__global__ void perlin_noise_kernel(
    float* accumulator,
    int width,
    int height,
    float frequency,
    float amplitude,
    int offset_x,
    int offset_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Get the pixel global coordinates with offset and frequency scaling
    float max_dimension = fmaxf((float)width, (float)height);
    float noise_x = ((float)(x + offset_x) / max_dimension) * frequency;
    float noise_y = ((float)(y + offset_y) / max_dimension) * frequency;

    // Determine the integer coordinates of the cell
    int cell_left = (int)floorf(noise_x);
    int cell_top = (int)floorf(noise_y);

    // Get the pixel local coordinates inside the chunk
    float local_x = noise_x - (float)cell_left;
    float local_y = noise_y - (float)cell_top;

    // Wrap cell coordinates to [0, 255]
    int xi = cell_left & 255;
    int yi = cell_top & 255;

    // Use the permutation table to get gradient indices
    int grad_index_top_left = d_lookUpTable[d_lookUpTable[xi] + yi] & 7;
    int grad_index_top_right = d_lookUpTable[d_lookUpTable[xi + 1] + yi] & 7;
    int grad_index_bottom_left = d_lookUpTable[d_lookUpTable[xi] + yi + 1] & 7;
    int grad_index_bottom_right = d_lookUpTable[d_lookUpTable[xi + 1] + yi + 1] & 7;

    // Select gradient vectors
    Vector2D grad_top_left = d_gradients[grad_index_top_left];
    Vector2D grad_top_right = d_gradients[grad_index_top_right];
    Vector2D grad_bottom_left = d_gradients[grad_index_bottom_left];
    Vector2D grad_bottom_right = d_gradients[grad_index_bottom_right];

    // Compute vectors from each corner
    Vector2D dist_to_top_left(local_x, local_y);
    Vector2D dist_to_top_right(local_x - 1.0f, local_y);
    Vector2D dist_to_bottom_left(local_x, local_y - 1.0f);
    Vector2D dist_to_bottom_right(local_x - 1.0f, local_y - 1.0f);

    // Calculate dot products
    float influence_top_left = grad_top_left.dot(dist_to_top_left);
    float influence_top_right = grad_top_right.dot(dist_to_top_right);
    float influence_bottom_left = grad_bottom_left.dot(dist_to_bottom_left);
    float influence_bottom_right = grad_bottom_right.dot(dist_to_bottom_right);

    // Interpolate along x-axis
    float interp_top = lerp(influence_top_left, influence_top_right, fade(local_x));
    float interp_bottom = lerp(influence_bottom_left, influence_bottom_right, fade(local_x));

    // Interpolate along y-axis
    float pixel_noise_value = lerp(interp_top, interp_bottom, fade(local_y));

    // Accumulate with atomic operation for thread safety
    int idx = y * width + x;
    atomicAdd(&accumulator[idx], pixel_noise_value * amplitude);
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

    if (benchmark) {
        CHECK(cudaEventRecord(cuda_start));
    }

    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    for (int o = 0; o < octaves; o++) {
        perlin_noise_kernel<<<gridSize, blockSize>>>(
            d_accumulator,
            width,
            height,
            frequency,
            amplitude,
            offset_x,
            offset_y
        );
        CHECK(cudaGetLastError());

        frequency *= lacunarity;
        amplitude *= persistence;
    }

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