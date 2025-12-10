/*
 * cuda__perlin_noise.cu - perlin noise: CUDA implementation
 *
 */

/*
 * Copyright 2025 Davide Chirichella, Filippo Giulietti
 * ... (License omitted for brevity) ...
 */

#include "utils_global.hpp"
#include "utils_cuda.hpp"
#include "perlin_noise.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // Added for IntelliSense
#include <stdio.h>
#include <curand_kernel.h>
#include <algorithm>
#include <numeric> // Added for std::iota
#include <random>  // Added for std::shuffle
#include <vector>
#include <cmath>
#include <chrono>

/* device variables */
#define BLOCK_SIZE 32

/**
 * @brief Kernel to generate Perlin noise for each pixel
 * Improvements: 
 * - Uses __restrict__ for compiler optimization (caching).
 * - Uses Permutation table for O(1) gradient lookup.
 * - Uses floorf (intrinsic) instead of std::floor.
 */
__global__ void gpu_generate_perlin_pixel(
    float* __restrict__ buffer,      // Output buffer
    const int* __restrict__ perm,    // Permutation table (512 ints)
    const Vector2D* __restrict__ grads, // Gradient vectors (256 vectors)
    int image_width,
    int image_height,
    float frequency,
    float amplitude,
    int offset_x,
    int offset_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    // 1. Calculate coordinate in noise space
    // Adding 0.5f ensures we sample center of pixel, not top-left corner
    float fx = ((float)(x + offset_x) / (float)image_width) * frequency;
    float fy = ((float)(y + offset_y) / (float)image_height) * frequency;

    // 2. Determine grid cell coordinates
    // Use floorf (CUDA intrinsic) for speed
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);

    // 3. Calculate local position in unit square (0.0 to 1.0)
    float sx = fx - (float)x0;
    float sy = fy - (float)y0;

    // 4. Hash coordinates using Permutation Table
    // We mask with 255 (0xFF) to wrap the index. 
    // This replaces the generic 'chunk' modulo logic with standard Perlin wrapping.
    int X = x0 & 255;
    int Y = y0 & 255;

    // Retrieve gradient indices from permutation table
    // The table is size 512, so [X+1] is safe because X is 0-255
    int gi00 = perm[perm[X] + Y] & 255;
    int gi10 = perm[perm[X + 1] + Y] & 255;
    int gi01 = perm[perm[X] + Y + 1] & 255;
    int gi11 = perm[perm[X + 1] + Y + 1] & 255;

    // 5. Fetch Gradients
    const Vector2D& g00 = grads[gi00];
    const Vector2D& g10 = grads[gi10];
    const Vector2D& g01 = grads[gi01];
    const Vector2D& g11 = grads[gi11];

    // 6. Calculate Dot Products
    // Distance vectors
    Vector2D d00(sx, sy);
    Vector2D d10(sx - 1.0f, sy);
    Vector2D d01(sx, sy - 1.0f);
    Vector2D d11(sx - 1.0f, sy - 1.0f);

    float dot00 = g00.dot(d00);
    float dot10 = g10.dot(d10);
    float dot01 = g01.dot(d01);
    float dot11 = g11.dot(d11);

    // 7. Interpolate (Fade + Lerp)
    float u = fade(sx); // fade is defined in your utils header
    float v = fade(sy);

    float nx0 = lerp(dot00, dot10, u);
    float nx1 = lerp(dot01, dot11, u);
    float value = lerp(nx0, nx1, v);

    // 8. Accumulate
    // Calculate 1D index only once
    int idx = y * image_width + x;
    buffer[idx] += value * amplitude;
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

    std::string output_filename = opts.output_filename;
    std::string output_format = opts.format;

    /* start profiling timers */
    auto start_total = std::chrono::high_resolution_clock::now();

    /* setting CUDA DEVICE */
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    /* allocate device memory for output */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    /* float accumulation buffer */
    std::vector<float> accumulator(width * height, 0.0f);
    size_t buffer_size = width * height * sizeof(float);
    float* d_accumulator;
    CHECK(cudaMalloc((void**)&d_accumulator, buffer_size));
    CHECK(cudaMemset(d_accumulator, 0, buffer_size));

    /* --- PERMUTATION TABLE GENERATION --- */
    // Standard Perlin requires a 0-255 array, shuffled, then duplicated to size 512
    // to avoid buffer overflow logic in the kernel.
    std::vector<int> p(256);
    std::iota(p.begin(), p.end(), 0); // Fill 0, 1, ..., 255

    // Seed the generator
    std::mt19937 generator(seed);
    std::shuffle(p.begin(), p.end(), generator);

    // Create the doubled permutation table (size 512)
    std::vector<int> h_perm(512);
    for (int i = 0; i < 512; i++) {
        h_perm[i] = p[i % 256];
    }

    int* d_perm;
    CHECK(cudaMalloc((void**)&d_perm, 512 * sizeof(int)));
    CHECK(cudaMemcpy(d_perm, h_perm.data(), 512 * sizeof(int), cudaMemcpyHostToDevice));

    /* --- GRADIENT VECTOR GENERATION --- */
    // We only need 256 random unit vectors now, not a vector for every chunk/pixel
    int gradient_count = 256;
    std::vector<Vector2D> h_gradients(gradient_count);

    // Using a simple host RNG for gradients (srand logic from original code is fine here)
    srand(seed); 
    for (int i = 0; i < gradient_count; i++) {
        float rx = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        float ry = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_gradients[i] = Vector2D(rx, ry).normalize();
    }

    Vector2D* d_gradients;
    size_t gradients_buffer_size = gradient_count * sizeof(Vector2D);
    CHECK(cudaMalloc((void**)&d_gradients, gradients_buffer_size));
    CHECK(cudaMemcpy(d_gradients, h_gradients.data(), gradients_buffer_size, cudaMemcpyHostToDevice));


    /* CUDA Events for Kernel Timing */
    cudaEvent_t start_kernel, stop_kernel;
    CHECK(cudaEventCreate(&start_kernel));
    CHECK(cudaEventCreate(&stop_kernel));

    /* Start Kernel Timer */
    CHECK(cudaEventRecord(start_kernel, 0));

    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    for (int o = 0; o < octaves; o++) {
        gpu_generate_perlin_pixel << <grid, block >> > (
            d_accumulator, 
            d_perm, 
            d_gradients, 
            width, 
            height, 
            frequency, 
            amplitude, 
            offset_x, 
            offset_y
        );
        
        // Error check after launch
        // CHECK(cudaGetLastError()); // Optional: enable for debugging

        frequency *= lacunarity;
        amplitude *= persistence;
    }

    /* Stop Kernel Timer */
    CHECK(cudaEventRecord(stop_kernel, 0));
    CHECK(cudaDeviceSynchronize());

    float kernel_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));

    /* copy and free */
    CHECK(cudaMemcpy(accumulator.data(), d_accumulator, buffer_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_accumulator));
    CHECK(cudaFree(d_gradients));
    CHECK(cudaFree(d_perm));

    CHECK(cudaEventDestroy(start_kernel));
    CHECK(cudaEventDestroy(stop_kernel));

    /* Convert to Output (0-255) */
    unsigned int channels = 1;
    std::vector<unsigned char> output(width * height * channels, 0);
    
    // Normalizing loop (CPU side)
    // Note: If width*height is massive, this loop could also be a kernel, 
    // but keeping it here as requested to not change flow too much.
    for (size_t i = 0; i < accumulator.size(); i++) {
        float v = accumulator[i];
        v = std::clamp(v, -1.0f, 1.0f);
        output[i] = static_cast<unsigned char>((v + 1.0f) * 0.5f * 255.0f);
    }

    /* Stop Total Time Measurement */
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_s = end_total - start_total;

    if (verbose) {
        printf("\nProfiling:\n");
        printf("  kernel Time       = %.3f ms\n", kernel_ms);
        printf("  total Time        = %.3f s\n", total_s.count());
    }

    if (!no_outputs) {
        save_output(output, width, height, channels, output_filename, output_format);
    }
}