/*
 * cpp__perlin_noise.cpp - perlin noise: C++ implementation
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
#include "utils_cpu.hpp"

#include <algorithm>
#include <inttypes.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <numeric> // Added for std::iota
#include <random>  // Added for std::default_random_engine

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32

/**
 * @brief Helper function to calculate dot product based on hash.
 * This replaces storing Vector2D objects. It picks a gradient vector
 * based on the hash value and returns the dot product with input (x,y).
 */
inline float grad(int hash, float x, float y) {
    // Convert low 4 bits of hash code into 12 gradient directions
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : ((h == 12 || h == 14) ? x : 0);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

/**
 * @brief Chunk: represents a square section of the noise map
 * */
struct Chunk {
    int chunk_x = 0;
    int chunk_y = 0;

    Chunk(int cx, int cy) : chunk_x(cx), chunk_y(cy) {}

    void generate_chunk_pixels(
        std::vector<float>& buffer, // reference to global float buffer
        int image_width,
        int image_height,
        const std::vector<int>& p, // CHANGED: Pass permutation table instead of gradients
        float frequency,
        float amplitude,
        int offset_x,
        int offset_y
    ) const {
        int start_x = chunk_x * CHUNK_SIDE_LENGTH;
        int start_y = chunk_y * CHUNK_SIDE_LENGTH;

        int end_x = std::min(start_x + CHUNK_SIDE_LENGTH, image_width);
        int end_y = std::min(start_y + CHUNK_SIDE_LENGTH, image_height);

        for (int y = start_y; y < end_y; y++) {
            for (int x = start_x; x < end_x; x++) {

                // normalized coordinates scaled by frequency
                float lerp_coeff = std::max(image_width, image_height);
                float fx = ((float)(x + offset_x) / (float)lerp_coeff) * frequency;
                float fy = ((float)(y + offset_y) / (float)lerp_coeff) * frequency;

                // Find unit grid cell containing point
                int X = (int)std::floor(fx) & 255;
                int Y = (int)std::floor(fy) & 255;

                // local coordinates within the cell (relative to top-left)
                float sx = fx - std::floor(fx);
                float sy = fy - std::floor(fy);

                // fade curves
                float u = fade(sx);
                float v = fade(sy);

                // Hash coordinates of the 4 square corners
                // This replaces the 2D array lookups
                int A  = p[X] + Y;
                int AA = p[A];
                int AB = p[A + 1];
                int B  = p[X + 1] + Y;
                int BA = p[B];
                int BB = p[B + 1];

                // Calculate dot products for the 4 corners
                // grad() handles the gradient vector selection implicitly
                float dot00 = grad(p[AA], sx, sy);         // Top-Left
                float dot10 = grad(p[BA], sx - 1, sy);     // Top-Right
                float dot01 = grad(p[AB], sx, sy - 1);     // Bottom-Left
                float dot11 = grad(p[BB], sx - 1, sy - 1); // Bottom-Right

                // bilinear interpolation
                float nx0 = lerp(dot00, dot10, u);
                float nx1 = lerp(dot01, dot11, u);
                float value = lerp(nx0, nx1, v);

                // multiply by amplitude for THIS octave
                buffer[y * image_width + x] += value * amplitude; 
            }
        }
    }
};

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

    // output info
    std::string output_filename = opts.output_filename;
    std::string output_format = opts.format;

    /* randomize from seed */
    srand(seed);

    /* start profiling timers */
    std::chrono::high_resolution_clock::time_point wall_start;
    clock_t cpu_start = 0;
    if (verbose) {
        wall_start = std::chrono::high_resolution_clock::now();
        cpu_start = std::clock();
    }

    /* float accumulation buffer (needed for octaves) */
    std::vector<float> accumulator(width * height, 0.0f);

    /* calculate chunk grid */
    int chunks_count_x = (width  + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;
    int chunks_count_y = (height + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;

    /* Generate Permutation Table */
    // Standard Perlin permutation table (0-255)
    std::vector<int> p(512); 
    
    // Fill first 256 with 0-255
    std::iota(p.begin(), p.begin() + 256, 0);

    // Shuffle using the seed
    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.begin() + 256, engine);

    // Duplicate the permutation to avoid buffer overflow
    for(int i = 0; i < 256; i++) {
        p[256 + i] = p[i];
    }

    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    for (int o = 0; o < octaves; o++) {

        // generate noise for this octave using the existing chunk pipeline
        for (int cy = 0; cy < chunks_count_y; cy++) {
            for (int cx = 0; cx < chunks_count_x; cx++) {
                Chunk chunk(cx, cy);
                
                chunk.generate_chunk_pixels(
                    accumulator,
                    width,
                    height,
                    p, // Pass the permutation vector
                    frequency, // Removed chunk counts from args as they aren't needed for hashing
                    amplitude,
                    offset_x,
                    offset_y
                );
            }
        }

        // prepare next octave (standard FBM rules)
        frequency *= lacunarity;   // controls frequency growth
        amplitude *= persistence;  // controls amplitude decay
    }

    /* stop profiling timers and report */
    if (verbose) {
        clock_t cpu_end = std::clock();
        auto wall_end = std::chrono::high_resolution_clock::now();

        double cpu_ticks = static_cast<double>(cpu_end - cpu_start);
        double cpu_seconds = cpu_ticks / static_cast<double>(CLOCKS_PER_SEC);
        double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_end - wall_start).count();

        size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        double ms_per_pixel = (num_pixels > 0) ? (wall_ms / (double)num_pixels) : 0.0;

        size_t perm_bytes = p.size() * sizeof(int);
        size_t accumulator_bytes = accumulator.size() * sizeof(float);
        size_t estimated_total_alloc = perm_bytes + accumulator_bytes;

        printf("\nProfiling:\n");
        printf("  wall time        = %.3f ms\n", wall_ms);
        printf("  cpu time         = %.6f s (clock ticks = %.0f)\n", cpu_seconds, cpu_ticks);
        printf("  time / pixel     = %.6f ms\n", ms_per_pixel);
        printf("  chunks           = %dx%d (total %d)\n", chunks_count_x, chunks_count_y, chunks_count_x * chunks_count_y);
        printf("  mem (approx)     = %zu bytes (permutation %zu + accumulator %zu)\n",
               estimated_total_alloc, perm_bytes, accumulator_bytes);
    }

    /* convert accumulator to final 0-255 output */
    unsigned int channels = 1;
    std::vector<unsigned char> output(width * height * channels, 0);

    for (int i = 0; i < width * height; i++) {

        // normalize fractal sum back to [-1,1]
        // Note: Perlin noise with sqrt(N/2) usually stays within range, 
        // but with multiple octaves, we might exceed [-1, 1].
        // Simple clamping is usually sufficient for visual output.
        float v = accumulator[i];
        v = std::clamp(v, -1.0f, 1.0f);

        output[i] = static_cast<unsigned char>((v + 1.0f) * 0.5f * 255.0f);
    }

    if (!no_outputs){
        save_output(
            output,
            width,
            height,
            channels,
            output_filename,
            output_format
        );
    }
}