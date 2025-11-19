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
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * Distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "perlin_noise.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <inttypes.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <iostream>

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32


/*
 * Vector2D - simple 2D vector structure
 */
struct Vector2D {
    float x = 0.0f;
    float y = 0.0f;

    Vector2D() = default;
    Vector2D(float x_, float y_) : x(x_), y(y_) {}

    Vector2D operator-(const Vector2D& other) const {
        return {x - other.x, y - other.y};
    }

    float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    float length() const {
        return std::sqrt(x * x + y * y);
    }

    Vector2D normalize() const {
        float len = length();
        return len > 0 ? Vector2D(x / len, y / len) : Vector2D(0, 0);
    }
};


/*
 * Fade function - smoother interpolation curve
 */
static float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

/*
 * Linear interpolation
 */
static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}


/*
 * Chunk - handles a sub-region of the image using global gradients
 */
struct Chunk {
    int chunk_x = 0;
    int chunk_y = 0;

    Chunk(int cx, int cy) : chunk_x(cx), chunk_y(cy) {}

    void generate_chunk_pixels(
        std::vector<float>& buffer, // reference to global float buffer
        int image_width,
        int image_height,
        const std::vector<std::vector<Vector2D>>& gradients,
        int chunks_count_x,
        int chunks_count_y,
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

                // integer grid cell
                int x0 = (int)std::floor(fx);
                int y0 = (int)std::floor(fy);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                // local coordinates within the cell
                float sx = fx - (float)x0;
                float sy = fy - (float)y0;

                // corner gradients (shared from global grid)
                const Vector2D& g00 = gradients[x0 % chunks_count_x][y0 % chunks_count_y];  // tl
                const Vector2D& g10 = gradients[x1 % chunks_count_x][y0 % chunks_count_y];  // tr
                const Vector2D& g01 = gradients[x0 % chunks_count_x][y1 % chunks_count_y];  // bl
                const Vector2D& g11 = gradients[x1 % chunks_count_x][y1 % chunks_count_y];  // br

                // distance vectors
                Vector2D d00(sx,     sy);
                Vector2D d10(sx - 1, sy);
                Vector2D d01(sx,     sy - 1);
                Vector2D d11(sx - 1, sy - 1);

                // dot products
                float dot00 = g00.dot(d00);
                float dot10 = g10.dot(d10);
                float dot01 = g01.dot(d01);
                float dot11 = g11.dot(d11);

                // fade curves
                float u = fade(sx);
                float v = fade(sy);

                // bilinear interpolation
                float nx0 = lerp(dot00, dot10, u);
                float nx1 = lerp(dot01, dot11, u);
                float value = lerp(nx0, nx1, v);

                // multiply by amplitude for THIS octave
                buffer[y * image_width + x] += value * amplitude;  // ADDED: accumulate float noise
            }
        }
    }
};


/*
 * generate_perlin_noise - generates a 2D Perlin noise map using chunks
 */
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

    // output info
    std::string output_filename = opts.output_filename;

    /* randomize from seed */
    srand(seed);

    /* start profiling timers */
    std::chrono::high_resolution_clock::time_point wall_start;
    clock_t cpu_start = 0;
    if (opts.verbose) {
        wall_start = std::chrono::high_resolution_clock::now();
        cpu_start = std::clock();
    }

    /* float accumulation buffer (needed for octaves) */
    std::vector<float> accumulator(width * height, 0.0f);

    /* calculate chunk grid */
    int chunks_count_x = (width  + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;
    int chunks_count_y = (height + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;

    /* initialize gradient vectors */
    std::vector<std::vector<Vector2D>> gradients(chunks_count_x, std::vector<Vector2D>(chunks_count_y));

    for (int gx = 0; gx < chunks_count_x; gx++) {
        for (int gy = 0; gy < chunks_count_y; gy++) {
            float rx = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float ry = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            gradients[gx][gy] = Vector2D(rx, ry).normalize();
        }
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
                    gradients,
                    chunks_count_x,
                    chunks_count_y,
                    frequency,
                    amplitude,
                    offset_x,
                    offset_y
                );
            }
        }

        // prepare next octave (standard FBM rules)
        // https://medium.com/@logan.margo314/procedural-generation-using-fractional-brownian-motion-b35b7231309f
        frequency *= lacunarity;   // controls frequency growth
        amplitude *= persistence;  // controls amplitude decay
    }

    /* stop profiling timers and report */
    if (opts.verbose) {
        clock_t cpu_end = std::clock();
        auto wall_end = std::chrono::high_resolution_clock::now();

        double cpu_ticks = static_cast<double>(cpu_end - cpu_start);
        double cpu_seconds = cpu_ticks / static_cast<double>(CLOCKS_PER_SEC);
        double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_end - wall_start).count();

        size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        double ms_per_pixel = (num_pixels > 0) ? (wall_ms / (double)num_pixels) : 0.0;

        size_t gradients_bytes = (size_t)chunks_count_x * (size_t)chunks_count_y * sizeof(Vector2D);
        size_t accumulator_bytes = accumulator.size() * sizeof(float);
        size_t estimated_total_alloc = gradients_bytes + accumulator_bytes;

        printf("\nProfiling:\n");
        printf("  wall time        = %.3f ms\n", wall_ms);
        printf("  cpu time         = %.6f s (clock ticks = %.0f)\n", cpu_seconds, cpu_ticks);
        printf("  time / pixel     = %.6f ms\n", ms_per_pixel);
        printf("  chunks           = %dx%d (total %d)\n", chunks_count_x, chunks_count_y, chunks_count_x * chunks_count_y);
        printf("  mem (approx)     = %zu bytes (gradients %zu + accumulator %zu)\n",
               estimated_total_alloc, gradients_bytes, accumulator_bytes);
    }

    /* convert accumulator to final 0-255 output */
    unsigned int channels = 1;
    std::vector<unsigned char> output(width * height * channels, 0);

    for (int i = 0; i < width * height; i++) {

        // normalize fractal sum back to [-1,1]
        float v = accumulator[i];

        v = std::clamp(v, -1.0f, 1.0f);

        output[i] = static_cast<unsigned char>((v + 1.0f) * 0.5f * 255.0f);
    }

    /* save the generated noise image */
    stbi_write_png(output_filename.c_str(), width, height, channels, output.data(), width * channels);
    printf("\nOutput saved as \"%s\"\n", output_filename.c_str());
}