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
#include "utils_global.hpp"

#include <algorithm>
#include <inttypes.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>


/* chunk variables */
#define CHUNK_SIDE_LENGTH 32


/**
 * @brief Smoothing function for Perlin noise
 * @param t 
 * @return float 
 */
static float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}



/** 
 * @brief Linear interpolation
 * @param a 
 * @param b 
 * @param t 
 * @return float 
 */
static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}



/**
 * @brief Simple 2D vector structure
 * 
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



// Declaring the global gradients' vectors
static const Vector2D gradients[] = {
    {1,1}, {-1,1}, {1,-1}, {-1,-1}, {1,0}, {-1,0}, {0,1}, {0,-1}
};



/**
 * @brief Chunk: represents a square section of the noise map
 */
struct Chunk {
    int chunk_x = 0;
    int chunk_y = 0;

    Chunk(int cx, int cy) : chunk_x(cx), chunk_y(cy) {}

    void generate_chunk_pixels(
        std::vector<float>& accumulator,
        int width,
        int height,
        const std::vector<int>& lookUpTable,
        float frequency,
        float amplitude,
        int offset_x,
        int offset_y
    ) const {

        // chunk's pixels ranges to iterate over
        int start_x = chunk_x * CHUNK_SIDE_LENGTH;
        int start_y = chunk_y * CHUNK_SIDE_LENGTH;
        int end_x = std::min(start_x + CHUNK_SIDE_LENGTH, width);
        int end_y = std::min(start_y + CHUNK_SIDE_LENGTH, height);

        /* iterate over chunk's pixels */
        for (int y = start_y; y < end_y; y++) {
            for (int x = start_x; x < end_x; x++) {

                // Get the pixel global coordinates with:
                // - offset
                // - frequency scaling (aspect ratio to 1:1)
                float max_dimension = std::max(width, height);
                float noise_x = ((float)(x + offset_x) / max_dimension) * frequency;
                float noise_y = ((float)(y + offset_y) / max_dimension) * frequency;

                // Determine the integer coordinates of the cell
                // (grid square) that contains this point
                int cell_left = (int)std::floor(noise_x);
                int cell_top = (int)std::floor(noise_y);
                int cell_right = cell_left + 1;
                int cell_bottom = cell_top + 1;

                // Get the pixel local coordinates inside the chunk 
                float local_x = noise_x - (float)cell_left;
                float local_y = noise_y - (float)cell_top;

                // Wrap cell coordinates to [0, 255] for indexing the permutation table
                // The & 255 operation is equivalent to modulo 256 but faster
                int xi = cell_left & 255;
                int yi = cell_top  & 255;

                // Use the permutation table to get pseudo-random gradient indices at the four corners of the cell
                int grad_index_top_left     = lookUpTable[ lookUpTable[xi]     + yi] & 7;
                int grad_index_top_right    = lookUpTable[ lookUpTable[xi + 1] + yi] & 7;
                int grad_index_bottom_left  = lookUpTable[ lookUpTable[xi]     + yi + 1] & 7;
                int grad_index_bottom_right = lookUpTable[ lookUpTable[xi + 1] + yi + 1] & 7;

                // Select the gradient vectors corresponding to these indices
                const Vector2D& grad_top_left     = gradients[grad_index_top_left];
                const Vector2D& grad_top_right    = gradients[grad_index_top_right];
                const Vector2D& grad_bottom_left  = gradients[grad_index_bottom_left];
                const Vector2D& grad_bottom_right = gradients[grad_index_bottom_right];

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
                float interp_top    = lerp(influence_top_left, influence_top_right, fade(local_x));
                float interp_bottom = lerp(influence_bottom_left, influence_bottom_right, fade(local_x));
                
                // Interpolate the top and bottom results along the y-axis to get the final Perlin noise value
                float pixel_noise_value = lerp(interp_top, interp_bottom, fade(local_y));

                // Accumulate the computed noise into the output array, scaling by the amplitude
                accumulator[y * width + x] += pixel_noise_value * amplitude;

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
    if (verbose) {
        wall_start = std::chrono::high_resolution_clock::now();
        cpu_start = std::clock();
    }

    /* calculate chunk grid with ceiling division */
    int chunks_count_x = (width + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;
    int chunks_count_y = (height + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;

    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    /* float accumulation accumulator (needed for octaves) */
    std::vector<float> accumulator(width * height, 0.0f);

    for (int o = 0; o < octaves; o++) {

        // generate noise for this octave using the existing chunk pipeline
        for (int cy = 0; cy < chunks_count_y; cy++) {
            for (int cx = 0; cx < chunks_count_x; cx++) {
                Chunk chunk(cx, cy);
                
                chunk.generate_chunk_pixels(
                    accumulator,
                    width,
                    height,
                    lookUpTable,
                    frequency,
                    amplitude,
                    offset_x,
                    offset_y
                );
            }
        }

        // controls frequency growth
        frequency *= lacunarity;
        // controls amplitude decay
        amplitude *= persistence;
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

        size_t accumulator_bytes = accumulator.size() * sizeof(float);
        size_t estimated_total_alloc = accumulator_bytes + lookUpTable.size();

        std::string csv_name = "cpp_v2_benchmark.csv";
        std::ifstream check_file(csv_name);
        bool exists = check_file.good();
        check_file.close();

        std::ofstream csv_file(csv_name, std::ios::app);
        if (csv_file.is_open()) {
            if (!exists) {
                csv_file << "timestamp,width,height,pixels,octaves,frequency,wall_ms,cpu_s,ms_per_pixel,mem_bytes\n";
            }
            std::time_t now = std::time(nullptr);
            csv_file << now << ","
                     << width << ","
                     << height << ","
                     << num_pixels << ","
                     << octaves << ","
                     << base_frequency << ","
                     << wall_ms << ","
                     << cpu_seconds << ","
                     << ms_per_pixel << ","
                     << estimated_total_alloc << "\n";
            
            csv_file.close();
        }

        // Report testuale (opzionale, mantenuto per comodità)
        printf("Profiling & Logging to %s complete.\n", csv_name.c_str());
        printf("  wall time: %.3f ms | ms/pixel: %.6f\n\n", wall_ms, ms_per_pixel);
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
}