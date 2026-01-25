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

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32

static const Vector2D g[] = {
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
        int width, int height,
        const std::vector<int>& p,         // Nuova lookup table
        //const std::vector<Vector2D>& g,    // Nuovi gradienti fissi
        float frequency, float amplitude,
        int offset_x, int offset_y
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

                // Get the pixel (seen as cell) corners' coordinates
                int cell_left = (int)std::floor(noise_x);
                int cell_top = (int)std::floor(noise_y);
                int cell_right = cell_left + 1;
                int cell_bottom = cell_top + 1;

                // Get the pixel local coordinates inside the chunk 
                float local_x = noise_x - (float)cell_left;
                float local_y = noise_y - (float)cell_top;

                int xi = cell_left & 255;
                int yi = cell_top  & 255;

                // Otteniamo gli indici per i 4 angoli dalla tabella di permutazione
                int gi_tl = p[p[xi] + yi] & 7;           
                int gi_tr = p[p[xi + 1] + yi] & 7;       
                int gi_bl = p[p[xi] + yi + 1] & 7;       
                int gi_br = p[p[xi + 1] + yi + 1] & 7;

                // Selezioniamo i vettori gradiente
                const Vector2D& grad_top_left     = g[gi_tl];
                const Vector2D& grad_top_right    = g[gi_tr];
                const Vector2D& grad_bottom_left  = g[gi_bl];
                const Vector2D& grad_bottom_right = g[gi_br];

                // Calculate distance vectors between each gradient and pixel coords
                Vector2D dist_to_top_left     (local_x,        local_y);
                Vector2D dist_to_top_right    (local_x - 1.0f, local_y);
                Vector2D dist_to_bottom_left  (local_x,        local_y - 1.0f);
                Vector2D dist_to_bottom_right (local_x - 1.0f, local_y - 1.0f);

                // Calculate gradients influences with dot products
                float influence_top_left     = grad_top_left.dot(dist_to_top_left);
                float influence_top_right    = grad_top_right.dot(dist_to_top_right);
                float influence_bottom_left  = grad_bottom_left.dot(dist_to_bottom_left);
                float influence_bottom_right = grad_bottom_right.dot(dist_to_bottom_right);

                // Interpolate the corners with the influence (in pairs)
                float lerp_top    = lerp(influence_top_left, influence_top_right, fade(local_x));
                float lerp_bottom = lerp(influence_bottom_left, influence_bottom_right, fade(local_x));
                
                // Get the final pixel value lerping top and bottom
                float pixel_noise_value = lerp(lerp_top, lerp_bottom, fade(local_y));

                // add the value to the global accumulator
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

    // 1. Crea la Permutation Table basata sul seed
    std::vector<int> p(512);
    for (int i = 0; i < 256; i++) p[i] = i;

    // Mischia usando Fisher-Yates
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(p[i], p[j]);
    }
    // Raddoppia per evitare overflow durante l'indicizzazione
    for (int i = 0; i < 256; i++) p[256 + i] = p[i];

    // 2. Definisci i gradienti costanti (8 o 12 direzioni standard)
    // Questo sostituisce la vecchia matrice 'gradients' casuale


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
                    accumulator, width, height,
                    p, //g, // <--- Passa p e g invece di gradients
                    frequency, amplitude, offset_x, offset_y
                );
            }
        }

        // prepare next octave (standard FBM rules)
        // https://medium.com/@logan.margo314/procedural-generation-using-fractional-brownian-motion-b35b7231309f
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

        size_t accumulator_bytes = accumulator.size() * sizeof(float);
        size_t estimated_total_alloc = accumulator_bytes + p.size(); //g.size();

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