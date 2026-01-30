/*
 * cpp__perlin_noise_simd.cpp - perlin noise: C++ SIMD implementation
 *
 */

/*
 * Copyright 2025 Davide Chirichella, Filippo Giulietti
 * Optimized with SIMD AVX2 by Gemini (Reference: Stefano Mattoccia's slides)
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
#include <immintrin.h> // Intrinsics per x86 (SSE/AVX/AVX2)

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32

/**
 * @brief Smoothing function for Perlin noise - SIMD Version
 * Utilizza la formula: t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f)
 */
inline __m256 fade_ps(__m256 t) {
    __m256 six = _mm256_set1_ps(6.0f);
    __m256 fifteen = _mm256_set1_ps(-15.0f);
    __m256 ten = _mm256_set1_ps(10.0f);

    // Calcolo polinomiale parallelo
    __m256 res = _mm256_mul_ps(t, six);              // t * 6
    res = _mm256_add_ps(res, fifteen);               // t * 6 - 15
    res = _mm256_mul_ps(t, res);                     // t * (t * 6 - 15)
    res = _mm256_add_ps(res, ten);                   // t * (t * 6 - 15) + 10
    
    __m256 t3 = _mm256_mul_ps(t, _mm256_mul_ps(t, t)); // t^3
    return _mm256_mul_ps(t3, res);
}

/** * @brief Linear interpolation - SIMD Version
 * Formula: a + t * (b - a)
 */
inline __m256 lerp_ps(__m256 a, __m256 b, __m256 t) {
    return _mm256_add_ps(a, _mm256_mul_ps(t, _mm256_sub_ps(b, a)));
}

// Gradienti separati per caricamento vettoriale (uso di array invece di struct Vector2D)
static const float gradients_x[] = {1, -1, 1, -1, 1, -1, 0, 0};
static const float gradients_y[] = {1, 1, -1, -1, 0, 0, 1, -1};

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

        int start_x = chunk_x * CHUNK_SIDE_LENGTH;
        int start_y = chunk_y * CHUNK_SIDE_LENGTH;
        int end_y = std::min(start_y + CHUNK_SIDE_LENGTH, height);
        int end_x = std::min(start_x + CHUNK_SIDE_LENGTH, width);

        float max_dimension = (float)std::max(width, height);
        __m256 v_freq = _mm256_set1_ps(frequency);
        __m256 v_amp = _mm256_set1_ps(amplitude);
        __m256 v_inv_max_dim = _mm256_set1_ps(1.0f / max_dimension);
        __m256 v_one = _mm256_set1_ps(1.0f);
        
        // Maschere per operazioni bitwise (come indicato nelle slide)
        __m256i v_mask255 = _mm256_set1_epi32(255);
        __m256i v_mask7 = _mm256_set1_epi32(7);

        const int* lut_ptr = lookUpTable.data();

        /* iterate over chunk's pixels */
        for (int y = start_y; y < end_y; y++) {
            
            // Calcolo coordinata Y costante per l'intera riga
            float noise_y_val = ((float)(y + offset_y) / max_dimension) * frequency;
            __m256 v_noise_y = _mm256_set1_ps(noise_y_val);
            __m256 v_cell_top = _mm256_floor_ps(v_noise_y);
            __m256 v_local_y = _mm256_sub_ps(v_noise_y, v_cell_top);
            __m256i v_yi = _mm256_and_si256(_mm256_cvtps_epi32(v_cell_top), v_mask255);
            __m256i v_yi_plus_1 = _mm256_and_si256(_mm256_add_epi32(v_yi, _mm256_set1_epi32(1)), v_mask255);

            // SIMD loop: processa 8 pixel in parallelo lungo l'asse X
            for (int x = start_x; x <= end_x - 8; x += 8) {

                // Get the pixel global coordinates
                __m256 v_x_indices = _mm256_set_ps(x+7, x+6, x+5, x+4, x+3, x+2, x+1, x+0);
                __m256 v_noise_x = _mm256_mul_ps(_mm256_mul_ps(_mm256_add_ps(v_x_indices, _mm256_set1_ps(offset_x)), v_inv_max_dim), v_freq);

                // Determine the integer coordinates of the cell
                __m256 v_cell_left = _mm256_floor_ps(v_noise_x);
                __m256 v_local_x = _mm256_sub_ps(v_noise_x, v_cell_left);
                __m256i v_xi = _mm256_and_si256(_mm256_cvtps_epi32(v_cell_left), v_mask255);
                __m256i v_xi_plus_1 = _mm256_and_si256(_mm256_add_epi32(v_xi, _mm256_set1_epi32(1)), v_mask255);

                // Permutation Table Lookup con Istruzione GATHER (AVX2)
                __m256i p_xi  = _mm256_i32gather_epi32(lut_ptr, v_xi, 4);
                __m256i p_xi1 = _mm256_i32gather_epi32(lut_ptr, v_xi_plus_1, 4);

                // Indici dei gradienti ai 4 angoli
                __m256i gi_tl = _mm256_and_si256(_mm256_i32gather_epi32(lut_ptr, _mm256_add_epi32(p_xi, v_yi), 4), v_mask7);
                __m256i gi_tr = _mm256_and_si256(_mm256_i32gather_epi32(lut_ptr, _mm256_add_epi32(p_xi1, v_yi), 4), v_mask7);
                __m256i gi_bl = _mm256_and_si256(_mm256_i32gather_epi32(lut_ptr, _mm256_add_epi32(p_xi, v_yi_plus_1), 4), v_mask7);
                __m256i gi_br = _mm256_and_si256(_mm256_i32gather_epi32(lut_ptr, _mm256_add_epi32(p_xi1, v_yi_plus_1), 4), v_mask7);

                // Caricamento componenti gradienti (X e Y) via gather
                __m256 g_tl_x = _mm256_i32gather_ps(gradients_x, gi_tl, 4);
                __m256 g_tl_y = _mm256_i32gather_ps(gradients_y, gi_tl, 4);
                __m256 g_tr_x = _mm256_i32gather_ps(gradients_x, gi_tr, 4);
                __m256 g_tr_y = _mm256_i32gather_ps(gradients_y, gi_tr, 4);
                __m256 g_bl_x = _mm256_i32gather_ps(gradients_x, gi_bl, 4);
                __m256 g_bl_y = _mm256_i32gather_ps(gradients_y, gi_bl, 4);
                __m256 g_br_x = _mm256_i32gather_ps(gradients_x, gi_br, 4);
                __m256 g_br_y = _mm256_i32gather_ps(gradients_y, gi_br, 4);

                // Calcolo Dot Products: influenze dei 4 angoli
                __m256 infl_tl = _mm256_add_ps(_mm256_mul_ps(g_tl_x, v_local_x), _mm256_mul_ps(g_tl_y, v_local_y));
                __m256 infl_tr = _mm256_add_ps(_mm256_mul_ps(g_tr_x, _mm256_sub_ps(v_local_x, v_one)), _mm256_mul_ps(g_tr_y, v_local_y));
                __m256 infl_bl = _mm256_add_ps(_mm256_mul_ps(g_bl_x, v_local_x), _mm256_mul_ps(g_bl_y, _mm256_sub_ps(v_local_y, v_one)));
                __m256 infl_br = _mm256_add_ps(_mm256_mul_ps(g_br_x, _mm256_sub_ps(v_local_x, v_one)), _mm256_mul_ps(g_br_y, _mm256_sub_ps(v_local_y, v_one)));

                // Interpolazione con funzioni di fade
                __m256 fade_x = fade_ps(v_local_x);
                __m256 interp_top    = lerp_ps(infl_tl, infl_tr, fade_x);
                __m256 interp_bottom = lerp_ps(infl_bl, infl_br, fade_x);
                __m256 pixel_noise_v = _mm256_mul_ps(lerp_ps(interp_top, interp_bottom, fade_ps(v_local_y)), v_amp);

                // Accumulo dei risultati: caricamento non allineato (loadu)
                float* target_addr = &accumulator[y * width + x];
                __m256 current_acc = _mm256_loadu_ps(target_addr);
                _mm256_storeu_ps(target_addr, _mm256_add_ps(current_acc, pixel_noise_v));
            }

            // Fallback scalare per i pixel rimanenti (se width non è multiplo di 8)
            for (int x = start_x + ((end_x - start_x) / 8) * 8; x < end_x; x++) {
                float noise_x = ((float)(x + offset_x) / max_dimension) * frequency;
                int cell_left = (int)std::floor(noise_x);
                int cell_top = (int)std::floor(noise_y_val);
                float local_x = noise_x - (float)cell_left;
                float local_y = noise_y_val - (float)cell_top;
                int xi = cell_left & 255;
                int yi = cell_top & 255;
                int gi_tl = lookUpTable[lookUpTable[xi] + yi] & 7;
                int gi_tr = lookUpTable[lookUpTable[xi + 1] + yi] & 7;
                int gi_bl = lookUpTable[lookUpTable[xi] + yi + 1] & 7;
                int gi_br = lookUpTable[lookUpTable[xi + 1] + yi + 1] & 7;
                auto fade_s = [](float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); };
                auto lerp_s = [](float a, float b, float t) { return a + t * (b - a); };
                float dot_tl = gradients_x[gi_tl]*local_x + gradients_y[gi_tl]*local_y;
                float dot_tr = gradients_x[gi_tr]*(local_x-1.0f) + gradients_y[gi_tr]*local_y;
                float dot_bl = gradients_x[gi_bl]*local_x + gradients_y[gi_bl]*(local_y-1.0f);
                float dot_br = gradients_x[gi_br]*(local_x-1.0f) + gradients_y[gi_br]*(local_y-1.0f);
                float it = lerp_s(dot_tl, dot_tr, fade_s(local_x));
                float ib = lerp_s(dot_bl, dot_br, fade_s(local_x));
                accumulator[y * width + x] += lerp_s(it, ib, fade_s(local_y)) * amplitude;
            }
        }
    }
};

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

    /* start profiling timers - Utilizzo di __rdtsc() come suggerito nelle slide */
    uint64_t clock_start = 0;
    std::chrono::high_resolution_clock::time_point wall_start;
    if (benchmark) {
        wall_start = std::chrono::high_resolution_clock::now();
        clock_start = __rdtsc();
    }

    int chunks_count_x = (width + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;
    int chunks_count_y = (height + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;

    float frequency = base_frequency;
    float amplitude = base_amplitude;
    std::vector<float> accumulator(width * height, 0.0f);

    for (int o = 0; o < octaves; o++) {
        for (int cy = 0; cy < chunks_count_y; cy++) {
            for (int cx = 0; cx < chunks_count_x; cx++) {
                Chunk chunk(cx, cy);
                chunk.generate_chunk_pixels(accumulator, width, height, lookUpTable, frequency, amplitude, offset_x, offset_y);
            }
        }
        frequency *= lacunarity;
        amplitude *= persistence;
    }

    if (benchmark) {
        uint64_t clock_end = __rdtsc();
        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        
        std::cout << "Elapsed Clocks: " << (clock_end - clock_start) << "\n";
        std::cout << "Wall Time (ms): " << wall_ms << "\n";
    }

    if (!no_outputs){
        std::vector<uint8_t> output(width * height, 0);
        for (int i = 0; i < width * height; i++) {
            float v = std::clamp(accumulator[i], -1.0f, 1.0f);
            output[i] = static_cast<uint8_t>((v + 1.0f) * 0.5f * 255.0f);
        }
        save_output(output, width, height, 1, output_filename, output_format);
    }
}