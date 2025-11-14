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
 * distributed under the License is distributed on an "AS IS" BASIS,
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
#include <cmath>
#include <vector>
#include <cstdio>

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
        std::vector<unsigned char>& output,
        int image_width,
        int image_height,
        const std::vector<std::vector<Vector2D>>& gradients,
        int chunks_x,
        int chunks_y,
        float frequency,
        float amplitude
    ) const {
        int start_x = chunk_x * CHUNK_SIDE_LENGTH;
        int start_y = chunk_y * CHUNK_SIDE_LENGTH;

        int end_x = std::min(start_x + CHUNK_SIDE_LENGTH, image_width);
        int end_y = std::min(start_y + CHUNK_SIDE_LENGTH, image_height);

        for (int y = start_y; y < end_y; y++) {
            for (int x = start_x; x < end_x; x++) {
                // normalized coordinates scaled by frequency
                float fx = ((float)x / (float)image_width) * frequency;
                float fy = ((float)y / (float)image_height) * frequency;

                // integer grid cell
                int x0 = (int)std::floor(fx);
                int y0 = (int)std::floor(fy);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                // local coordinates within the cell
                float sx = fx - (float)x0;
                float sy = fy - (float)y0;

                // corner gradients (shared from global grid)
                const Vector2D& g00 = gradients[x0 % chunks_x][y0 % chunks_y];
                const Vector2D& g10 = gradients[x1 % chunks_x][y0 % chunks_y];
                const Vector2D& g01 = gradients[x0 % chunks_x][y1 % chunks_y];
                const Vector2D& g11 = gradients[x1 % chunks_x][y1 % chunks_y];

                // distance vectors from corners
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

                // amplitude adjustment and normalization
                value *= amplitude;
                value = std::clamp(value, -1.0f, 1.0f);

                // map [-1,1] -> [0,255]
                unsigned char pixel = static_cast<unsigned char>((value + 1.0f) * 0.5f * 255.0f);
                output[y * image_width + x] = pixel;
            }
        }
    }
};


/*
 * generate_perlin_noise - generates a 2D Perlin noise map using chunks
 */
void generate_perlin_noise(const Options& opts) {

    /* initialize parameters */
    int width = opts.width;
    int height = opts.height;
    float frequency = opts.frequency;
    float amplitude = opts.amplitude;
    int seed = opts.seed;
    
    /* randomize from seed */
    srand(seed);

    /* output buffer preparation*/
    unsigned int channels = 1;
    std::vector<unsigned char> output(width * height * channels, 0);

    /* calculate chunk grid */
    int chunks_x = (width  + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH; // for the CUDA case, is better to check which 
                                                                         // chunk won't be filled
    int chunks_y = (height + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;

    /* initialize gradient vectors */
    // matrix of Vector2D
    std::vector<std::vector<Vector2D>> gradients(chunks_x, std::vector<Vector2D>(chunks_y));

    for (int gx = 0; gx < chunks_x; gx++) {
        for (int gy = 0; gy < chunks_y; gy++) {
            float rx = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float ry = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            gradients[gx][gy] = Vector2D(rx, ry).normalize();
        }
    }

    /* generate noise per chunk (continuous thanks to shared gradients) */
    for (int cy = 0; cy < chunks_y; cy++) {
        for (int cx = 0; cx < chunks_x; cx++) {
            Chunk chunk(cx, cy);
            chunk.generate_chunk_pixels(output, width, height, gradients, chunks_x, chunks_y, frequency, amplitude);
        }
    }

    /* save the generated noise image */
    stbi_write_png(opts.output_filename.c_str(), width, height, channels, output.data(), width * channels);
    printf("Output saved as %s\n", opts.output_filename.c_str());
}
