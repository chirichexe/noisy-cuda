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

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32


/* vector structure */
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


static float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}


void generate_perlin_noise(const Options& opts) {
    int width = opts.width;
    int height = opts.height;
    float frequency = opts.frequency;   // più alto = dettagli più fini
    float amplitude = opts.amplitude;   // più alto = contrasti più forti

    unsigned int channels = 1;
    std::vector<unsigned char> output(width * height * channels, 0);

    // calcolo griglia gradienti globale
    int grad_w = width / CHUNK_SIDE_LENGTH + 2;   //CHANGED
    int grad_h = height / CHUNK_SIDE_LENGTH + 2;  //CHANGED

    srand(opts.seed);
    std::vector<std::vector<Vector2D>> gradients(grad_w, std::vector<Vector2D>(grad_h));

    for (int gx = 0; gx < grad_w; gx++) {
        for (int gy = 0; gy < grad_h; gy++) {
            float rx = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float ry = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            gradients[gx][gy] = Vector2D(rx, ry).normalize();
        }
    }

    // Genera rumore continuo su tutta l'immagine (non per chunk)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // Coordinate normalizzate (da 0 a frequency)
            float fx = ((float)x / (float)width) * frequency;
            float fy = ((float)y / (float)height) * frequency;

            // Determina le celle integer
            int x0 = (int)std::floor(fx);
            int y0 = (int)std::floor(fy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // coordinate locali dentro la cella
            float sx = fx - (float)x0;
            float sy = fy - (float)y0;

            // gradienti dei 4 angoli
            const Vector2D& g00 = gradients[x0 % grad_w][y0 % grad_h];
            const Vector2D& g10 = gradients[x1 % grad_w][y0 % grad_h];
            const Vector2D& g01 = gradients[x0 % grad_w][y1 % grad_h];
            const Vector2D& g11 = gradients[x1 % grad_w][y1 % grad_h];

            // distanze
            Vector2D d00(sx,     sy);
            Vector2D d10(sx - 1, sy);
            Vector2D d01(sx,     sy - 1);
            Vector2D d11(sx - 1, sy - 1);

            // dot products
            float dot00 = g00.dot(d00);
            float dot10 = g10.dot(d10);
            float dot01 = g01.dot(d01);
            float dot11 = g11.dot(d11);

            // fade curve
            float u = fade(sx);
            float v = fade(sy);

            // interpolazione
            float nx0 = lerp(dot00, dot10, u);
            float nx1 = lerp(dot01, dot11, u);
            float value = lerp(nx0, nx1, v);

            value *= amplitude;
            value = std::clamp(value, -1.0f, 1.0f);
            unsigned char pixel = static_cast<unsigned char>((value + 1.0f) * 0.5f * 255.0f);

            output[y * width + x] = pixel;
        }
    }

    stbi_write_png(opts.output_filename.c_str(), width, height, channels, output.data(), width * channels);
    printf("Output saved as %s\n", opts.output_filename.c_str());
}
