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

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32

/* vector structure */
struct Vector2D {
    float x = 0.0f;
    float y = 0.0f;

    /* constructors */
    Vector2D() = default;
    Vector2D(float x_, float y_) : x(x_), y(y_) {}

    /* sum */
    Vector2D operator+(const Vector2D& other) const {
        return {x + other.x, y + other.y};
    }

    /* sub */
    Vector2D operator-(const Vector2D& other) const {
        return {x - other.x, y - other.y};
    }

    /* scalar multiplication */
    Vector2D operator*(float scalar) const {
        return {x * scalar, y * scalar};
    }

    /* dot product */
    float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    /* length */
    float length() const {
        return std::sqrt(x * x + y * y);
    }

    /* normalize */
    Vector2D normalize() const {
        float len = length();
        return len > 0 ? Vector2D(x / len, y / len) : Vector2D(0, 0);
    }
};

/* chunk structure */
struct Chunk {
    int chunk_idx_x = 0;
    int chunk_idx_y = 0;
    /* one for each corner: 0-upLeft 1-upRight 2-downRight 3-downLeft */
    Vector2D unit_vectors[4];

    /* constructor */
    Chunk(int idx_x_, int idx_y_) : chunk_idx_x(idx_x_), chunk_idx_y(idx_y_) {}
    
    // Fade function for smoother interpolation
    static float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    static float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }

    void generate_chunk_pixels(unsigned char* output, int image_width) const {
        for (int x = 0; x < CHUNK_SIDE_LENGTH; x++) {
            for (int y = 0; y < CHUNK_SIDE_LENGTH; y++) {
                Vector2D pixel_pos = {static_cast<float>(x), static_cast<float>(y)};

                // Local coordinates relative to each corner
                float dot00 = unit_vectors[0].dot(pixel_pos - get_corner_position_in_chunk(0));
                float dot01 = unit_vectors[1].dot(pixel_pos - get_corner_position_in_chunk(1));
                float dot10 = unit_vectors[2].dot(pixel_pos - get_corner_position_in_chunk(2));
                float dot11 = unit_vectors[3].dot(pixel_pos - get_corner_position_in_chunk(3));

                // Fractional distances
                float fx = static_cast<float>(x) / (CHUNK_SIDE_LENGTH - 1);
                float fy = static_cast<float>(y) / (CHUNK_SIDE_LENGTH - 1);

                float u = fade(fx);
                float v = fade(fy);

                // Bilinear interpolation
                float nx0 = lerp(dot00, dot10, u);
                float nx1 = lerp(dot01, dot11, u);
                float value = lerp(nx0, nx1, v);

                // Map to [0, 255]
                value = (value + 1.0f) * 0.5f * 255.0f; // map [-1,1] -> [0,255]

                Vector2D global = get_global_pixel_position(x, y);
                output[static_cast<int>(global.y) * image_width + static_cast<int>(global.x)] =
                    static_cast<unsigned char>(value);
            }
        }
    }

    /* return the pixel position relative to this chunk */
    Vector2D get_global_pixel_position(int inside_x, int inside_y) const {
        return Vector2D(
            chunk_idx_x * CHUNK_SIDE_LENGTH + inside_x,
            chunk_idx_y * CHUNK_SIDE_LENGTH + inside_y
        );
    }

    Vector2D get_corner_position_in_chunk(int corner_id) const {
        switch(corner_id){
            default: return Vector2D(0,0);
            case 1: return Vector2D(CHUNK_SIDE_LENGTH-1, 0);
            case 2: return Vector2D(CHUNK_SIDE_LENGTH-1, CHUNK_SIDE_LENGTH-1);
            case 3: return Vector2D(0, CHUNK_SIDE_LENGTH-1);
        }
    }

};


void generate_perlin_noise(const Options& opts) {

    /* preparing buffer image */
    unsigned int channels = 1;
    unsigned int imageSize = opts.width * opts.height * channels;
    unsigned char* output = (unsigned char*)malloc(imageSize);

    /* perlin variables */
    int width = opts.width;
    int height = opts.height;
    
    int chunks_grid_width = width / CHUNK_SIDE_LENGTH;
    int chunks_grid_height = height / CHUNK_SIDE_LENGTH;

    /* applying the seed on the C pseudorandomicity rand() function */
    srand(opts.seed);

    // generate corner unit vectors
    int corners_needed = (chunks_grid_width+1) * (chunks_grid_height+1);
    Vector2D* corner_unit_vectors = (Vector2D*)malloc(sizeof(Vector2D) * corners_needed);
    for(int j = 0; j < chunks_grid_width; j++){
        for(int i = 0; i < chunks_grid_height; i++){
            float gx = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            float gy = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            corner_unit_vectors[j * chunks_grid_width + i] = Vector2D(gx, gy).normalize();
        }
    }

    // generate chunk pixels
    for (int j = 0; j < chunks_grid_height; j++) {
        for (int i = 0; i < chunks_grid_width; i++) {            
            Chunk c_test = Chunk(i, j);

            // assign 4 corners from global grid
            c_test.unit_vectors[0] = corner_unit_vectors[j * chunks_grid_width + i];       // TL
            c_test.unit_vectors[1] = corner_unit_vectors[j * chunks_grid_width + (i+1)];   // TR
            c_test.unit_vectors[2] = corner_unit_vectors[(j+1) * chunks_grid_width + (i+1)]; // BR
            c_test.unit_vectors[3] = corner_unit_vectors[(j+1) * chunks_grid_width + i];   // BL

            c_test.generate_chunk_pixels(output, width);
        }
    }

    /* Save result */
    stbi_write_png( opts.output_filename.c_str(), width, height, channels, output, width * channels);
    printf("Output saved as %s\n", opts.output_filename.c_str());
    
    /* free memory */
    free(corner_unit_vectors);
    free(output);
}
