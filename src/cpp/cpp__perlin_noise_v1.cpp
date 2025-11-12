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

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32

float frequency = 1.0f;   // più alto = dettagli più fini
float amplitude = 2.0f;   // più alto = contrasti più forti



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
    /* one for each angle: 0-upLeft 1-upRight 2-downRight 3-downLeft */
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

    void generate_chunk_pixels(unsigned char* output, int image_width, int image_height) const {
        printf("Generating chunk [%d,%d]\n", chunk_idx_x, chunk_idx_y);

        int maxPixel_x = chunk_idx_x * CHUNK_SIDE_LENGTH + CHUNK_SIDE_LENGTH;
        int max_chunk_pixel_x = maxPixel_x > image_width ? maxPixel_x - image_width : CHUNK_SIDE_LENGTH; 

        int maxPixel_y = chunk_idx_y * CHUNK_SIDE_LENGTH + CHUNK_SIDE_LENGTH;
        int max_chunk_pixel_y = maxPixel_y > image_height ? maxPixel_y - image_height : CHUNK_SIDE_LENGTH;

        for (int y = 0; y < max_chunk_pixel_y; y++) {
            for (int x = 0; x < max_chunk_pixel_x; x++) {
                // coordinate locali normalizzate [0,1]
                float sx = float(x) / CHUNK_SIDE_LENGTH;
                float sy = float(y) / CHUNK_SIDE_LENGTH;
                
                // applica frequenza
                sx *= frequency;
                sy *= frequency;

                // vettori distanza da ogni corner (0,0), (1,0), (1,1), (0,1)
                Vector2D d00 = {sx - 0.0f, sy - 0.0f};
                Vector2D d10 = {sx - 1.0f, sy - 0.0f};
                Vector2D d11 = {sx - 1.0f, sy - 1.0f};
                Vector2D d01 = {sx - 0.0f, sy - 1.0f};

                // dot product con i gradienti dei 4 angoli
                float dot00 = unit_vectors[0].dot(d00);
                float dot10 = unit_vectors[1].dot(d10);
                float dot11 = unit_vectors[2].dot(d11);
                float dot01 = unit_vectors[3].dot(d01);

                // fade su coordinate normalizzate
                Vector2D globalPos = get_global_pixel_position(x, y);

                float gx = (globalPos.x) / float(image_width - 1);
                float gy = (globalPos.y) / float(image_height - 1);
                float u = fade(sx);
                float v = fade(sy);

                // interpolazioni
                float nx0 = lerp(dot00, dot10, u) ;
                float nx1 = lerp(dot01, dot11, u);
                float value = lerp(nx0, nx1, v);

                // applica ampiezza
                value *= amplitude;
                value = std::clamp(value, -1.0f, 1.0f);
                
                // mappa [-1,1] → [0,255]
                value = (value + 1.0f) * 0.5f * 255.0f;

                // salva pixel nel buffer globale
                
                output[static_cast<int>(globalPos.y) * image_width + static_cast<int>(globalPos.x)] =
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

};




void generate_perlin_noise(const Options& opts) {

    /* perlin variables */
    int width = opts.width;
    int height = opts.height;

    /* preparing buffer image */
    unsigned int channels = 1;
    unsigned int imageSize = opts.width * opts.height * channels;
    unsigned char output[imageSize];
    
    /* calculating chunk size */
    int chunks_grid_width = floor(width / CHUNK_SIDE_LENGTH);
    if (width % CHUNK_SIDE_LENGTH != 0) 
        chunks_grid_width++;
    
    int chunks_grid_height = floor(height / CHUNK_SIDE_LENGTH);
    if (height % CHUNK_SIDE_LENGTH != 0) 
        chunks_grid_height++;

    /* applying the seed on the C pseudorandomicity rand() function */
    srand(opts.seed);
    

    /* generating angles unit vector */
    int angles_needed = (chunks_grid_width + 1) * (chunks_grid_height + 1);

    Vector2D angle_unit_vectors[chunks_grid_width + 1][chunks_grid_height + 1];

    for(int x = 0; x < chunks_grid_width + 1; x++){
        for(int y = 0; y < chunks_grid_height + 1; y++){

            float gx = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            float gy = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            
            angle_unit_vectors[x][y] = Vector2D(gx, gy).normalize();
        }
    }

    // generate chunk pixels
    for (int x = 0; x < chunks_grid_width; x++) {
        for (int y = 0; y < chunks_grid_height; y++) {            
            Chunk c_test = Chunk(x, y);

            // assign 4 angles from global grid
            c_test.unit_vectors[0] = angle_unit_vectors[x][y];       // TL
            c_test.unit_vectors[1] = angle_unit_vectors[x+1][y];   // TR
            c_test.unit_vectors[2] = angle_unit_vectors[x+1][y+1]; // BR
            c_test.unit_vectors[3] = angle_unit_vectors[x][y+1];   // BL

            c_test.generate_chunk_pixels(output, width, height);
        }
    }

    /* Save result */
    stbi_write_png( opts.output_filename.c_str(), width, height, channels, output, width * channels);
    printf("Output saved as %s\n", opts.output_filename.c_str());
    
    /* free memory */
    //free(angle_unit_vectors);
    //free(output);
}