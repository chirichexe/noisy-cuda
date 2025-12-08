/*
 * cuda__perlin_noise.cu - perlin noise: CUDA implementation
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

// Features
/*
    - Generate a perlin noise with octaves
    - Use a kernel to calculate a perlin noise pixel
*/

// Future ideas
/*
    GENERAL:
    - Add the permutations instead of gradients giant matrix
    - Check for variable types used on algorithm 
      (for example, float, char ... )
    - Limit the size of the variables of the image to avoid 
      crash or too large outputs
    
    FOR CUDA:
    - Generate gradients with a kernel
    - Manage chunks and octaves on device
    - try to convert the output on the wanted format 
      directly on the kernel (not in CPU)
    - Adapt the algorithm execution to the device hardware capabilities
*/

#include "perlin_noise.hpp"
#include "cuda__utils.hpp"

// suppress image lib warnings
#pragma nv_diag_suppress 550

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <algorithm>
#include <inttypes.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>

/* device variables */
#define BLOCK_SIZE 16

/* chunk variables */
#define NOISE_GRID_CELL_SIZE 64

/**
 * @brief Kernel to generate Perlin noise for each pixel
 *  
 */
__global__ void gpu_generate_perlin_pixel(
    float * buffer, // reference to global float buffer
    int seed, 
    int image_width,
    int image_height,
    const Vector2D* gradients,
    float frequency,
    float amplitude,
    int chunks_count_x,
    int chunks_count_y,
    int offset_x,
    int offset_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= image_width || y >= image_height ) 
        return;

    // normalized coordinates scaled by frequency
    //float lerp_coeff =  image_width < image_height ? image_height : image_width;
    float fx = ((float)(x + offset_x) / (float)image_width) * frequency;
    float fy = ((float)(y + offset_y) / (float)image_height) * frequency;

    // integer grid cell
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // local coordinates within the cell
    float sx = fx - (float)x0;
    float sy = fy - (float)y0;

    // corner gradients (shared from global grid)
    // uses 1D index: Index = y * num_cols + x
    int grad_x0 = x0 % chunks_count_x;
    int grad_y0 = y0 % chunks_count_y;
    int grad_x1 = x1 % chunks_count_x;
    int grad_y1 = y1 % chunks_count_y;

    // tl (top-left)
    const Vector2D& g00 = gradients[grad_y0 * chunks_count_x + grad_x0];  
    // tr (top-right)
    const Vector2D& g10 = gradients[grad_y0 * chunks_count_x + grad_x1];  
    // bl (bottom-left)
    const Vector2D& g01 = gradients[grad_y1 * chunks_count_x + grad_x0];  
    // br (bottom-right)
    const Vector2D& g11 = gradients[grad_y1 * chunks_count_x + grad_x1];

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
    std::string output_format = opts.format;

    /* setting CUDA DEVICE */
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    
    /* allocate device memory for output */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    /* float accumulation buffer (needed for octaves) */
    // host accumulator
    std::vector<float> accumulator(width * height, 0.0f);
    
    // device accumulator
    size_t buffer_size = width * height * sizeof(float);
    float * d_accumulator;
    CHECK(cudaMalloc((void **)&d_accumulator, buffer_size));
    
    /* calculate chunk grid size */
    int chunks_count_x = (width  + NOISE_GRID_CELL_SIZE - 1) / NOISE_GRID_CELL_SIZE;
    int chunks_count_y = (height + NOISE_GRID_CELL_SIZE - 1) / NOISE_GRID_CELL_SIZE;
    
    /* initialize gradient vectors */
    // host gradient vectors
    int gradients_size = chunks_count_x * chunks_count_y;
    std::vector<Vector2D> h_gradients(gradients_size);
    
    for (int gx = 0; gx < chunks_count_x; gx++) {
        for (int gy = 0; gy < chunks_count_y; gy++) {
            float rx = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float ry = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            
            int idx = gy * chunks_count_x + gx; 
            h_gradients[idx] = Vector2D(rx, ry).normalize();
        }
    }

    // device gradient vectors
    Vector2D* d_gradients;
    size_t gradients_buffer_size = gradients_size * sizeof(Vector2D);
    CHECK(cudaMalloc((void **)&d_gradients, gradients_buffer_size));

    // host -> device copy
    CHECK(cudaMemcpy(d_gradients, h_gradients.data(), gradients_buffer_size, cudaMemcpyHostToDevice));
    
    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;
    
    for (int o = 0; o < octaves; o++) {
        
        /* launch the kernel */
        // generate noise for this octave using the chunk pipeline
        gpu_generate_perlin_pixel<<<grid, block>>>(d_accumulator, seed, width, height, d_gradients, frequency, amplitude, chunks_count_x, chunks_count_y, offset_x, offset_y);

        frequency *= lacunarity;   // controls frequency growth
        amplitude *= persistence;  // controls amplitude decay
    }

    
    /* wait for kernel to complete */
    CHECK(cudaDeviceSynchronize());
   
    /* copy and free the result from the device to the output pointer */
    CHECK(cudaMemcpy(accumulator.data(), d_accumulator, buffer_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_accumulator));
    CHECK(cudaFree(d_gradients));

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
    //stbi_write_png(output_filename.c_str(), width, height, channels, output.data(), width * channels);
    //printf("\nOutput saved as \"%s\"\n", output_filename.c_str());

    save_output(
        output,
        width,
        height,
        channels,
        output_filename,
        output_format
    );

}


/**
 * @brief Saves the pixel buffer in various formats (PNG, RAW, CSV, PPM).
 */
void save_output(
    const std::vector<unsigned char>& output_data,
    int width,
    int height,
    unsigned int channels,
    const std::string& filename_in,
    const std::string& format_str 
) {
    // Normalize the format string to lowercase
    std::string format = format_str;
    std::transform(format.begin(), format.end(), format.begin(), 
                   [](unsigned char c){ return std::tolower(c); });

    // Determine the correct extension for the output filename
    std::string extension;
    if (format == "png") {
        extension = ".png";
    } else if (format == "raw") {
        extension = ".raw";
    } else if (format == "csv") {
        extension = ".csv";
    } else if (format == "ppm") {
        extension = ".ppm";
    } else {
        // Fallback: PNG
        fprintf(stderr, "Warning: Unknown output format '%s'. Defaulting to PNG.\n", format_str.c_str());
        format = "png";
        extension = ".png";
    }
    
    // Assume filename_in is already normalized or contains the correct extension.
    // However, we ensure the filename *ends* with the correct extension for the chosen format.
    std::string filename = filename_in;
    
    // Check if filename_in already has the required extension.
    if (filename.size() < extension.size() || 
        filename.substr(filename.size() - extension.size()) != extension) 
    {
        // If the file doesn't end with the determined extension, append it.
        // This ensures "perlin" + "png" -> "perlin.png", but "perlin.csv" + "png" -> "perlin.csv.png" 
        // which might be undesirable. Given your constraint, we'll append it safely.
        filename += extension;
    }


    // Save the output

    if (format == "png") {
        // PNG (uses stb_image_write)
        int success = stbi_write_png(filename.c_str(), width, height, channels, output_data.data(), width * channels);
        if (success) {
            printf("\nOutput saved as \"%s\"\n", filename.c_str());
        } else {
            fprintf(stderr, "ERROR: Could not write PNG file %s\n", filename.c_str());
        }
    } else if (format == "raw") {
        // RAW (binary raw data)
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(output_data.data()), output_data.size());
            file.close();
            printf("\nOutput saved as RAW file \"%s\"\n", filename.c_str());
        } else {
            fprintf(stderr, "ERROR: Could not open RAW file %s for writing.\n", filename.c_str());
        }
    } else if (format == "csv") {
        // CSV (comma-separated textual data)
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "Width," << width << "\n";
            file << "Height," << height << "\n";
            file << "Channels," << channels << "\n";

            size_t index = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    file << (int)output_data[index]; 
                    index += channels; 
                    if (x < width - 1) {
                        file << ",";
                    }
                }
                file << "\n";
            }
            file.close();
            printf("\nOutput saved as CSV file \"%s\"\n", filename.c_str());
        } else {
            fprintf(stderr, "ERROR: Could not open CSV file %s for writing.\n", filename.c_str());
        }
    } else if (format == "ppm") {
        // PPM (Portable Graymap P5 binary format)
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file << "P5\n"; 
            file << width << " " << height << "\n";
            file << "255\n"; // Max value
            file.write(reinterpret_cast<const char*>(output_data.data()), output_data.size());
            file.close();
            printf("\nOutput saved as PPM (P5) file \"%s\"\n", filename.c_str());
        } else {
            fprintf(stderr, "ERROR: Could not open PPM file %s for writing.\n", filename.c_str());
        }
    }
}