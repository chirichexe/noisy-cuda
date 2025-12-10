/*
 * utils_global.hpp - utility function for noisy-cuda, global
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

#pragma once

#include "utils_global.hpp" 

#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * @brief save_output: saves the output data to a file in the specified format
 * 
 * @param output_data the output pixel data
 * @param width 
 * @param height 
 * @param channels 
 * @param filename_in 
 * @param format_str 
 */
void save_output(
    const std::vector<unsigned char>& output_data,
    int width,
    int height,
    unsigned int channels,
    const std::string& filename_in,
    const std::string& format_str 
) {
    std::string format = format_str;
    std::transform(format.begin(), format.end(), format.begin(), 
                   [](unsigned char c){ return std::tolower(c); });

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
        fprintf(stderr, "Warning: Unknown output format '%s'. Defaulting to PNG.\n", format_str.c_str());
        format = "png";
        extension = ".png";
    }
    
    std::string filename = filename_in;
    
    if (filename.size() < extension.size() || 
        filename.substr(filename.size() - extension.size()) != extension) 
    {
        filename += extension;
    }


    if (format == "png") {
        int success = stbi_write_png(filename.c_str(), width, height, channels, output_data.data(), width * channels);
        if (success) {
            printf("\nOutput saved as \"%s\"\n", filename.c_str());
        } else {
            fprintf(stderr, "ERROR: Could not write PNG file %s\n", filename.c_str());
        }
    } else if (format == "raw") {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(output_data.data()), output_data.size());
            file.close();
            printf("\nOutput saved as RAW file \"%s\"\n", filename.c_str());
        } else {
            fprintf(stderr, "ERROR: Could not open RAW file %s for writing.\n", filename.c_str());
        }
    } else if (format == "csv") {
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
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file << "P5\n"; 
            file << width << " " << height << "\n";
            file << "255\n"; 
            file.write(reinterpret_cast<const char*>(output_data.data()), output_data.size());
            file.close();
            printf("\nOutput saved as PPM (P5) file \"%s\"\n", filename.c_str());
        } else {
            fprintf(stderr, "ERROR: Could not open PPM file %s for writing.\n", filename.c_str());
        }
    }
}