/*
 * options.hpp - definition of Options struct
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

#include <string>
#include <cstdint>

/**
 * @brief Backend identification macros
 */
#ifndef BACKEND_NAME
#define BACKEND_NAME "unknown"
#endif

#ifndef BACKEND_VERSION
#define BACKEND_VERSION "latest"
#endif

#ifndef PROGRAM_VERSION
#define PROGRAM_VERSION "0.0.1"
#endif

#ifndef GIT_HASH
#define GIT_HASH "unknown"
#endif

/** 
 * @brief Options - structure to hold program options
 */
struct Options {
    int width = 2048;
    int height = 2048;
    int octaves = 4;
    float frequency = 50.0f;
    float amplitude = 1.0f;
    float lacunarity = 2.0f;
    float persistence = 0.5f;
    int offset_x = 0;
    int offset_y = 0;
    std::string format = "png";
    std::string output_filename = "perlin.png";
    bool verbose = false;
    std::uint64_t seed = 0;
    bool seed_provided = false;
    bool no_outputs;
};

/**
 * @brief parse_options - parse command line arguments
 * 
 * @param argc the number of command line arguments
 * @param argv the command line arguments
 * @return Options 
 */
Options parse_options(int argc, char** argv);

/**
 * @brief print_program_options - print program options if verbose is enabled
 * 
 * @param opts the program options
 */
void print_program_options(Options opts);


