/*
 * options.cpp - implementation of options parsing
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

#include "options.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

/**
 * @brief parse_options - parse command line arguments
 * 
 * @param argc the number of command line arguments
 * @param argv the command line arguments
 * @return Options 
 */
Options parse_options(int argc, char** argv) {
    Options opts;

    // Set default options
    opts.width = 2048;
    opts.height = 2048;
    opts.octaves = 1;
    opts.format = "png";
    opts.output_filename = "perlin.png";
    opts.verbose = false;
    opts.seed = 0;
    opts.seed_provided = false;

    // Help message lambda
    auto show_help = [argv]() {
        std::cout <<
        "Usage: " << argv[0] << " [OPTIONS] [seed]\n\n"
        "Perlin noise generator — option parsing module only.\n\n"
        "  -h, --help                Show this help message and exit\n"
        "  -s, --size <WxH>          Image size in pixels (width x height). Default: 2048x2048\n"
        "  -O, --octaves <int>       Number of octaves (>=1). Default: 1\n"
        "  -f, --format <string>     Output format: png, raw, csv, ppm. Default: png\n"
        "  -o, --output <filename>   Output filename. Default: perlin.<ext>\n"
        "  -v, --verbose             Print processing steps and timings\n"
        "  -S, --seed <uint64>       Provide explicit seed (alternative to positional)\n\n"
        "Positional 'seed' is accepted as an unsigned integer: if present it is\n"
        "interpreted as the RNG seed (e.g. './perlin 13813'). If both positional\n"
        "seed and --seed are provided the parser fails (ambiguous).\n";
    };

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Help option
        if (arg == "-h" || arg == "--help") {
            show_help();
            std::exit(0);
        }

        // Size option
        else if (arg == "-s" || arg == "--size") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --size");
            std::string val = argv[++i];
            std::replace(val.begin(), val.end(), 'x', ' ');
            std::istringstream ss(val);
            ss >> opts.width >> opts.height;
            if (!ss || opts.width <= 0 || opts.height <= 0)
                throw std::invalid_argument("Invalid size format, expected WxH");
        }

        // Octaves option
        else if (arg == "-O" || arg == "--octaves") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --octaves");
            opts.octaves = std::stoi(argv[++i]);
            if (opts.octaves < 1) throw std::invalid_argument("Octaves must be >= 1");
        }

        // Format option
        else if (arg == "-f" || arg == "--format") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --format");
            opts.format = argv[++i];
        }

        // Output filename option
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --output");
            opts.output_filename = argv[++i];
        }

        // Verbose option
        else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        }

        // Seed option
        else if (arg == "-S" || arg == "--seed") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --seed");
            opts.seed = std::stoull(argv[++i]);
            opts.seed_provided = true;
        }

        // Positional seed
        else if (arg[0] != '-') {
            if (opts.seed_provided)
                throw std::invalid_argument("Cannot provide both positional seed and --seed");
            opts.seed = std::stoull(arg);
            opts.seed_provided = true;
        }

        // Unknown option
        else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    // Auto-set output filename extension if default name is still used
    if (opts.output_filename == "perlin.png" && opts.format != "png") {
        opts.output_filename = "perlin." + opts.format;
    }

    return opts;
}

