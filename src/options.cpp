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
#include <random>
#include <chrono>

Options parse_options(int argc, char** argv) {
    Options opts;

    // Help message lambda
    auto show_help = [argv]() {
        std::cout <<
        "Usage: " << argv[0] << " [seed] [OPTIONS]\n\n"
        "Perlin noise generator — option parsing module only.\n\n"
        "Positional 'seed' is accepted as an unsigned integer: if present it is\n"
        "interpreted as the RNG seed (e.g. './perlin 13813').\n"
        "\n"
        "  -h, --help                Show this help message and exit\n"
        "  --version                 Show program version and exit\n\n"
        "  -o, --output <filename>   Output filename. Default: perlin.<ext>\n"
        "  -f, --format <string>     Output format: png, raw, csv, ppm. Default: png\n"
        "  -s, --size <WxH>          Output size in pixels (width x height). Default: 2048x2048\n"
        "  -v, --verbose             Print processing steps and timings\n"
        "  -n, --no-output           Disable output file generation\n"
        "  -b, --benchmark           Generate CSV benchmark data\n"
        "\n"
        "  Perlin noise generation options:\n\n"
        "  -F, --frequency <float>   Base frequency (scale factor). Default: 1.0\n"
        "  -A, --amplitude <float>   Base amplitude. Default: 1.0\n"
        "  -L, --lacunarity <float>  Frequency multiplier per octave. Default: 2.0\n"
        "  -P, --persistence <float> Amplitude multiplier per octave. Default: 0.5\n"
        "  -O, --offset <x,y>        Offset for the noise generation. Default: 0,0\n"
        "  -C, --octaves <int>       Number of octaves (>=1). Default: 4\n";
    };

    auto show_version = []() {
    std::cout << "noisy-cuda " << PROGRAM_VERSION
            << " (" << BACKEND_NAME
            << " backend, version " << BACKEND_VERSION << ")"
            << " [commit " << GIT_HASH << "]" << std::endl;
    };

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Help option
        if (arg == "-h" || arg == "--help") {
            show_help();
            std::exit(0);
        }

        // Version option
        if (arg == "--version") {
            show_version();
            std::exit(0);
        }

        // Positional seed
        else if (arg[0] != '-') {
            opts.seed = std::stoull(arg);
            opts.seed_provided = true;
        }

        // Output filename option (help order: -o)
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --output");
            opts.output_filename = argv[++i];
        }

        // No output option (help order: -n)
        else if (arg == "-n" || arg == "--no-output") {
            opts.no_outputs = true;
        }

        // Format option (help order: -f)
        else if (arg == "-f" || arg == "--format") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --format");
            opts.format = argv[++i];
        }

        // Size option (help order: -s)
        else if (arg == "-s" || arg == "--size") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --size");
            std::string val = argv[++i];
            std::replace(val.begin(), val.end(), 'x', ' ');
            std::istringstream ss(val);
            ss >> opts.width >> opts.height;
            if (!ss || opts.width <= 0 || opts.height <= 0)
                throw std::invalid_argument("Invalid size format, expected WxH");
        }

        // Verbose option (help order: -v)
        else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        }

        // Frequency option (Perlin options order: -F)
        else if (arg == "-F" || arg == "--frequency") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --frequency");
            opts.frequency = std::stof(argv[++i]);
            if (opts.frequency <= 0.0f) throw std::invalid_argument("Frequency must be > 0");
        }

        // Amplitude option (Perlin options order: -A)
        else if (arg == "-A" || arg == "--amplitude") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --amplitude");
            opts.amplitude = std::stof(argv[++i]);
            if (opts.amplitude <= 0.0f) throw std::invalid_argument("Amplitude must be > 0");
        }

        // Lacunarity option (Perlin options order: -L)
        else if (arg == "-L" || arg == "--lacunarity") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --lacunarity");
            opts.lacunarity = std::stof(argv[++i]);
            if (opts.lacunarity <= 1.0f) throw std::invalid_argument("Lacunarity must be > 1");
        }

        // Persistence option (Perlin options order: -P)
        else if (arg == "-P" || arg == "--persistence") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --persistence");
            opts.persistence = std::stof(argv[++i]);
            if (opts.persistence <= 0.0f || opts.persistence >= 1.0f)
                throw std::invalid_argument("Persistence must be in (0,1)");
        }

        // Offset option (Perlin options order: -O)
        else if (arg == "-O" || arg == "--offset") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --offset");

            std::string val = argv[++i];
            std::replace(val.begin(), val.end(), ',', ' ');
            std::istringstream ss(val);
            ss >> opts.offset_x >> opts.offset_y;

            if (!ss)
                throw std::invalid_argument("Invalid offset format, expected x,y");
        }

        // Octaves option (Perlin options order: -C)
        else if (arg == "-C" || arg == "--octaves") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for --octaves");
            opts.octaves = std::stoi(argv[++i]);
            if (opts.octaves < 1) throw std::invalid_argument("Octaves must be >= 1");
        }

        // Benchmark option (help order: -b)
        else if (arg == "-b" || arg == "--benchmark") {
            opts.benchmark = true;
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

    // If no seed provided, generate one from time and random_device
    if (!opts.seed_provided) {
        unsigned long long time_seed =
            static_cast<unsigned long long>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::seed_seq seq{time_seed, static_cast<unsigned long long>(std::random_device{}())};
        std::vector<unsigned long long> seeds(1);
        seq.generate(seeds.begin(), seeds.end());
        opts.seed = seeds[0];
    }

    return opts;
}

void print_program_options(Options opts) {
    if (opts.verbose) {
        printf("\n");
        printf("              .__                                         .___       \n");
        printf("  ____   ____ |__| _________.__.           ____  __ __  __| _/____   \n");
        printf(" /    \\ /  _ \\|  |/  ___<   |  |  ______ _/ ___\\|  |  \\/ __ |\\__  \\  \n");
        printf("|   |  (  <_> )  |\\___ \\ \\___  | /_____/ \\  \\___|  |  / /_/ | / __ \\_\n");
        printf("|___|  /\\____/|__/____  >/ ____|          \\___  >____/\\____ |(____  /\n");
        printf("     \\/               \\/ \\/                   \\/           \\/     \\/ \n");
        printf("\n");

        fprintf(stderr, "Configuration:\n");
        fprintf(stderr, "  Size:        %d x %d\n", opts.width, opts.height);
        fprintf(stderr, "  Format:      %s\n", opts.format.c_str());
        fprintf(stderr, "  No output:   %s\n", opts.no_outputs ? "enabled" : "disabled");
        if (!opts.no_outputs)
            fprintf(stderr, "  Output file: %s\n", opts.output_filename.c_str());
        fprintf(stderr, "  Offset:      (%d, %d)\n", opts.offset_x, opts.offset_y);

        if (opts.seed_provided) {
            std::cerr << "  Seed:        " << opts.seed << " (provided)\n";
        } else {
            std::cerr << "  Seed:        " << opts.seed << " (auto-generated)\n";
        }
        fprintf(stderr, "  Verbose:     enabled\n");
        printf("\n");

        /* perlin parameters */
            std::cout << "Generating Perlin noise with options:\n"
            << "  freq=" << opts.frequency << "\n"
            << "  amp=" << opts.amplitude << "\n"
            << "  octaves=" << opts.octaves << "\n"
            << "  lacunarity=" << opts.lacunarity << "\n"
            << "  persistence=" << opts.persistence << "\n"
            << "\n";
    }
}
