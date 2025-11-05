/*
 * options.h - public interface for command-line parsing module.
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

#ifndef NOISYCUDA_OPTIONS_H
#define NOISYCUDA_OPTIONS_H

#define NOISYCUDA_OPTIONS_FORMAT_SIZE 16
#define NOISYCUDA_OPTIONS_FILENAME_SIZE 4096

#include <stdint.h>

typedef struct {
    int width;
    int height;
    int octaves;
    char format[NOISYCUDA_OPTIONS_FORMAT_SIZE];
    char output_filename[NOISYCUDA_OPTIONS_FILENAME_SIZE];
    int verbose;
    int cpu_mode;
    uint64_t seed;
    int seed_provided;
} ProgramOptions;

/*
 * parse_program_options
 *
 * Parse argc/argv and fill 'out'. Do not exit the process inside this
 * function; return codes allow caller to decide exit behaviour.
 *
 * Return values:
 *   0  - success: out filled with valid options
 *   1  - help was printed (caller should exit 0)
 *  -1  - error: diagnostics already printed to stderr (caller should exit 2)
 *
 * Notes:
 *   - The parser accepts an optional positional argument interpreted as a
 *     seed (unsigned integer). It also accepts --seed/-S long/short form.
 *   - If both positional seed and --seed are provided, parser fails (ambiguous).
 */

int parse_program_options(int argc, char **argv, ProgramOptions *out);

/*
* If verbose, prints the given program options
*/
void print_program_options( ProgramOptions opts );

#endif
