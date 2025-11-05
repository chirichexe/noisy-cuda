/*
 * main.c - common main function
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
#include "options.h"
#include <stdio.h>
#include <perlin.h>

int main(int argc, char **argv) {
    ProgramOptions opts;
    int r = parse_program_options(argc, argv, &opts);

    if (r == 1) {
        /* Help printed by parser — exit success as expected by Linux conventions */
        return 0;
    } else if (r == -1) {
        /* Parse error; diagnostics already printed; return conventional exit code 2 */
        return 2;
    }

    /* Strict summary when verbose */
    print_program_options(opts);

    /* program starts */
    generate_perlin(&opts);
    /* program ends */

    return 0;
}

