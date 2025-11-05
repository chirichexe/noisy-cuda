/*
 * perlin.c - generate the perlin noise
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

 #define _POSIX_C_SOURCE 199309L // x Timing
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#include "options.h"
#include "cuda_kernel.h"


struct timespec clockStart, clockEnd;
void performance_print(int ended, char* label){
    if(ended == 0)
        clock_gettime(CLOCK_MONOTONIC, &clockStart);
    else {
        clock_gettime(CLOCK_MONOTONIC, &clockEnd);
        double milliseconds = (clockEnd.tv_sec - clockStart.tv_sec) * 1000.0 + (clockEnd.tv_nsec - clockStart.tv_nsec) / 1e6;
        printf("[%s]: elapsed time: %.3f ms\n", label, milliseconds);
    }
}


/**
 * @brief Run the CUDA backend
 * @param opts Program options
 * @return 0 on success, error code on failure
 */
int generate_perlin(const ProgramOptions *opts) {
    /* starting the generation... */
    if(opts->verbose)
        printf("Starting generating perlin noise...\n");

    /* applying the seed on the C pseudorandomicity rand() function */
    srand(opts->seed);
    
    //
    printf("%d\n", rand());
    printf("%d\n", rand());
    printf("%d\n", rand());

   // return run_cuda_kernel(num_threads);
}
