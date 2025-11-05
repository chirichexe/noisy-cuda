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
#include "gpu_generate_perlin.h"

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



struct timespec clockStart, clockEnd;
void performance_print(int ended, char* label);
void cpu_generate_perlin(ProgramOptions *opts, unsigned char* output);


/**
 * @brief Run the CUDA backend
 * @param opts Program options
 * @return 0 on success, error code on failure
 */
int generate_perlin(const ProgramOptions *opts) {

    /* starting the generation... */
    if(opts->verbose)
        printf("Starting generating perlin noise...\n");
    
    /* preparing buffer image */
    unsigned int channels = 1;
    unsigned int imageSize = opts->width * opts->height * channels;
    unsigned char* output = (unsigned char*)malloc(imageSize);
    
    /* generation */
    performance_print(0, "generation");

    if (opts->cpu_mode)
        cpu_generate_perlin(opts, output);
    else
        if (gpu_generate_perlin(opts, output) != 0)
            return -1;
    
    performance_print(1, "generation");


    /* Save result */
    stbi_write_png(opts->output_filename, opts->width, opts->height, channels, output, opts->width * channels);
    printf("Output saved as %s\n", opts->output_filename);
    
    /* free memory */
    free(output);
    
    /* end generation */
    return 0;
}



void performance_print(int ended, char* label){
    if(ended == 0)
        clock_gettime(CLOCK_MONOTONIC, &clockStart);
    else {
        clock_gettime(CLOCK_MONOTONIC, &clockEnd);
        double milliseconds = (clockEnd.tv_sec - clockStart.tv_sec) * 1000.0 + (clockEnd.tv_nsec - clockStart.tv_nsec) / 1e6;
        printf("[%s]: elapsed time: %.3f ms\n", label, milliseconds);
    }
}


void cpu_generate_perlin(ProgramOptions *opts, unsigned char* output){
    /* applying the seed on the C pseudorandomicity rand() function */
    srand(opts->seed);
    
    /* randomize the image */
    for(int y = 0; y < opts->height; y++){
        for(int x = 0; x < opts->width; x++){
            output[y * opts->width + x] = (unsigned char)(rand() * 255);
        }
    }
}