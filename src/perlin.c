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

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "options.h"

// Perlin imports
#include "gpu_generate_perlin.h"
#include "cpu_generate_perlin.h"

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * @brief Run the CUDA backend
 * @param opts Program options
 * @return 0 on success, error code on failure
 */
int generate_perlin(const ProgramOptions *opts) {
    
    /* preparing buffer image */
    unsigned int channels = 1;
    unsigned int imageSize = opts->width * opts->height * channels;
    unsigned char* output = (unsigned char*)malloc(imageSize);
    
    /* starting the generation... */
    if(opts->verbose)
        printf("Starting generating perlin noise...\n");
    
    if (opts->cpu_mode) {
        if (cpu_generate_perlin(opts, output) != 0)
            return -1;
    } else {
        if (gpu_generate_perlin(opts, output) != 0)
            return -1;
    }

    /* Save result */
    stbi_write_png(opts->output_filename, opts->width, opts->height, channels, output, opts->width * channels);
    printf("Output saved as %s\n", opts->output_filename);
    
    /* free memory */
    free(output);
    
    /* end generation */
    return 0;
}


