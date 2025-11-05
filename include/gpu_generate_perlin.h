/*
 * cuda_kernel.h - header file for cuda kernel
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
#ifndef GPU_GENERATE_PERLIN_H
#define GPU_GENERATE_PERLIN_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Simple CUDA kernel that prints thread info
 * @param num_threads Number of threads to launch
 * @return 0 on success, error code on failure
 */
int gpu_generate_perlin(ProgramOptions *opts, unsigned char* output);

#ifdef __cplusplus
}
#endif

#endif // GPU_GENERATE_PERLIN_H
