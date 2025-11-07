/*
 * cuda__utils.hpp - CUDA utility macros and functions
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

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * @brief check CUDA call for errors and exit on failure
 * @param call the CUDA call to check
 * 
 * @copyright 2025 Fabio Tosi, Alma Mater Studiorum - Università di Bologna
 * 
 */
#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n",        \
                __FILE__, __LINE__, error, cudaGetErrorString(error));      \
        std::exit(1);                                                       \
    }                                                                       \
}
