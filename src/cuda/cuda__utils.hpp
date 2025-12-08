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

#include <cmath>

/**
 * @brief Simple 2D vector structure
 * 
 */
struct Vector2D {
    float x = 0.0f;
    float y = 0.0f;

    Vector2D() = default;

    __host__ __device__ Vector2D(float x_, float y_) : x(x_), y(y_) {}

    __host__ __device__ Vector2D operator-(const Vector2D& other) const {
        return {x - other.x, y - other.y};
    }

    __host__ __device__ float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    __host__ __device__ float length() const {
        return std::sqrt(x * x + y * y);
    }

    __host__ __device__ Vector2D normalize() const {
        float len = length();
        return len > 0 ? Vector2D(x / len, y / len) : Vector2D(0, 0);
    }
};

/**
 * @brief Fade function for smooth interpolation on CUDA device
 * 
 * @param t 
 * @return __device__ 
 */
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

/**
 * @brief Linear interpolation function for CUDA device
 * 
 * @param a 
 * @param b 
 * @param t 
 * @return __device__ 
 */
__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}