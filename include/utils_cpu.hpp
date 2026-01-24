/*
 * utils_cpu.cpp - utility function for noisy-cuda, CPU implementations
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
#include <cmath>

/**
 * @brief Simple 2D vector structure
 * 
 */
struct Vector2D {
    float x = 0.0f;
    float y = 0.0f;

    Vector2D() = default;

    Vector2D(float x_, float y_) : x(x_), y(y_) {}

    Vector2D operator-(const Vector2D& other) const {
        return {x - other.x, y - other.y};
    }

    float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    float length() const {
        return std::sqrt(x * x + y * y);
    }

    Vector2D normalize() const {
        float len = length();
        return len > 0 ? Vector2D(x / len, y / len) : Vector2D(0, 0);
    }
};


/**
 * @brief Smoothing function for Perlin noise
 * 
 * @param t 
 * @return float 
 */
static float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
/** 
 * @brief Linear interpolation
 * 
 * @param a 
 * @param b 
 * @param t 
 * @return float 
 */
static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}