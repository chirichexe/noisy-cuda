/*
 * options.hpp - definition of Options struct
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

#include <string>
#include <cstdint>

struct Options {
    int width = 0;
    int height = 0;
    int octaves = 1;
    std::string format = "png";
    std::string output_filename = "output.png";
    bool verbose = false;
    bool cpu_mode = false;
    std::uint64_t seed = 0;
    bool seed_provided = false;
};
