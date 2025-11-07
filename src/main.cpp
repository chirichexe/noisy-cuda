/*
 * main.cpp - main function of noisy-cuda
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

#include "perlin_noise.hpp"
#include "options.hpp"
#include <iostream>

int main(int argc, char** argv) {

    try {
        Options opts = parse_options(argc, argv);

        if (opts.verbose) {
            print_program_options(opts);
        }

        generate_perlin_noise(opts);
        
        std::cout << "Done!" << std::endl;
        
    } 
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << ".\nFor more information, try '--help'." << '\n';
        return 1;
    }

    return 0;
}
