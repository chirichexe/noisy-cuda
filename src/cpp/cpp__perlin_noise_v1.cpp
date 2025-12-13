/*
 * cpp__perlin_noise.cpp - perlin noise: C++ implementation
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
 * Distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "perlin_noise.hpp"
#include "utils_global.hpp"
#include "utils_cpu.hpp"


#include <algorithm>
#include <inttypes.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>

// Future ideas
/*
    GENERAL:
    - Add the permutations instead of gradients giant matrix
    - Check for variable types used on algorithm 
      (for example, float, char ... ):

      - float per Perlin noise: scelta corretta
      - int -> size_t per indici
      - Random: rand() funziona ma è subottimale; meglio std::mt19937 + uniform_real_distribution<float> per qualità e determinism

    - Limit the size of the variables of the image to avoid 
      crash or too large outputs
    
    IMPROVEMENTS
      - buffer[...] += value * amplitude; 
      può saturare prima del clamp e perdere precisione nelle ottave alte che hanno amp piccola
      soluzioni: 
        - normalizzare alla fine con un massimo teorico (somma delle ampiezze)

        float max_amplitude = 0.0f;
        float amplitude = base_amplitude;

        for (...) {
            max_amplitude += amplitude;
            amplitude *= persistence;
        }

        // dopo
        v /= max_amplitude;

        - normalizzare ogni ottava in [-1,1] prima di sommare all'accumulatore 
    
        - usa size_t per indici di vettori e prodotti width*height

        - sistema VEctor2d: due float sono ideali; eliminare funzioni inutili, rendere i metodi inline, minimizzare sqrt.

        struct alignas(8) Vector2D {
            float x;
            float y;

            inline Vector2D() = default;
            inline Vector2D(float x_, float y_) : x(x_), y(y_) {}

            inline float dot(const Vector2D& o) const {
                return x * o.x + y * o.y;
            }

            inline Vector2D normalized() const {
                float len2 = x * x + y * y;
                if (len2 > 0.0f) {
                    float inv_len = 1.0f / std::sqrt(len2);
                    return { x * inv_len, y * inv_len };
                }
                return { 0.0f, 0.0f };
            }
        };

    // NOTES
    // PERCHÈ LA 

    Usare una LUT di gradienti significa sostituire la generazione casuale seguita da normalizzazione con un insieme finito di vettori già unitari e ben distribuiti. Questo elimina completamente la necessità di calcolare norme, divisioni e radici quadrate, che sono operazioni relativamente costose e difficili da ottimizzare in modo aggressivo dal compilatore. Il vantaggio non è solo aritmetico: il codice diventa più semplice e più prevedibile dal punto di vista delle prestazioni, perché il costo per gradiente si riduce a un accesso in memoria e a un prodotto scalare.

    Dal punto di vista numerico, una LUT consente di controllare esplicitamente la distribuzione angolare dei gradienti. Nel Perlin noise la qualità visiva dipende fortemente dall’isotropia del campo di gradienti; gradienti casuali normalizzati introducono inevitabilmente piccole anisotropie e bias statistici, mentre una LUT costruita su direzioni equispaziate sul cerchio garantisce simmetria e uniformità. Questo si traduce in un rumore più “pulito”, con meno artefatti direzionali e pattern ripetuti.

    C’è poi un aspetto di robustezza e portabilità. Con una LUT non dipendi dal comportamento di rand(), dalle differenze tra implementazioni, né da dettagli di floating point come underflow o precisione della normalizzazione. L’output diventa deterministico a parità di seed e dimensioni, e il codice non contiene operazioni delicate o potenzialmente problematiche dal punto di vista dello standard C++.

    Infine, l’uso di una LUT favorisce l’ottimizzazione a basso livello. Gli accessi ai gradienti sono regolari, facilmente prefetchabili e ben compatibili con la vectorization automatica. In un algoritmo come il tuo, dove il costo dominante è nel loop per pixel, questo tipo di regolarità è più prezioso di micro-ottimizzazioni aritmetiche isolate. In sintesi, la LUT sposta il problema da calcoli numerici ripetuti a una scelta strutturale migliore, con benefici simultanei su prestazioni, qualità e affidabilità.

    // SQRT DA TOGLIERE

    sqrt “pesa” perché, a livello di CPU, non è un’operazione elementare come + o *.

    Una somma o una moltiplicazione in floating point vengono eseguite in 1–3 cicli e possono essere pipelinate e vectorizzate molto facilmente. La radice quadrata, invece, richiede un algoritmo iterativo hardware (basato su approssimazioni successive) e viene implementata da unità funzionali dedicate, molto più lente e con latenza elevata. Su CPU moderne una sqrt in single precision ha tipicamente una latenza dell’ordine di 10–20 cicli, contro 3–5 cicli di una moltiplicazione.

    Inoltre, la sqrt ha throughput limitato: anche se la pipeline è piena, la CPU può emettere poche radici per ciclo, mentre può emettere molte moltiplicazioni o addizioni in parallelo. Questo la rende un collo di bottiglia nei loop numerici intensivi.

    C’è poi un aspetto di ottimizzazione del compilatore. Operazioni semplici vengono facilmente fuse, riordinate e vectorizzate; sqrt invece è una barriera più “rigida”, che limita il riordino delle istruzioni e spesso impedisce una vectorization aggressiva, soprattutto se seguita da divisioni dipendenti dal risultato.

    Infine, dal punto di vista energetico e micro-architetturale, sqrt attiva unità più complesse della FPU, consumando più risorse rispetto a operazioni aritmetiche di base. Per questo, in algoritmi come il tuo, si tende a ridurne l’uso al minimo indispensabile o a spostarlo fuori dai loop critici.

    // PERCHÈ USARE LE PERMUTAZIONI

    Le permutazioni servono a introdurre pseudo-casualità deterministica nella scelta dei gradienti partendo da una griglia regolare. Trasformano le coordinate intere in un indice apparentemente casuale, evitando pattern ripetitivi e allineamenti visibili nel noise. Allo stesso tempo garantiscono che lo stesso punto produca sempre lo stesso risultato, rendendo il rumore stabile e riproducibile. Sono usate perché sono semplici, veloci e permettono di controllare la periodicità del noise senza ricorrere a generatori casuali costosi.

*/

/* chunk variables */
#define CHUNK_SIDE_LENGTH 32

/**
 * @brief Chunk: represents a square section of the noise map
 * 
 */
struct Chunk {
    int chunk_x = 0;
    int chunk_y = 0;

    Chunk(int cx, int cy) : chunk_x(cx), chunk_y(cy) {}

    void generate_chunk_pixels(
        std::vector<float>& buffer, // reference to global float buffer
        int image_width,
        int image_height,
        const std::vector<std::vector<Vector2D>>& gradients,
        int chunks_count_x,
        int chunks_count_y,
        float frequency,
        float amplitude,
        int offset_x,
        int offset_y
    ) const {
        int start_x = chunk_x * CHUNK_SIDE_LENGTH;
        int start_y = chunk_y * CHUNK_SIDE_LENGTH;

        int end_x = std::min(start_x + CHUNK_SIDE_LENGTH, image_width);
        int end_y = std::min(start_y + CHUNK_SIDE_LENGTH, image_height);

        for (int y = start_y; y < end_y; y++) {
            for (int x = start_x; x < end_x; x++) {

                // normalized coordinates scaled by frequency
                float lerp_coeff = std::max(image_width, image_height);
                float fx = ((float)(x + offset_x) / (float)lerp_coeff) * frequency;
                float fy = ((float)(y + offset_y) / (float)lerp_coeff) * frequency;

                // integer grid cell (floor handles negative values correctly)
                int x0 = (int)std::floor(fx);
                int y0 = (int)std::floor(fy);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                // local coordinates within the cell
                float sx = fx - (float)x0;
                float sy = fy - (float)y0;

                // corner gradients (shared from global grid)
                // Use positive modulo for proper wrapping with negative indices
                int gx0 = ((x0 % chunks_count_x) + chunks_count_x) % chunks_count_x;
                int gy0 = ((y0 % chunks_count_y) + chunks_count_y) % chunks_count_y;
                int gx1 = ((x1 % chunks_count_x) + chunks_count_x) % chunks_count_x;
                int gy1 = ((y1 % chunks_count_y) + chunks_count_y) % chunks_count_y;

                const Vector2D& g00 = gradients[gx0][gy0];  // tl
                const Vector2D& g10 = gradients[gx1][gy0];  // tr
                const Vector2D& g01 = gradients[gx0][gy1];  // bl
                const Vector2D& g11 = gradients[gx1][gy1];  // br

                // distance vectors
                Vector2D d00(sx,     sy);
                Vector2D d10(sx - 1, sy);
                Vector2D d01(sx,     sy - 1);
                Vector2D d11(sx - 1, sy - 1);

                // dot products
                float dot00 = g00.dot(d00);
                float dot10 = g10.dot(d10);
                float dot01 = g01.dot(d01);
                float dot11 = g11.dot(d11);

                // fade curves
                float u = fade(sx);
                float v = fade(sy);

                // bilinear interpolation
                float nx0 = lerp(dot00, dot10, u);
                float nx1 = lerp(dot01, dot11, u);
                float value = lerp(nx0, nx1, v);

                // multiply by amplitude for THIS octave
                buffer[y * image_width + x] += value * amplitude;
            }
        }
    }
};

void generate_perlin_noise(const Options& opts) {

    /* initialize parameters */
    // noise parameters
    std::uint64_t seed = opts.seed;
    int width = opts.width;
    int height = opts.height;
    float base_frequency = opts.frequency;
    float base_amplitude = opts.amplitude;
    int octaves = opts.octaves;
    int lacunarity = opts.lacunarity;
    float persistence = opts.persistence;
    int offset_x = opts.offset_x;
    int offset_y = opts.offset_y;
    bool no_outputs = opts.no_outputs;
    bool verbose = opts.verbose;

    // output info
    std::string output_filename = opts.output_filename;
    std::string output_format = opts.format;

    /* randomize from seed */
    srand(seed);

    /* start profiling timers */
    std::chrono::high_resolution_clock::time_point wall_start;
    clock_t cpu_start = 0;
    if (verbose) {
        wall_start = std::chrono::high_resolution_clock::now();
        cpu_start = std::clock();
    }

    /* float accumulation buffer (needed for octaves) */
    std::vector<float> accumulator(width * height, 0.0f);

    /* calculate chunk grid */
    int chunks_count_x = (width  + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;
    int chunks_count_y = (height + CHUNK_SIDE_LENGTH - 1) / CHUNK_SIDE_LENGTH;

    /* initialize gradient vectors */
    std::vector<std::vector<Vector2D>> gradients(chunks_count_x, std::vector<Vector2D>(chunks_count_y));

    for (int gx = 0; gx < chunks_count_x; gx++) {
        for (int gy = 0; gy < chunks_count_y; gy++) {
            float rx = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float ry = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            gradients[gx][gy] = Vector2D(rx, ry).normalize();
        }
    }

    /* octave loop */
    float frequency = base_frequency;
    float amplitude = base_amplitude;

    for (int o = 0; o < octaves; o++) {

        // generate noise for this octave using the existing chunk pipeline
        for (int cy = 0; cy < chunks_count_y; cy++) {
            for (int cx = 0; cx < chunks_count_x; cx++) {
                Chunk chunk(cx, cy);
                
                chunk.generate_chunk_pixels(
                    accumulator,
                    width,
                    height,
                    gradients,
                    chunks_count_x,
                    chunks_count_y,
                    frequency,
                    amplitude,
                    offset_x,
                    offset_y
                );
            }
        }

        // prepare next octave (standard FBM rules)
        // https://medium.com/@logan.margo314/procedural-generation-using-fractional-brownian-motion-b35b7231309f
        frequency *= lacunarity;   // controls frequency growth
        amplitude *= persistence;  // controls amplitude decay
    }

    /* stop profiling timers and report */
    if (verbose) {
        clock_t cpu_end = std::clock();
        auto wall_end = std::chrono::high_resolution_clock::now();

        double cpu_ticks = static_cast<double>(cpu_end - cpu_start);
        double cpu_seconds = cpu_ticks / static_cast<double>(CLOCKS_PER_SEC);
        double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_end - wall_start).count();

        size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        double ms_per_pixel = (num_pixels > 0) ? (wall_ms / (double)num_pixels) : 0.0;

        size_t gradients_bytes = (size_t)chunks_count_x * (size_t)chunks_count_y * sizeof(Vector2D);
        size_t accumulator_bytes = accumulator.size() * sizeof(float);
        size_t estimated_total_alloc = gradients_bytes + accumulator_bytes;

        printf("\nProfiling:\n");
        printf("  wall time        = %.3f ms\n", wall_ms);
        printf("  cpu time         = %.6f s (clock ticks = %.0f)\n", cpu_seconds, cpu_ticks);
        printf("  time / pixel     = %.6f ms\n", ms_per_pixel);
        printf("  chunks           = %dx%d (total %d)\n", chunks_count_x, chunks_count_y, chunks_count_x * chunks_count_y);
        printf("  mem (approx)     = %zu bytes (gradients %zu + accumulator %zu)\n",
               estimated_total_alloc, gradients_bytes, accumulator_bytes);
        printf("\n");

    }

    /* convert accumulator to final 0-255 output */
    unsigned int channels = 1;
    std::vector<unsigned char> output(width * height * channels, 0);

    for (int i = 0; i < width * height; i++) {

        
        // normalize fractal sum back to [-1,1]
        float v = accumulator[i];
        
        v = std::clamp(v, -1.0f, 1.0f);
        
        output[i] = static_cast<unsigned char>((v + 1.0f) * 0.5f * 255.0f);
    }

    /* save the generated noise image */
    if (!no_outputs){
        save_output(
            output,
            width,
            height,
            channels,
            output_filename,
            output_format
        );
    }

}