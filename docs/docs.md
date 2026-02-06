# NoisyCuda: Perlin Noise Generation with GPU Acceleration

## Introduction

NoisyCuda is a high-performance library for Perlin Noise generation, leveraging parallel processing on GPUs via CUDA and CPU optimization through SIMD instructions. The main goal is to significantly accelerate noise map production for procedural graphics, physical simulations, and rendering applications while maintaining high visual quality and noise coherence.

The approach includes parallelizing operations in CUDA and employing SIMD instructions to optimize vector calculations and reduce CPU bottlenecks. The project provides comparative benchmarks between traditional CPU implementations and GPU-accelerated versions, highlighting gains in throughput and scalability. The resulting implementation offers an efficient, modular tool easily integrated into real-time graphics pipelines and simulation software.

## Project Structure

The project is highly modular, allowing users to select at compile time the technology (pure CPU, CPU SIMD, not implemented yet, or CUDA) and version to use.

### Directory Organization

- **include/**: Contains main headers used for uniform Noise generation signatures and global data structures (e.g., Vector2D) for representing unit vectors of noise chunks.
- **src/**: Divided into three technology implementations, each with multiple versions. The first version implements all core functionality, while subsequent versions exploit technology-specific advantages to optimize throughput and execution time.

## Standard Perlin Noise Generation Algorithm

The algorithm follows these steps:

1. **Grid Definition**: Create a regular grid of control points, each associated with a random gradient vector
2. **Relative Coordinates**: Determine fractional coordinates (0-1) of each point relative to grid vertices
3. **Dot Product Calculation**: Calculate dot products between point vectors and gradient vectors for each vertex
4. **Fade Function**: Apply a fade function (6t⁵ - 15t⁴ + 10t³) to fractional coordinates for smooth transitions
5. **Interpolation**: Interpolate dot products using fade coordinates across axes
6. **Value Combination**: Obtain a single noise value, normalized to [-1, 1] range
7. **Full Area Repetition**: Repeat for all points in the target area to produce continuous, coherent noise maps
