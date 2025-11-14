# noisy-cuda

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Platform](https://img.shields.io/badge/Platform-CPU%2FCUDA-green.svg)
![Output](https://img.shields.io/badge/Output-PNG%2FRAW%2FCSV%2FPPM-orange.svg)
![University](https://img.shields.io/badge/UniBO-Alma%20Mater%20Studiorum-lightgrey.svg)

A high-performance Perlin noise generator developed for the "Accelerated Processing Systems - M" course at University of Bologna, with the professors **Stefano Mattoccia** and **Fabio Tosi**.

## Features

- **Multi-platform**: CPU and CUDA GPU acceleration support
- **Flexible output**: Multiple output formats (PNG, RAW, CSV, PPM)
- **Customizable parameters**: Adjustable octaves, image size, and seed
- **High performance**: Optimized noise generation algorithms

## Available Backends and Versions

| Backend | Available Versions | Description                           | Dependencies |
|---------|--------------------|---------------------------------------|--------------|
| CUDA    | v1                 | It uses  CUDA implementation          | CUDA         |
| SIMD    | v1, v2             | It uses SIMD instructions for ISA x86 | SSE4         |
| CPP     | v1                 | It is the naive C++ Implementation    | C++ compiler |

> Note: The `latest` keyword automatically selects the highest available version.

## Building

This project supports both CPU and GPU acceleration. CUDA toolkit is required for GPU acceleration, but you can also build for CPU-only execution.

### Build Tutorial

```sh
# Build CUDA backend
cmake -B build_cuda -DUSE_CUDA=ON
cmake --build build_cuda

# Build SIMD backend version v1
cmake -B build_simd -DUSE_SIMD=ON -DSIMD_VERSION=v1
cmake --build build_simd

# Build C++ backend (CPU only)
cmake -B build_cpp -DUSE_CPP=ON
cmake --build build_cpp
```

> Tip: Use separate build directories per backend to avoid conflicts and unnecessary recompilation.

## Command-Line Options

| Flag | Long Option | Argument | Default | Description |
|------|--------------|-----------|----------|-------------|
| `-h` | `--help` | none | — | Displays the help message and exits. |
| `-s` | `--size` | `<WxH>` | `2048x2048` | Sets the image resolution in pixels, where `W` = width and `H` = height. |
| `-O` | `--octaves` | `<int>` | `1` | Defines the number of noise layers (octaves) to combine. Higher values add more detail. Must be >= 1. |
| `-F` | `--frequency` | `<float>` | `50.0` | Sets the base spatial frequency (scale factor). Higher values produce more dense variations. |
| `-A` | `--amplitude` | `<float>` | `1.0` | Controls the base intensity (height) of the noise values. |
| `-L` | `--lacunarity` | `<float>` | `2.0` | Multiplies frequency between successive octaves. Controls how quickly detail frequency increases. |
| `-P` | `--persistence` | `<float>` | `0.5` | Multiplies amplitude between successive octaves. Controls how quickly amplitude decreases. |
| `-f` | `--format` | `<string>` | `png` | Specifies the output format. Supported values: `png`, `raw`, `csv`, `ppm`. |
| `-o` | `--output` | `<filename>` | `perlin.<ext>` | Defines the output filename. The extension is inferred from the chosen format. |
| `-v` | `--verbose` | none | `false` | Prints detailed processing steps and timing information. |
| `-S` | `--seed` | `<uint64>` | `0` | Sets a specific random seed for reproducible noise. |
| *positional* | *(seed)* | `<uint64>` | — | Alternative way to provide the seed (e.g., `./perlin 12345`). Cannot be combined with `--seed`. |


## Output formats

- `PNG`: Portable Network Graphics (lossless compression)
- `RAW`: Raw binary data
- `CSV`: Comma-separated values

## Start profiling session

```sh
ncu -o cuda --target-processes all ./build/cuda/noisy_cuda
```

## License

This project is released under the **Apache License 2.0**. See LICENSE file for details.
