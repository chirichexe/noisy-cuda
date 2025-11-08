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

| Backend | Available Versions | Description                           | Dependencies |   |
|---------|--------------------|---------------------------------------|--------------|---|
| CUDA    | v1                 | It uses  CUDA implementation          | CUDA         |   |
| SIMD    | v1, v2             | It uses SIMD instructions for ISA x86 | SSE4         |   |
| CPP     | v1                 | It is the naive C++ Implementation    | C++ compiler |   |

> Note: The `latest` keyword automatically selects the highest available version.

## Building

This project supports both CPU and GPU acceleration. CUDA toolkit is required for GPU acceleration, but you can also build for CPU-only execution.

### Build Tutorial

```sh
# Build CUDA backend
cmake -B build_cuda -DUSE_CUDA=ON
cmake --build build_cuda

# Build SIMD backend version v1
cmake -B build_simd_v1 -DUSE_SIMD=ON -DSIMD_VERSION=v1
cmake --build build_simd_v1

# Build C++ backend (CPU only)
cmake -B build_cpp -DUSE_CPP=ON
cmake --build build_cpp
```

> Tip: Use separate build directories per backend to avoid conflicts and unnecessary recompilation.

## Command-Line Options

```
-h, --help         - Show help message and exit
-s, --size <WxH>   - Image size in pixels (width x height). Default: 2048x2048
-O, --octaves <int> - Number of octaves (>=1). Default: 1
-f, --format <string> - Output format: png, raw, csv, ppm. Default: png
-o, --output <filename> - Output filename. Default: perlin.<ext>
-v, --verbose      - Print processing steps and timings
-S, --seed <uint64> - Provide explicit seed (alternative to positional)
```

## Output formats

- `PNG`: Portable Network Graphics (lossless compression)
- `RAW`: Raw binary data
- `CSV`: Comma-separated values

## License

This project is released under the **Apache License 2.0**. See LICENSE file for details.
