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
|------|-------------|----------|---------|-------------|
| `-h` | `--help` | none | — | Show this help message and exit. |
| — | `--version` | none | — | Show program version and exit. |
| `-o` | `--output` | `<filename>` | `perlin.<ext>` | Output filename. Extension inferred from format. |
| `-f` | `--format` | `<string>` | `png` | Output format. Supported: `png`, `raw`, `csv`, `ppm`. |
| `-s` | `--size` | `<WxH>` | `2048x2048` | Output size in pixels (width x height). |
| `-v` | `--verbose` | none | `false` | Print processing steps and timings. |
| `-F` | `--frequency` | `<float>` | `1.0` | Base frequency (scale factor). |
| `-A` | `--amplitude` | `<float>` | `1.0` | Base amplitude. |
| `-L` | `--lacunarity` | `<float>` | `2.0` | Frequency multiplier per octave. |
| `-P` | `--persistence` | `<float>` | `0.5` | Amplitude multiplier per octave. |
| `-O` | `--offset` | `<x,y>` | `0,0` | Offset applied to noise coordinates. |
| `-C` | `--octaves` | `<int>` | `1` | Number of octaves (>= 1). |
| *positional* | *(seed)* | `<uint64>` | — | Positional unsigned integer seed (e.g. `./perlin 13813`). |

Usage: `./perlin [seed] [OPTIONS]`

## Output formats

- `PNG`: Portable Network Graphics (lossless compression)
- `RAW`: Raw binary data
- `CSV`: Comma-separated values

## Start profiling session

```sh
ncu -o cuda --target-processes all ./build/cuda/noisy_cuda
```

# Examples

Below are four example terrains generated with Perlin noise using seed `1234`.  
The output images are located in `docs/examples/png/`.

<table>
<tr>
  <td align="center">
    <img src="docs/examples/png/terrain_1.png" width="200"><br>
    <b>Terrain 1</b><br>
    Freq: 1.0<br>
    Amp: 1.0<br>
    Lac: 2.0<br>
    Pers: 0.5<br>
    Octaves: 4
  </td>
  <td align="center">
    <img src="docs/examples/png/terrain_2.png" width="200"><br>
    <b>Terrain 2</b><br>
    Freq: 2.0<br>
    Amp: 1.0<br>
    Lac: 2.5<br>
    Pers: 0.6<br>
    Octaves: 5
  </td>
  <td align="center">
    <img src="docs/examples/png/terrain_3.png" width="200"><br>
    <b>Terrain 3</b><br>
    Freq: 4.0<br>
    Amp: 1.0<br>
    Lac: 2.0<br>
    Pers: 0.7<br>
    Octaves: 6
  </td>
  <td align="center">
    <img src="docs/examples/png/terrain_4.png" width="200"><br>
    <b>Terrain 4</b><br>
    Freq: 0.5<br>
    Amp: 1.0<br>
    Lac: 2.0<br>
    Pers: 0.4<br>
    Octaves: 3
  </td>
</tr>
</table>


## License

This project is released under the **Apache License 2.0**. See LICENSE file for details.
