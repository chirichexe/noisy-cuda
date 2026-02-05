#!/bin/bash

# Check parameter
if [ -z "$1" ]; then
    echo "Usage: $0 <VERSIONE>"
    exit 1
fi

VERSION="$1"

# Configuration
APP="./build/cuda/v${VERSION}/noisy_cuda"
#RESOLUTIONS="512 1024 2048 4096 8192 16384 "
RESOLUTIONS="512 6272 8832 10880 13184 15360"
OCTAVES="1 2 4 16 32 64 128" 
OUTPUT_DIR="./tests/outputs/csv"
OUTPUT_FILE="${OUTPUT_DIR}/cuda_v${VERSION}_output.csv"
CSV_HEADER="timestamp,width,height,pixels,octaves,frequency,wall_ms,gpu_s,ms_per_pixel,mem_bytes"

# Sanity checks
if [ ! -x "$APP" ]; then
    echo "Error: executable not found or not executable: $APP"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Starting CUDA benchmark for version v${VERSION}"
echo "Executable: $APP"
echo "Output: $OUTPUT_FILE"

# Initialize CSV file with header
echo "$CSV_HEADER" > "$OUTPUT_FILE"

for OCTAVE in $OCTAVES; do
    echo "Running octave: ${OCTAVE}"
    for RES in $RESOLUTIONS; do
        echo "Running resolution: ${RES}x${RES}"

        "$APP" \
            --no-output \
            --benchmark \
            --size "${RES}x${RES}" \
            -C "${OCTAVE}" >> "$OUTPUT_FILE"
    done
done

echo "Benchmark completed. Results saved to $OUTPUT_FILE"