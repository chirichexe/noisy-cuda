#!/bin/bash

CPP_APP="./build/cpp/v2/noisy_cuda"
CUDA_APP="./build/cuda/v3/noisy_cuda"

RESOLUTIONS="5260 6115 6970 7826 8681 9536 10392 11247 12102 12958 13813 14668 15524"
OUTPUT_DIR="./tests/outputs/csv"
CSV_HEADER="timestamp,width,height,pixels,octaves,frequency,wall_ms,cpu_s,ms_per_pixel,mem_bytes"

mkdir -p "$OUTPUT_DIR"

CPP_OUT="$OUTPUT_DIR/benchmark_cpp_v2.csv"
CUDA_OUT="$OUTPUT_DIR/benchmark_cuda_v3.csv"

echo "$CSV_HEADER" > "$CPP_OUT"
echo "$CSV_HEADER" > "$CUDA_OUT"


for RES in $RESOLUTIONS; do
    SIZE="${RES}x${RES}"
    echo "[Testing $SIZE]"

    if [ -f "$CPP_APP" ]; then
        echo "  > Executing CPP (v2)..."
        "$CPP_APP" --no-output --benchmark --size "$SIZE" >> "$CPP_OUT"
    else
        echo "  ! Error: $CPP_APP not found"
    fi

    # Executing CUDA version
    if [ -f "$CUDA_APP" ]; then
        echo "  > Executing CUDA (v3)..."
        "$CUDA_APP" --no-output --benchmark --size "$SIZE" >> "$CUDA_OUT"
    else
        echo "  ! Error: $CUDA_APP not found"
    fi

done

echo "Benchmark completed."
echo "C++ results: $CPP_OUT"
echo "CUDA results: $CUDA_OUT"