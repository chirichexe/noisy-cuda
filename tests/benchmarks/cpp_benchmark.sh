#!/bin/bash

# Configuration
APPS=("./build/cpp/v2/noisy_cuda" "./build/cpp/v1/noisy_cuda")
RESOLUTIONS="128 983 1838 2694 3549 4404 5260 6115 6970 7826 8681 9536 10392 11247 12102 12958 13813 14668 15524 16384"
OUTPUT_DIR="./tests/csv/outputs"
CSV_HEADER="timestamp,width,height,pixels,octaves,frequency,wall_ms,cpu_s,ms_per_pixel,mem_bytes"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

for APP in "${APPS[@]}"; do
    # Extract version identifier (e.g., v1, v2) from path
    VERSION=$(echo "$APP" | grep -o "v[0-9]")
    OUTPUT_FILE="$OUTPUT_DIR/cpp_${VERSION}_output.csv"

    echo "Starting benchmark for $VERSION ($APP)..."
    
    # Initialize CSV file with header
    echo "$CSV_HEADER" > "$OUTPUT_FILE"

    for RES in $RESOLUTIONS; do
        echo "Running Resolution: ${RES}x${RES}"
        
        # Execute and append output to CSV
        "$APP" --no-output --benchmark --size "${RES}x${RES}" >> "$OUTPUT_FILE"
    done

    echo -e "Finished $VERSION. Results saved to $OUTPUT_FILE\n"
done