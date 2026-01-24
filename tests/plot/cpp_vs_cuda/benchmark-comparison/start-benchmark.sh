#!/bin/bash

# output CSV file
OUTPUT_CSV="benchmark_v1.csv"

# CSV header
echo "dimension,CPP_time_ms,CUDA_time_ms" >"$OUTPUT_CSV"

# Parameters for generating dimensions
START=800
END=25000
STEP=200

# Dynamically generate the dimensions array
DIMENSIONS=()
for ((i=START; i<=END; i+=STEP)); do
  DIMENSIONS+=($i)
done

# Paths to the programs
CPP_PROG="./build/cpp/v1/noisy_cuda"
CUDA_PROG="./build/cuda/v1/noisy_cuda"

# Verify that the programs exist
if [ ! -f "$CPP_PROG" ] || [ ! -f "$CUDA_PROG" ]; then
  echo "Error: Make sure the files $CPP_PROG and $CUDA_PROG have been compiled."
  exit 1
fi

echo "Starting benchmark. Results will be saved in $OUTPUT_CSV"
echo "Parameters: START=$START, END=$END, STEP=$STEP"

# Loop through all dimensions
for SIZE in "${DIMENSIONS[@]}"; do
  SQUARE_SIZE="${SIZE}x${SIZE}"
  echo "--- Running benchmark for dimension: $SQUARE_SIZE ---"

  # --- CPP Version Benchmark ---
  # Run the CPP program and capture the output
  CPP_OUTPUT=$($CPP_PROG -n -v -s "$SQUARE_SIZE" | grep 'wall time')
  # Extract execution time in milliseconds (ms)
  # Example output: "wall time = 116.881 ms"
  CPP_TIME_MS=$(echo "$CPP_OUTPUT" | awk '{print $4}')

  if [ -z "$CPP_TIME_MS" ]; then
    echo "WARNING: Unable to extract CPP time for $SQUARE_SIZE. (Skipping)"
    continue
  fi

  # --- CUDA Version Benchmark ---
  # Run the CUDA program and capture the output
  # Note: your CUDA version provides "kernel time" and "total time".
  # We will use the "total time" in seconds (s) and convert it to ms.
  CUDA_OUTPUT=$($CUDA_PROG -n -v -s "$SQUARE_SIZE" | grep 'total time')
  # Extract total execution time in seconds (s)
  # Example output: "total time = 0.140 s"
  CUDA_TIME_S=$(echo "$CUDA_OUTPUT" | awk '{print $4}')

  if [ -z "$CUDA_TIME_S" ]; then
    echo "WARNING: Unable to extract CUDA time for $SQUARE_SIZE. (Skipping)"
    continue
  fi

  # Convert CUDA time from seconds to milliseconds for consistency
  # Using bc for floating point calculations
  CUDA_TIME_MS=$(echo "$CUDA_TIME_S * 1000" | bc -l)

  # Round CUDA_TIME_MS to 3 decimal places (optional for precision)
  # CUDA_TIME_MS=$(printf "%.3f" "$CUDA_TIME_MS")

  # Print and save results
  echo " Results: CPP: $CPP_TIME_MS ms | CUDA: $CUDA_TIME_MS ms"
  echo "$SIZE,$CPP_TIME_MS,$CUDA_TIME_MS" >>"$OUTPUT_CSV"
done

echo "Benchmark completed. Data saved in $OUTPUT_CSV"