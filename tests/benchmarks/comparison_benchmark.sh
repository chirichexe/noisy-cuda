#!/bin/bash

# Definizioni degli eseguibili
CPP_APP="./build/cpp/v2/noisy_cuda"
CUDA_APP="./build/cuda/v3/noisy_cuda"

# Parametri di benchmark
RESOLUTIONS="5260 6115 6970 7826 8681 9536 10392 11247 12102 12958 13813 14668 15524"
OUTPUT_DIR="./tests/outputs/csv"
CSV_HEADER="timestamp,width,height,pixels,octaves,frequency,wall_ms,cpu_s,ms_per_pixel,mem_bytes"

# Creazione cartella di output
mkdir -p "$OUTPUT_DIR"

# File di output specifici
CPP_OUT="$OUTPUT_DIR/benchmark_cpp_v2.csv"
CUDA_OUT="$OUTPUT_DIR/benchmark_cuda_v3.csv"

# Inizializzazione file con header
echo "$CSV_HEADER" > "$CPP_OUT"
echo "$CSV_HEADER" > "$CUDA_OUT"

echo "--- Inizio Benchmark Comparativo: CPP vs CUDA ---"

for RES in $RESOLUTIONS; do
    SIZE="${RES}x${RES}"
    echo "[Testing $SIZE]"

    # Esecuzione versione C++
    if [ -f "$CPP_APP" ]; then
        echo "  > Esecuzione CPP (v2)..."
        "$CPP_APP" --no-output --benchmark --size "$SIZE" >> "$CPP_OUT"
    else
        echo "  ! Errore: $CPP_APP non trovato"
    fi

    # Esecuzione versione CUDA
    if [ -f "$CUDA_APP" ]; then
        echo "  > Esecuzione CUDA (v3)..."
        "$CUDA_APP" --no-output --benchmark --size "$SIZE" >> "$CUDA_OUT"
    else
        echo "  ! Errore: $CUDA_APP non trovato"
    fi
    
    echo "-------------------------------------------"
done

echo "Benchmark completati."
echo "Risultati C++: $CPP_OUT"
echo "Risultati CUDA: $CUDA_OUT"