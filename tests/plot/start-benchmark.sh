#!/bin/bash

# File di output CSV
OUTPUT_CSV="benchmark_v1.csv"

# Intestazione del file CSV
echo "Dimensione,Tempo_CPP_ms,Tempo_CUDA_ms" >"$OUTPUT_CSV"

# Definisci le dimensioni della griglia da testare (da 800 a 25000)
# Aggiungi qui le dimensioni che preferisci per un'analisi dettagliata.
DIMENSIONS=(
  800
  1000
  2000
  4000
  6000
  8000
  10000
  15000
  20000
  25000
)

# Percorsi ai programmi
CPP_PROG="./build/cpp/v1/noisy_cuda"
CUDA_PROG="./build/cuda/v1/noisy_cuda"

# Verifica che i programmi esistano
if [ ! -f "$CPP_PROG" ] || [ ! -f "$CUDA_PROG" ]; then
  echo "Errore: Assicurati che i file $CPP_PROG e $CUDA_PROG siano stati compilati."
  exit 1
fi

echo "Avvio del benchmark. I risultati verranno salvati in $OUTPUT_CSV"

# Loop su tutte le dimensioni
for SIZE in "${DIMENSIONS[@]}"; do
  SQUARE_SIZE="${SIZE}x${SIZE}"
  echo "--- Esecuzione per dimensione: $SQUARE_SIZE ---"

  # --- Benchmark Versione CPP ---
  # Esegui il programma CPP e cattura l'output
  CPP_OUTPUT=$($CPP_PROG -n -v -s "$SQUARE_SIZE" | grep 'wall time')
  # Estrai il tempo di esecuzione in millisecondi (ms)
  # Esempio di output: "wall time = 116.881 ms"
  CPP_TIME_MS=$(echo "$CPP_OUTPUT" | awk '{print $4}')

  if [ -z "$CPP_TIME_MS" ]; then
    echo "AVVISO: Impossibile estrarre il tempo CPP per $SQUARE_SIZE. (Skipping)"
    continue
  fi

  # --- Benchmark Versione CUDA ---
  # Esegui il programma CUDA e cattura l'output
  # Si noti che la tua versione CUDA fornisce "kernel time" e "total time".
  # Utilizzeremo il "total time" in secondi (s) e lo convertiremo in ms.
  CUDA_OUTPUT=$($CUDA_PROG -n -v -s "$SQUARE_SIZE" | grep 'total time')
  # Estrai il tempo totale di esecuzione in secondi (s)
  # Esempio di output: "total time = 0.140 s"
  CUDA_TIME_S=$(echo "$CUDA_OUTPUT" | awk '{print $4}')

  if [ -z "$CUDA_TIME_S" ]; then
    echo "AVVISO: Impossibile estrarre il tempo CUDA per $SQUARE_SIZE. (Skipping)"
    continue
  fi

  # Converti il tempo CUDA da secondi a millisecondi per uniformità
  # Usiamo bc per calcoli in virgola mobile
  CUDA_TIME_MS=$(echo "$CUDA_TIME_S * 1000" | bc -l)

  # Arrotonda CUDA_TIME_MS a 3 decimali (o mantieni così se preferisci la precisione)
  # CUDA_TIME_MS=$(printf "%.3f" "$CUDA_TIME_MS") # Se vuoi arrotondare

  # Stampa e salva i risultati
  echo " Risultati: CPP: $CPP_TIME_MS ms | CUDA: $CUDA_TIME_MS ms"
  echo "$SIZE,$CPP_TIME_MS,$CUDA_TIME_MS" >>"$OUTPUT_CSV"
done

echo "Benchmark completato. Dati salvati in $OUTPUT_CSV"
