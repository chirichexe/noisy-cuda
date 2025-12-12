import pandas as pd
import matplotlib.pyplot as plt
import os

# Nome del file di input CSV
INPUT_CSV = "benchmark_v1.csv"
# Nome del file di output del grafico
OUTPUT_PNG = "v1_confronto.png"

# Verifica l'esistenza del file CSV
if not os.path.exists(INPUT_CSV):
    print(f"Errore: File '{INPUT_CSV}' non trovato. Esegui prima lo script shell.")
    exit()

# Carica i dati dal CSV
try:
    df = pd.read_csv(INPUT_CSV)
except Exception as e:
    print(f"Errore durante la lettura del file CSV: {e}")
    exit()

# Converti i tempi da millisecondi (ms) a secondi (s) per l'asse Y
# (Il benchmark script salva in ms, il tuo requisito è in secondi)
df['Tempo_CPP_s'] = df['Tempo_CPP_ms'] / 1000
df['Tempo_CUDA_s'] = df['Tempo_CUDA_ms'] / 1000

# Inizia a creare il grafico
plt.figure(figsize=(12, 6))

# Trama la linea per la versione CPP
plt.plot(
    df['Dimensione'], 
    df['Tempo_CPP_s'], 
    label='Versione C++ (CPU)', 
    marker='o', 
    linestyle='-', 
    color='blue'
)

# Trama la linea per la versione CUDA
plt.plot(
    df['Dimensione'], 
    df['Tempo_CUDA_s'], 
    label='Versione CUDA (GPU)', 
    marker='x', 
    linestyle='--', 
    color='red'
)

# Aggiungi titolo ed etichette
plt.title('Confronto Prestazioni Generazione Perlin Noise (CPU vs GPU)')
plt.xlabel('Dimensione della Griglia (N x N)', fontsize=12)
plt.ylabel('Tempo di Esecuzione (Secondi)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Usa scala logaritmica sull'asse Y se la differenza di tempo è molto grande
# Se noti che la linea CUDA è quasi invisibile, decommenta la riga sotto:
# plt.yscale('log')
# plt.ylabel('Tempo di Esecuzione (Secondi) [Scala Logaritmica]', fontsize=12)


# Imposta i tick dell'asse X per maggiore chiarezza
plt.xticks(df['Dimensione'], rotation=45)
plt.tight_layout()

# Salva il grafico
plt.savefig(OUTPUT_PNG)

print(f"\nGrafico creato con successo e salvato come '{OUTPUT_PNG}'")