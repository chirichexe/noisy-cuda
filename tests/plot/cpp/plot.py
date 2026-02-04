import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def load_and_clean(path):
    if not os.path.exists(path):
        print(f"Errore: Il file '{path}' non esiste.")
        sys.exit(1)
    df = pd.read_csv(path)
    return df.sort_values('pixels')

def main():
    if len(sys.argv) < 3:
        print("Errore: Devi specificare due file CSV.")
        print("Utilizzo: python3 script.py versione1.csv versione2.csv")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Caricamento e pulizia
    df1 = load_and_clean(file1)
    df2 = load_and_clean(file2)

    # Setup figura
    fig, ax = plt.subplots(figsize=(11, 6))

    # Label basate sui nomi dei file
    label1 = os.path.basename(file1).replace('.csv', '')
    label2 = os.path.basename(file2).replace('.csv', '')

    # Grafico CPU Time (Confronto)
    ax.plot(df1['width'], df1['cpu_s'], marker='o', linestyle='--', color='#95a5a6', linewidth=2, label=f'{label1} (Original)')
    ax.plot(df2['width'], df2['cpu_s'], marker='o', linestyle='-', color='#e67e22', linewidth=2, label=f'{label2} (Optimized)')
    
    ax.set_title('Performance CPU: V1 vs V2', fontsize=14, fontweight='bold')
    ax.set_xlabel('Larghezza Immagine (px)')
    ax.set_ylabel('Tempo CPU (secondi)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # Salvataggio
    output_png = f"comparison_{label1}_vs_{label2}.png"
    plt.tight_layout()
    plt.savefig(output_png)
    
    print(f"✅ Grafico di confronto salvato come: {output_png}")
    
    # Calcolo rapido dello speedup medio
    if len(df1) == len(df2):
        avg_speedup = (df1['cpu_s'].mean() / df2['cpu_s'].mean())
        print(f"Speedup medio calcolato: {avg_speedup:.2f}x")

    plt.show()

if __name__ == "__main__":
    main()
