import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) < 3:
        print("Utilizzo: python plot_comparison.py file_cpp.csv file_cuda.csv [output.png]")
        sys.exit(1)

    cpp_path, cuda_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "benchmark_plot.png"

    # Caricamento dati
    try:
        df_cpp = pd.read_csv(cpp_path)
        df_cuda = pd.read_csv(cuda_path)
    except Exception as e:
        print(f"Errore nella lettura dei file: {e}")
        sys.exit(1)

    # Dati per il plot
    res_labels = df_cpp['width'].astype(str)
    t_cpp = df_cpp['wall_ms']
    t_cuda = df_cuda['wall_ms']
    speedup = t_cpp / t_cuda

    # Setup grafico piccolo e denso
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # --- ASSE 1: Durata Esecuzione (Linee) ---
    ln1 = ax1.plot(res_labels, t_cpp, marker='o', color='#e74c3c', label='CPP Wall Time', linewidth=2)
    ln2 = ax1.plot(res_labels, t_cuda, marker='s', color='#2ecc71', label='CUDA Wall Time', linewidth=2)
    ax1.set_xlabel('Grid size (px)', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=9)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- ASSE 2: Speedup (Area Ombreggiata o Linea tratteggiata) ---
    ax2 = ax1.twinx()
    ln3 = ax2.plot(res_labels, speedup, color='#3498db', linestyle=':', label='Speedup (x)', alpha=0.7)
    ax2.fill_between(res_labels, speedup, 1, color='#3498db', alpha=0.1) # Ombreggiatura speedup
    ax2.set_ylabel('Speedup Factor (x)', fontsize=9, color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')

    # Unione Legende
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=8, frameon=True)

    plt.title(f"CPP vs CUDA: Performance Analysis", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Grafico salvato: {out_path} (Avg Speedup: {speedup.mean():.2f}x)")

if __name__ == "__main__":
    main()