import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

METRIC = "gpu_s"  # metrica di partenza (secondi)

def load_csv(path):
    if not os.path.exists(path):
        print(f"Errore: il file '{path}' non esiste.")
        sys.exit(1)

    df = pd.read_csv(path)

    required = {"octaves", "width", METRIC}
    if not required.issubset(df.columns):
        print(f"Errore: colonne mancanti in '{path}', richieste: {required}")
        sys.exit(1)

    df = df.sort_values(["octaves", "width"])

    # conversione secondi -> millisecondi
    df["gpu_ms"] = df[METRIC] * 1000.0

    return df


def main():
    if len(sys.argv) < 2:
        print("Utilizzo:")
        print("  python3 plot.py cuda_v1.csv cuda_v2.csv ...")
        sys.exit(1)

    files = sys.argv[1:]

    fig, ax = plt.subplots(figsize=(14, 6))

    x_positions = None
    x_labels = None
    octave_groups = None

    for csv_path in files:
        df = load_csv(csv_path)
        label = os.path.basename(csv_path).replace(".csv", "")

        df["x_pos"] = range(len(df))

        if x_positions is None:
            x_positions = df["x_pos"]
            x_labels = df["width"].astype(str)
            octave_groups = df.groupby("octaves")["x_pos"]

            # separatori verticali tra ottave
            prev_oct = None
            for i, octv in enumerate(df["octaves"]):
                if prev_oct is not None and octv != prev_oct:
                    ax.axvline(i - 0.5, linestyle="--", color="black", alpha=0.3)
                prev_oct = octv

        ax.plot(
            df["x_pos"],
            df["gpu_ms"],
            marker="o",
            linewidth=2,
            label=label
        )

    # Asse X: SOLO risoluzione
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)

    # Testo ottave in grassetto sotto
    for octv, idxs in octave_groups:
        center = (idxs.min() + idxs.max()) / 2
        ax.text(
            center,
            -0.12,
            f"{octv} octaves",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylabel("GPU Time (ms)")
    ax.set_title("GPU Performance vs Resolution (Grouped by Octaves)", fontweight="bold")

    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    output_png = "gpu_time_resolution_by_octaves.png"
    plt.savefig(output_png, dpi=300)

    print(f"Grafico salvato in: {output_png}")
    plt.show()


if __name__ == "__main__":
    main()
