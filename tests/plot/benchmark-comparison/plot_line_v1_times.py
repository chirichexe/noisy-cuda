import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# check if CSV file path is provided as argument
if len(sys.argv) < 2:
    print("Usage: python plot_line_v1_times.py <csv_file>")
    exit(1)

INPUT_CSV = sys.argv[1]
OUTPUT_PNG = "v1_comparison.png"

# verify CSV file existence
if not os.path.exists(INPUT_CSV):
    print(f"error: file '{INPUT_CSV}' not found.")
    exit(1)

# load data from CSV
try:
    df = pd.read_csv(INPUT_CSV)
except Exception as e:
    print(f"error reading CSV file: {e}")
    exit(1)

# convert times from milliseconds (ms) to seconds (s) for Y axis
df['CPP_Time_s'] = df['CPP_time_ms'] / 1000
df['CUDA_Time_s'] = df['CUDA_time_ms'] / 1000

# create the plot
plt.figure(figsize=(12, 6))

# plot line for CPP version
plt.plot(
    df['dimension'], 
    df['CPP_Time_s'], 
    label='C++ Version (CPU)', 
    marker='o', 
    linestyle='-', 
    color='blue'
)

# plot line for CUDA version
plt.plot(
    df['dimension'], 
    df['CUDA_Time_s'], 
    label='CUDA Version (GPU)', 
    marker='x', 
    linestyle='--', 
    color='red'
)

# add title and labels
plt.title('Performance Comparison: Perlin Noise Generation (CPU vs GPU)')
plt.xlabel('Grid Dimension (N x N)', fontsize=12)
plt.ylabel('Execution Time (Seconds)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# use logarithmic scale on Y axis if time difference is very large
# uncomment the lines below if CUDA line is nearly invisible:
# plt.yscale('log')
# plt.ylabel('Execution Time (Seconds) [Logarithmic Scale]', fontsize=12)

# set X axis ticks for better clarity
min_dim = int(df['dimension'].min())
max_dim = int(df['dimension'].max())
tick_span = 200

# create ticks: from min to max with the specified span
x_ticks = list(range(min_dim, max_dim + 1, tick_span))
# ensure max value is included
if x_ticks[-1] != max_dim:
    x_ticks.append(max_dim)

# set ticks without labels to avoid clutter
plt.xticks(x_ticks, [])

# add text annotation with info
info_text = f"Start: {min_dim} | End: {max_dim} | Span: {tick_span}"
plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 1])

# save the plot
plt.savefig(OUTPUT_PNG)

print(f"\nplot successfully created and saved as '{OUTPUT_PNG}'")