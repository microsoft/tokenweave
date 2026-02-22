import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os

indir = sys.argv[1]
outdir = sys.argv[2]
csvdir = sys.argv[3]

data_path = os.path.join(indir, 'figure_5_hs_8192.csv')
df = pd.read_csv(data_path)

fig, ax = plt.subplots(figsize=(12, 6), dpi=1200)
sns.set_style("whitegrid")
colors = sns.color_palette("husl", 5)
line1, = ax.plot(df["BL"], df["baseline_us"], color=colors[0], label="No Split", linestyle='-', alpha=0.7, linewidth=2)
line2, = ax.plot(df["BL"], df["smart_us"], color=colors[1], label="Smart Split", linestyle='-', alpha=1, linewidth=2)
line3, = ax.plot(df["BL"], df["offset_0"], color=colors[2], label="Two Equal Splits", linestyle='-', alpha=0.7, linewidth=2)
ax.set_xlabel("Seq Length", fontsize=28, fontweight='bold')
ax.set_ylabel("Time (us)", fontsize=28, fontweight='bold')
ax2 = ax.twinx()
normalized_smart = df["smart_us"] / df["baseline_us"]
normalized_offset_0 = df["offset_0"] / df["baseline_us"]

line4, = ax2.plot(df["BL"], normalized_smart, color=colors[3], label="Smart Split (Normalized)", linestyle='--', alpha=1, linewidth=2)
line5, = ax2.plot(df["BL"], normalized_offset_0, color=colors[4], label="Two Equal Splits (Normalized)", linestyle='--', alpha=0.7, linewidth=2)

ax2.set_ylabel("Normalized wrt No Split", fontsize=20, fontweight='bold')
lines = [line1, line2, line3, line4, line5]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, fontsize=20, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.12), frameon=False)
xticks = [1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=22, rotation=45)
ax.tick_params(axis='y', labelsize=25)
ax2.tick_params(axis='y', labelsize=25)

ax.grid(True, linestyle='--', linewidth=0.5)
pdf_path = os.path.join(outdir, 'figure_5.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()
