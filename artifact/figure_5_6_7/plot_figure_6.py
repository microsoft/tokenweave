import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os

indir = sys.argv[1]
outdir = sys.argv[2]
csvdir = sys.argv[3]
seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
x_ticks = ['64', '128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32K']

data_path = os.path.join(indir, 'figure_5_6_7_hs_8192.csv')
df_runtime = pd.read_csv(data_path)

filtered_df = df_runtime.loc[df_runtime.groupby('BL')['multimem_ar_us'].idxmin()].copy()


filtered_path = os.path.join(csvdir, 'figure_6.csv')
filtered_df.to_csv(filtered_path, index=False)

df_runtime = filtered_df
seq_lens = np.array(seq_lens)
df_runtime['achieved_bandwidth'] = (seq_lens * 2 * 8192 * 2 * 7 / 8) / (df_runtime['multimem_ar_us'] / 1e6) / (1024**3)  # GB/s
fig, ax = plt.subplots(figsize=(14, 8))

sns.set_theme(style="whitegrid", palette="deep")
sns.set_context("paper")

x_pos = np.arange(len(seq_lens))
ax.plot(x_pos, df_runtime['achieved_bandwidth'], marker='o', markersize=10,
        label='AllReduce (Multimem)', linewidth=3, color='tab:green')
for i, val in enumerate(df_runtime['achieved_bandwidth']):
    ax.annotate(f"{val:.1f}\nGB/s", (x_pos[i], val), textcoords="offset points",
                xytext=(-10, -40), fontsize=16)
ax.set_ylim(60, 1.05 * df_runtime['achieved_bandwidth'].max())

ax.set_xticks(x_pos)
ax.set_xticklabels(x_ticks, rotation=45, fontsize=20)
ax.set_xlabel("Seq Length", fontweight='bold', fontsize=30)
ax.set_ylabel("Achieved Bandwidth (GB/s)", fontsize=28, weight='bold')
ax.tick_params(axis='both', labelsize=28)

# Grid and legend
ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)
ax.legend(fontsize=24, frameon=False)

# Tight layout and save
plt.tight_layout()
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "figure_6.pdf"), bbox_inches='tight', dpi=1200)
