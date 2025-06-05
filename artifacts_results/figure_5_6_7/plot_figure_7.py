import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os


indir = sys.argv[1]
outdir = sys.argv[2]
csvdir = sys.argv[3]
data_path = os.path.join(indir, 'figure_5_6_7_hs_8192.csv')
df = pd.read_csv(data_path)
grouped = df.groupby(['BL', 'hidden_size', 'multimem_ctas'])
idx = grouped['multimem_ar_us'].idxmin()
df_best = df.loc[idx].copy()
pytorch_dist_avg = df.groupby(['BL', 'hidden_size'])['pytorch_dist_ar_us'].mean().reset_index()
df_best = df_best.merge(pytorch_dist_avg, on=['BL', 'hidden_size'], suffixes=('', '_avg'))
df_best['pytorch_dist_ar_us'] = df_best['pytorch_dist_ar_us_avg']
df_best.drop(columns=['pytorch_dist_ar_us_avg'], inplace=True)


pivot_df = df_best.pivot(
    index=['BL', 'hidden_size', 'pytorch_dist_ar_us'],
    columns='multimem_ctas',
    values='multimem_ar_us'
).reset_index()


pivot_df = pivot_df.rename(columns={
    32: 'multimem_ar_us_ctas_32',
    16: 'multimem_ar_us_ctas_16',
    8: 'multimem_ar_us_ctas_8',
    4: 'multimem_ar_us_ctas_4',
    2: 'multimem_ar_us_ctas_2',
})


filtered_path = os.path.join(csvdir, 'figure_7.csv')
pivot_df.to_csv(filtered_path, index=False)

df_runtime = pivot_df

cfgs = [
    'multimem_ar_us_ctas_16',
    'multimem_ar_us_ctas_8',
    'multimem_ar_us_ctas_4',
    'multimem_ar_us_ctas_2',
]

cfg_labels = {
    'multimem_ar_us_ctas_16': '16 #SMs',
    'multimem_ar_us_ctas_8': '8 #SMs',
    'multimem_ar_us_ctas_4': '4 #SMs',
    'multimem_ar_us_ctas_2': '2 #SMs',
}

runtime_colors = {
    '16 #SMs': 'tab:green',
    '8 #SMs': 'tab:cyan',
    '4 #SMs': 'tab:olive',
    '2 #SMs': 'tab:gray',
}

runtime_hatches = {
    '16 #SMs': '//',
    '8 #SMs': '',
    '4 #SMs': '\\',
    '2 #SMs': 'o.',
}

runtime_alpha = {
    '16 #SMs': 0.6,
    '8 #SMs': 0.6,
    '4 #SMs': 0.6,
    '2 #SMs': 0.6,
}

import matplotlib as mpl
mpl.rcParams.update({
    "pgf.rcfonts": False,              # Ignore matplotlib font settings
})
seq_lens = [64, 128, 2048, 4096, 16384, 32768]
x_ticks = ['64', '128', '2K', '4K', '16K', '32K']
fig, ax = plt.subplots(figsize=(12, 8))

width = 0.20
x_pos = np.arange(len(seq_lens))
for i, cfg in enumerate(cfgs):
    label = cfg_labels[cfg]
    if cfg not in df_runtime.columns:
        print(f"Warning: {cfg} not found in data.")
        continue
    row = df_runtime[cfg].values[:len(seq_lens)]
    offset = (i - (len(cfgs) - 1) / 2) * width
    bars = ax.bar(x_pos + offset, row[:len(seq_lens)], width,
            label=label,
            color=runtime_colors[label],
            edgecolor='black', linewidth=0.8,
            hatch=runtime_hatches[label],
            alpha=runtime_alpha[label])

    if 'multimem_ar_us_ctas_2' in df_runtime.columns:
        baseline = df_runtime['multimem_ar_us_ctas_2'].values[:len(seq_lens)]
        for j, (bar, y_val) in enumerate(zip(bars, row[:len(seq_lens)])):
            if baseline[j] > 0:
                rel = baseline[j] / y_val 
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03 * max(row[:len(seq_lens)]),
                    f"{rel:.2f}x",
                    ha='center', va='bottom',
                    fontsize=14, rotation=90,
                    fontweight='bold',
                )

ax.set_xticks(x_pos)
ax.set_xticklabels(x_ticks, fontsize=28, rotation=45)

ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)

max_height = max([bar.get_height() for bar in ax.patches])

ax.set_ylim(top=max_height * 1.15)

ax.set_ylabel("Time (us)", fontweight='bold', fontsize=30)
ax.set_yticks(ax.get_yticks())
ax.tick_params(axis='y', labelsize=28)

handles, labels = [], []
h, l = ax.get_legend_handles_labels()
handles.extend(h)
labels.extend(l)

if ax.get_legend():
    ax.get_legend().remove()


legend = ax.legend(handles, labels,
           loc='upper left',
           ncol=2,
           frameon=False, fontsize=22)


fig.tight_layout()
os.makedirs(outdir, exist_ok=True)
out_path_pdf = os.path.join(outdir, "figure_7.pdf")
fig.supxlabel("Seq Length", fontweight='bold', fontsize=28)
plt.tight_layout()
plt.savefig(out_path_pdf, bbox_inches='tight', dpi=1200)
