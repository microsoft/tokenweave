import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os
import matplotlib as mpl

mpl.rcParams.update({
    "pgf.rcfonts": False,
})

indir = sys.argv[1]
outdir = sys.argv[2]
csvdir = sys.argv[3]

data_path = os.path.join(indir, 'figure_4_10_hs_8192.csv')
df = pd.read_csv(data_path)

metrics = [
    "full_rms_norm_us", "partial_rms_norm_us", "torch_all_reduce_us",
    "multimem_all_reduce_us", "fused_arln_us", "simple_fusion_arln_us",
    "multimem_rs_us", "multimem_ag_us"
]

df_min = df.groupby(['BL', 'hidden_size'])[metrics].min().reset_index()

filtered_path = os.path.join(csvdir, 'figure_4.csv')
df_min.to_csv(filtered_path, index=False)
df_runtime = df_min

seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
x_ticks = ['128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32K']

# Group settings
base_cfgs = ['multimem_all_reduce_us']
base_cfgs2 = ['multimem_rs_us', 'multimem_ag_us']
stack_cfg = 'full_rms_norm_us'
stack_cfg2 = 'partial_rms_norm_us'
other_cfgs = [
    'fused_arln_us',
    'simple_fusion_arln_us',
]

cfg_labels = {
    'full_rms_norm_us': 'RMSNorm (Full)',
    'partial_rms_norm_us': 'RMSNorm (1 / N)',
    'multimem_all_reduce_us': 'AllReduce (Multimem)',
    'multimem_rs_us': 'ReduceScatter (Multimem)',
    'multimem_ag_us': 'AllGather (Multimem)',
    'fused_arln_us': 'Fused(AR, RMSNorm) (Ours)',
    'simple_fusion_arln_us': 'RS+RMSNorm+AG (Simple Fusion)',
}

runtime_colors = {
    'RMSNorm (Full)': 'tab:olive',
    'RMSNorm (1 / N)': 'tab:red',
    'AllReduce (Multimem)': 'tab:cyan',
    'ReduceScatter (Multimem)': 'tab:orange',
    'AllGather (Multimem)': 'tab:pink',
    'Fused(AR, RMSNorm) (Ours)': 'tab:green',
    'RS+RMSNorm+AG (Simple Fusion)': 'tab:gray',
}

runtime_hatches = {
    'RMSNorm (Full)': '//',
    'RMSNorm (1 / N)': '\\\\',
    'AllReduce (Multimem)': '-',
    'ReduceScatter (Multimem)': 'o',
    'AllGather (Multimem)': '.',
    'Fused(AR, RMSNorm) (Ours)': '',
    'RS+RMSNorm+AG (Simple Fusion)': 'o.',
}

runtime_alpha = {label: 0.6 for label in cfg_labels.values()}

split_idx = seq_lens.index(2048)

short_lens = seq_lens[:split_idx]
long_lens = seq_lens[split_idx:]
x_ticks_left = x_ticks[:split_idx]
x_ticks_right = x_ticks[split_idx:]

total_bars = len(base_cfgs) + len(other_cfgs)
runtime_alpha = {label: 0.6 for label in cfg_labels.values()}
base_cfgs = ['multimem_all_reduce_us']
other_cfgs = ['fused_arln_us', 'simple_fusion_arln_us']
stacked_cfgs = [
    ('multimem_all_reduce_us', 'full_rms_norm_us', None),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharey=False, gridspec_kw={'wspace': 0.2, 'width_ratios': [4, 5]})

def plot_bars(ax, lens, x_pos, is_left, annotate=True):
    width = 0.24
    total_bars = len(base_cfgs) + len(other_cfgs)
    if annotate:
        if is_left:
            baseline = df_runtime['multimem_all_reduce_us'].iloc[:len(lens)].values + \
                df_runtime['full_rms_norm_us'].iloc[:len(lens)].values
        else:
            baseline = df_runtime['multimem_all_reduce_us'].iloc[split_idx+1:].values + \
                df_runtime['full_rms_norm_us'].iloc[split_idx+1:].values

    for j, cfg in enumerate(other_cfgs):
        label = cfg_labels[cfg]
        if is_left:
            row = df_runtime[cfg].iloc[:len(lens)].values[:len(lens)]
        else:
            row = df_runtime[cfg].iloc[split_idx+1:].values[:len(lens)]
        offset = (j + len(base_cfgs) - (total_bars - 1) / 2) * width
        bars = ax.bar(x_pos + offset, row, width=width,
                      label=label,
                      color=runtime_colors[label],
                      edgecolor='black', linewidth=0.8,
                      hatch=runtime_hatches[label],
                      alpha=runtime_alpha[label])
        if annotate:
            for k, (bar, y_val) in enumerate(zip(bars, row)):
                rel = baseline[k] / y_val 
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03 * max(row),
                    f"{rel:.2f}x",
                    ha='center', va='bottom',
                    fontsize=14, rotation=90,
                    fontweight='bold'
                )
        
    for i, (base_cfg, ln_cfg, top_cfg) in enumerate(stacked_cfgs):
        offset = (i - (total_bars - 1) / 2) * width
        label_base = cfg_labels[base_cfg]
        label_ln = cfg_labels[ln_cfg]
        label_top = cfg_labels[top_cfg] if top_cfg else None

        if is_left:
            base_vals = df_runtime[base_cfg].iloc[:len(lens)].values[:len(lens)]
            ln_vals = df_runtime[ln_cfg].iloc[:len(lens)].values[:len(lens)]
            if top_cfg:
                top_vals = df_runtime[top_cfg].iloc[:len(lens)].values[:len(lens)]
        else:
            base_vals = df_runtime[base_cfg].iloc[split_idx+1:].values[:len(lens)]
            ln_vals = df_runtime[ln_cfg].iloc[split_idx+1:].values[:len(lens)]
            if top_cfg:
                top_vals = df_runtime[top_cfg].iloc[split_idx+1:].values[:len(lens)]

        bars_base = ax.bar(x_pos + offset, base_vals, width=width,
                           label=label_base, color=runtime_colors[label_base],
                           edgecolor='black', linewidth=0.8,
                           hatch=runtime_hatches[label_base],
                           alpha=runtime_alpha[label_base])

        bars_ln = ax.bar(x_pos + offset, ln_vals, width=width,
                         bottom=base_vals, label=label_ln,
                         color=runtime_colors[label_ln],
                         edgecolor='black', linewidth=0.8,
                         hatch=runtime_hatches[label_ln],
                         alpha=runtime_alpha[label_ln])

        if top_cfg:
            bars_top = ax.bar(x_pos + offset, top_vals, width=width,
                              bottom=base_vals + ln_vals,
                              label=label_top,
                              color=runtime_colors[label_top],
                              edgecolor='black', linewidth=0.8,
                              hatch=runtime_hatches[label_top],
                              alpha=runtime_alpha[label_top])
    if annotate:
        for k, x in enumerate(x_pos):
            total_height = base_vals[k] + ln_vals[k] + (top_vals[k] if top_cfg else 0)
            rel = baseline[k] / total_height
            ax.text(
                x + offset,
                total_height + 0.05 * max(base_vals + ln_vals + (top_vals if top_cfg else 0)),
                f"{rel:.2f}x",
                ha='center', va='bottom',
                fontsize=14, rotation=90,
                fontweight='bold'
            )

    ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)

x_pos_left = np.arange(len(short_lens))
x_pos_right = np.arange(len(long_lens))

plot_bars(ax1, short_lens, x_pos_left, True, True)
plot_bars(ax2, long_lens, x_pos_right, False, True)

ax1.set_xticks(x_pos_left)
ax1.set_xticklabels(x_ticks_left, fontsize=28, rotation=45)
ax2.set_xticks(x_pos_right)
ax2.set_xticklabels(x_ticks_right, fontsize=28, rotation=45)

ax1.set_ylabel("Time (us)", fontweight='bold', fontsize=30)
ax1.set_yticks(ax1.get_yticks())
ax1.tick_params(axis='y', labelsize=28)
ax2.set_yticks(ax2.get_yticks()) 
ax2.tick_params(axis='y', labelsize=28)
fig.supxlabel("Seq Length", fontweight='bold', fontsize=30, y=-0.07)


max_height = max([bar.get_height() for bar in ax1.patches])
ax1.set_ylim(top=max_height * 1.3)
max_height = max([bar.get_height() for bar in ax2.patches])
ax2.set_ylim(top=max_height * 1.3)

ax2.legend(
    loc='upper left',
    ncol=1,
    fontsize=20,
    frameon=False,
)

plt.subplots_adjust(top=0.85, wspace=0.2)
os.makedirs(outdir, exist_ok=True)
out_path_pdf = os.path.join(outdir, "figure_4.pdf")
plt.savefig(out_path_pdf, bbox_inches='tight', dpi=1200)