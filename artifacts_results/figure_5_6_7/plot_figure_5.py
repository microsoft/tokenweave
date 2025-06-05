import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
indir = sys.argv[1]
outdir = sys.argv[2]
csvdir = sys.argv[3]
import matplotlib as mpl
mpl.rcParams.update({
    "pgf.rcfonts": False,              # Ignore matplotlib font settings
})

seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
x_ticks = ['64', '128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32K']

cfg_labels = {
    'pytorch_dist_ar_us': 'AllReduce (Default)',
    'pytorch_dist_rs_us': 'ReduceScatter (Default)',
    'pytorch_dist_ag_us': 'AllGather (Default)',
    'multimem_ar_us': 'AllReduce (Multimem)',
    'multimem_rs_us': 'ReduceScatter (Multimem)',
    'multimem_ag_us': 'AllGather (Multimem)',
}

runtime_colors = {
    'AllReduce (Default)': 'tab:orange',
    'ReduceScatter (Default)': 'tab:purple',
    'AllGather (Default)': 'tab:green',
    'AllReduce (Multimem)': 'tab:blue',
    'ReduceScatter (Multimem)': 'tab:gray',
    'AllGather (Multimem)': 'tab:olive',
}

runtime_hatches = {
    'AllReduce (Default)': '',
    'ReduceScatter (Default)': '//',
    'AllGather (Default)': 'o.',
    'AllReduce (Multimem)': '',
    'ReduceScatter (Multimem)': '\\',
    'AllGather (Multimem)': '*',
}

runtime_alpha = {key: 0.6 for key in cfg_labels.values()}

data_path = os.path.join(indir, 'figure_5_6_7_hs_8192.csv')
df_runtime = pd.read_csv(data_path)
filtered_df = df_runtime.loc[df_runtime.groupby('BL')['multimem_ar_us'].idxmin()].copy()
filtered_path = os.path.join(csvdir, 'figure_5.csv')
filtered_df.to_csv(filtered_path, index=False)
df_runtime = filtered_df

width = 0.20
x_pos = np.arange(len(seq_lens))
offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width] 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplots_adjust(wspace=0.5) 
split_idx = seq_lens.index(2048)
left_lens = seq_lens[:split_idx]
right_lens = seq_lens[split_idx:]
x_ticks_left = x_ticks[:split_idx]
x_ticks_right = x_ticks[split_idx:]
x_pos_left = np.arange(len(left_lens))
x_pos_right = np.arange(len(right_lens))

def plot_bars(ax, lens, x_pos, cfgs, offset_idx):
    for i, cfg_group in enumerate(cfgs):
        if isinstance(cfg_group, str): 
            cfg = cfg_group
            label = cfg_labels[cfg]
            row = df_runtime[cfg].iloc[:len(lens)] if offset_idx == 0 else df_runtime[cfg].iloc[split_idx:]
            ax.bar(x_pos + offsets[i], row.values, width,
                   label=label,
                   color=runtime_colors[label],
                   edgecolor='black', linewidth=0.8,
                   hatch=runtime_hatches[label],
                   alpha=runtime_alpha[label])
        else:
            cfg1, cfg2 = cfg_group
            label1, label2 = cfg_labels[cfg1], cfg_labels[cfg2]
            row1 = df_runtime[cfg1].iloc[:len(lens)] if offset_idx == 0 else df_runtime[cfg1].iloc[split_idx:]
            row2 = df_runtime[cfg2].iloc[:len(lens)] if offset_idx == 0 else df_runtime[cfg2].iloc[split_idx:]
            ax.bar(x_pos + offsets[i], row1.values, width,
                   label=label1,
                   color=runtime_colors[label1],
                   edgecolor='black', linewidth=0.8,
                   hatch=runtime_hatches[label1],
                   alpha=runtime_alpha[label1])
            ax.bar(x_pos + offsets[i], row2.values, width,
                   bottom=row1.values,
                   label=label2,
                   color=runtime_colors[label2],
                   edgecolor='black', linewidth=0.8,
                   hatch=runtime_hatches[label2],
                   alpha=runtime_alpha[label2])

cfg_sets = [
    'pytorch_dist_ar_us',
    ('pytorch_dist_rs_us', 'pytorch_dist_ag_us'),
    'multimem_ar_us',
    ('multimem_rs_us', 'multimem_ag_us')
]

plot_bars(ax1, left_lens, x_pos_left, cfg_sets, offset_idx=0)
plot_bars(ax2, right_lens, x_pos_right, cfg_sets, offset_idx=1)

ax1.set_xticks(x_pos_left)
ax1.set_xticklabels(x_ticks_left, fontsize=28, rotation=45)
ax2.set_xticks(x_pos_right)
ax2.set_xticklabels(x_ticks_right, fontsize=28, rotation=45)
ax1.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)
ax2.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)

ax1.set_ylabel("Time (us)", fontweight='bold', fontsize=30)
ax1.set_yticks(ax1.get_yticks())
ax1.tick_params(axis='y', labelsize=28)
ax2.set_yticks(ax2.get_yticks())
ax2.tick_params(axis='y', labelsize=28, labelright=True, right=True)
fig.supxlabel("Seq Length", fontweight='bold', fontsize=30)

ax1_twin = ax1.twinx()
ax2_twin = ax2.twinx()

df_runtime['normalized_rs_ag_pytorch_dist_us'] = (df_runtime['pytorch_dist_rs_us'] + df_runtime['pytorch_dist_ag_us']) / df_runtime['pytorch_dist_ar_us']
df_runtime['normalized_rs_ag_us'] = (df_runtime['multimem_rs_us'] + df_runtime['multimem_ag_us']) / df_runtime['multimem_ar_us']
norm_pytorch_dist = df_runtime['normalized_rs_ag_pytorch_dist_us']
norm_mm = df_runtime['normalized_rs_ag_us']


ax1_twin.plot(x_pos_left, norm_pytorch_dist[:split_idx], color='tab:orange', marker='o', markersize=8, label='Ratio (Default)')
ax1_twin.plot(x_pos_left, norm_mm[:split_idx], color='tab:blue', marker='s', markersize=8, label='Ratio (Multimem)')


ax2_twin.plot(x_pos_right, norm_pytorch_dist[split_idx:], color='tab:orange', marker='o', markersize=8, label='Ratio (Default)')
ax2_twin.plot(x_pos_right, norm_mm[split_idx:], color='tab:blue', marker='s', markersize=8, label='Ratio (Multimem)')


for ax_twin in [ax1_twin, ax2_twin]:
    ax_twin.set_ylim(0, 2.2)
    if ax_twin == ax2_twin:
       ax_twin.set_ylabel("(RS + AG) / AR", fontweight='bold', fontsize=30)
    ax_twin.tick_params(axis='y', labelsize=22)



handles1, labels1 = ax1.get_legend_handles_labels()
handles_line1, labels_line1 = ax1_twin.get_legend_handles_labels()


legend = fig.legend(handles1 + handles_line1, labels1 + labels_line1,
           loc='upper center', ncol=4, fontsize=22,
           bbox_to_anchor=(0.5, 1.12), frameon=False)

os.makedirs(outdir, exist_ok=True)
out_path_pdf = os.path.join(outdir, "figure_5.pdf")
plt.tight_layout()
plt.savefig(out_path_pdf, bbox_inches='tight', dpi=1200)
