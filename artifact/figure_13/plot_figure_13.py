import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sysname = "TokenWeave"
prompt_lens = [512, 1024, 2048, 0]
x_ticks = ['(512, 128)', '(1K, 128)', '(2K, 128)', '(sharegpt)']
impl_methods = [
    'baseline_multimem', 'no_ar', 'overlap_fused'
]

cfgs = [
    'baseline_multimem',
    'no_ar',
    'overlap_fused',
]

cfg_labels = {
    'baseline_multimem': 'vllm-Multimem',
    'no_ar': 'vllm-nocomm',
    'overlap_fused': f'{sysname}',
}

runtime_colors = {
    'vllm-Multimem': 'tab:olive',
    'vllm-nocomm': 'tab:brown',
    f'{sysname}': 'tab:green',
}

runtime_hatches = { 
    'vllm-Multimem': '-',
    'vllm-nocomm': 'o.',
    f'{sysname}': '//',
}

runtime_alpha = {label: 0.6 for label in cfg_labels.values()}

def filter_and_aggregate(data, cs, gpu_count):
    filtered = data[(data['cs'] == cs) & (data['gpu_count'] == gpu_count)]
    result = pd.DataFrame()
    for prompt_len in prompt_lens:
        if prompt_len == 0:
            subset = filtered[filtered['dataset'] == 'sharegpt']
            row = {'experiment': 'ShareGPT'}
        else:
            subset = filtered[filtered['input_len'] == prompt_len]
            row = {'experiment': f'({prompt_len}, 128)'}
        for impl in impl_methods[:3]:
            row[impl] = subset[subset['impl'] == impl]['total_tokens_per_sec'].median()
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result


def plot_graphs(data, gpu_count, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(36, 8), sharey=True, sharex=True)
    width = 0.25
    x_pos = np.arange(len(x_ticks))
    css = [1024, 2048, 4096, 8192]
    for ax, cs in zip(axes.flatten(), css):
        model_data = filter_and_aggregate(data, cs, gpu_count)
        ax.set_title(f'CS: {cs} ({gpu_count}x H100)', fontsize=30, fontweight='bold')
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, fontsize=25, rotation=45)
        ax.grid(axis='y', linestyle='--', linewidth=1, alpha=0.5)
        ax.tick_params(axis='y', labelsize=28)
        for i, impl in enumerate(impl_methods):
            bars = ax.bar(x_pos + (i - 1) * width, model_data[impl], width=width, label=cfg_labels[impl],
                          color=runtime_colors[cfg_labels[impl]], hatch=runtime_hatches[cfg_labels[impl]],
                          alpha=runtime_alpha[cfg_labels[impl]])
        max_y = model_data[impl_methods].max().max() * 1.15
        ax.set_ylim(0, max_y)
        baseline_multimem = model_data['baseline_multimem']
        overlap_fused = model_data['overlap_fused']
        for j, val in enumerate(zip(baseline_multimem, overlap_fused)):
            baseline, overlap = val
            ax.hlines(overlap, j - 0.25, j + 0.375, colors='red', linestyles='dashed', linewidth=1.5)
        for j in range(len(x_ticks)):
            ax.vlines(j - 0.25, overlap_fused.iloc[j], baseline_multimem.iloc[j], colors='blue', linestyles='dashed', linewidth=1.5)
            speedup = overlap_fused.iloc[j] / baseline_multimem.iloc[j]
            ax.text(j - 0.375, (baseline_multimem.iloc[j] + overlap_fused.iloc[j]) / 2, f'{speedup:.2f}x',
                    ha='center', va='bottom', fontsize=18, rotation=90, fontweight='bold')
    
    handles, labels = axes[0].get_legend_handles_labels()
    legend_fontsize = 28
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=legend_fontsize, frameon=False)
    fig.supxlabel("Fixed(Input, Output) or ShareGPT", fontweight='bold', fontsize=28, y=0.02)
    fig.supylabel("Tokens per sec", fontweight='bold', fontsize=28, x=0.01)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.3, wspace=0.1, hspace=0.1)
    pdf_path = os.path.join(outdir, 'figure_13.pdf')
    plt.savefig(pdf_path, dpi=1200)
    plt.close()


def main():
    indir = sys.argv[1]
    outdir = sys.argv[2]
    csvdir = sys.argv[3]
    data_path = os.path.join(indir, 'figure_13_summary.csv')
    data = pd.read_csv(data_path)

    os.makedirs(outdir, exist_ok=True)

    css = [1024, 2048, 4096, 8192]
    all_filtered = []

    for cs in css:
        filtered_data = filter_and_aggregate(data, cs, 8)
        filtered_data["chunk_size"] = cs
        all_filtered.append(filtered_data)
    combined_df = pd.concat(all_filtered, ignore_index=True)
    cols = ['chunk_size'] + [col for col in combined_df.columns if col != 'chunk_size']
    combined_df = combined_df[cols]
    combined_df.to_csv(os.path.join(csvdir, 'figure_13.csv'), index=False)

    plot_graphs(data, 8, outdir)


if __name__ == '__main__':
    main()