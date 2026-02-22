import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sysname = "TokenWeave"
prompt_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
x_ticks = ['1K', '2K', '4K', '8K', '16K', '32K', '64K']
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
    'overlap_fused': f"{sysname}",
}

runtime_colors = {
    'vllm-Multimem': 'tab:olive',
    'vllm-nocomm': 'tab:brown',
    f"{sysname}": 'tab:green',
}

runtime_hatches = { 
    'vllm-Multimem': '-',
    'vllm-nocomm': 'o.',
    f"{sysname}": '//',
}

runtime_alpha = {label: 0.6 for label in cfg_labels.values()}

def filter_and_aggregate(data, model_name, gpu_count):
    filtered = data[(data['model_name'] == model_name) & (data['gpu_count'] == gpu_count)]
    result = pd.DataFrame()
    for prompt_len in prompt_lens:
        subset = filtered[filtered['prompt_len'] == prompt_len]
        row = {'prompt_len': prompt_len}
        for impl in impl_methods[:3]:
            row[impl] = subset[subset['impl'] == impl]['avg_latency(ms)'].median()
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result


def plot_graphs(data, gpu_count, models, graph_title, outdir):
    fig_x_width = len(models) * 12
    fig_y_width = len(models) * 4
    fig, axes = plt.subplots(1, len(models), figsize=(fig_x_width, 8))
    width = 0.25
    x_pos = np.arange(len(x_ticks))
    for ax, model in zip(axes, models):
        model_data = filter_and_aggregate(data, model, gpu_count)
        if model == "Mixtral-8x22B-Instruct-v0.1":
            model = "Mixtral-8x22B-Instruct"
        ax.set_title(f'{model} ({gpu_count}x H100)', fontsize=30, fontweight='bold')
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, fontsize=25, rotation=45)
        ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)
        ax.tick_params(axis='y', labelsize=28)
        for i, impl in enumerate(impl_methods):
            bars = ax.bar(x_pos + (i - 1) * width, model_data[impl], width=width, label=cfg_labels[impl],
                          color=runtime_colors[cfg_labels[impl]], hatch=runtime_hatches[cfg_labels[impl]],
                          alpha=runtime_alpha[cfg_labels[impl]])
        max_y = model_data[impl_methods].max().max() * 1.15
        ax.set_ylim(0, max_y)
        baseline_multimem = model_data['baseline_multimem']
        for j, val in enumerate(baseline_multimem):
            ax.hlines(val, j-0.375, j + 0.375, colors='red', linestyles='dashed', linewidth=0.5)
        overlap_fused = model_data['overlap_fused']
        for j in range(len(x_ticks)):
            ax.vlines(j + 0.375, baseline_multimem.iloc[j], overlap_fused.iloc[j], colors='blue', linestyles='dashed', linewidth=1.2)
            speedup = baseline_multimem.iloc[j] / overlap_fused.iloc[j]
            ax.text(j + 0.5, (baseline_multimem.iloc[j] + overlap_fused.iloc[j]) / 2, f'{speedup:.2f}x',
                    ha='center', va='bottom', fontsize=20, rotation=90, fontweight='bold')
    plt.tight_layout(rect=[0.02, 0.05, 1, 0.93])
    handles, labels = axes[-1].get_legend_handles_labels()
    legend_fontsize = 28
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=legend_fontsize, frameon=False)
    fig.supxlabel("Seq Length", fontweight='bold', fontsize=28)
    axes[0].set_ylabel('Time (ms)', fontsize=28, fontweight='bold')
    filename = f'figure_11a' if gpu_count == 8 else f'figure_11b'
    pdf_path = os.path.join(outdir, f'{filename}.pdf')
    plt.savefig(pdf_path, dpi=1200)
    plt.close()


def main():
    indir = sys.argv[1]
    outdir = sys.argv[2]
    csvdir = sys.argv[3]
    data_path = os.path.join(indir, 'figure_2_11_summary.csv')
    data = pd.read_csv(data_path)

    os.makedirs(outdir, exist_ok=True)

    # Plot for GPU count 8
    model_list_8 = ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct", "Mixtral-8x22B-Instruct-v0.1"]
    all_filtered_8 = pd.concat([
        filter_and_aggregate(data, model, 8).assign(model=model)
        for model in model_list_8
    ], ignore_index=True)
    # Move 'model' column to the front
    cols_8 = ['model'] + [col for col in all_filtered_8.columns if col != 'model']
    all_filtered_8 = all_filtered_8[cols_8]
    all_filtered_8.to_csv(os.path.join(csvdir, 'figure_11a.csv'), index=False)
    plot_graphs(data, 8, model_list_8, 'Prefill (8x H100)', outdir)

if __name__ == '__main__':
    main()