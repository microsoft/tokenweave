import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sysname = "TokenWeave"
# Settings
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
    'overlap_fused': f'{sysname}',
}

runtime_colors = {
    'vllm-Multimem': 'tab:olive',
    'vllm-nocomm': 'tab:brown',
    f'{sysname}': 'tab:green',
}

runtime_hatches = { 
    'vllm-Multimem': '//',
    'vllm-nocomm': '\\\\',
    f'{sysname}': '',
}

runtime_markers = { 
    'vllm-Multimem': 'o',
    'vllm-nocomm': 'd',
    f'{sysname}': '*',
}

runtime_alpha = { 
    'vllm-Multimem': 0.8,
    'vllm-nocomm': 0.6,
    f'{sysname}': 0.8,
}

def filter_and_aggregate(data, model_name, gpu_count):
    filtered = data[(data['model_name'] == model_name) & (data['gpu_count'] == gpu_count)]
    result = pd.DataFrame()
    for prompt_len in prompt_lens:
        subset = filtered[filtered['prompt_len'] == prompt_len]
        row = {'prompt_len': prompt_len}
        for impl in impl_methods[:4]:
            row[impl] = subset[subset['impl'] == impl]['avg_latency(ms)'].median()
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result


def plot_graphs(data, gpu_count, models, outdir):
    fig_x_width = 10
    fig_y_width = 6
    fig, ax = plt.subplots(figsize=(fig_x_width, fig_y_width))
    x_pos = np.arange(len(x_ticks))
    model = models[0]

    model_data = filter_and_aggregate(data, model, gpu_count)
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, fontsize=20, rotation=45)
    ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.tick_params(axis='y', labelsize=20)
    for i, impl in enumerate(impl_methods):
        ax.bar(x_pos + (i - 1) * 0.25, model_data[impl], width=0.25, label=cfg_labels[impl],
               color=runtime_colors[cfg_labels[impl]], hatch=runtime_hatches[cfg_labels[impl]],
               alpha=runtime_alpha[cfg_labels[impl]])
        for j, val in enumerate(model_data[impl]):
            ax.text(x_pos[j] + (i - 1) * 0.25, val + 0.5, f"{val:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=90)
    max_y = model_data[impl_methods].max().max() * 1.2
    ax.set_ylim(0, max_y)
    baseline_multimem = model_data['baseline_multimem']
    overlap_fused = model_data['overlap_fused']
    no_ar = model_data['no_ar']
    ax2 = ax.twinx()
    ax2.set_ylabel('Relative Performance\nwrt vllm-Multimem', fontsize=18, fontweight='bold')
    ax2.plot(x_pos, (baseline_multimem / overlap_fused), label=f"{sysname}", color=runtime_colors[f"{sysname}"], marker=runtime_markers[f"{sysname}"], markersize=10, alpha=runtime_alpha[f"{sysname}"], linestyle='--')
    ax2.plot(x_pos, (baseline_multimem / no_ar), label='vllm-nocomm', color=runtime_colors['vllm-nocomm'], marker=runtime_markers['vllm-nocomm'], markersize=10, alpha=runtime_alpha['vllm-nocomm'], linestyle='--')
    ax2.set_ylim(1, 1.5)
    ax2.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.tick_params(axis='y', labelsize=22)
    plt.tight_layout(rect=[0.03, 0.05, 1, 1])
    ax.legend(loc='upper left', ncol=1, fontsize=16, frameon=False)
    ax.set_xlabel("Seq Length", fontweight='bold', fontsize=20)
    ax.set_ylabel('Time (ms)', fontsize=20, fontweight='bold')

    pdf_path = os.path.join(outdir, f'figure_2.pdf')
    plt.savefig(pdf_path, dpi=1200)
    plt.close()


def main():
    indir = sys.argv[1]
    outdir = sys.argv[2]
    csvdir = sys.argv[3]
    data_path = os.path.join(indir, 'figure_2_10_summary.csv')
    data = pd.read_csv(data_path)

    os.makedirs(outdir, exist_ok=True)

    plot_graphs(data, 8, ["Llama-3.3-70B-Instruct"], outdir)


if __name__ == '__main__':
    main()