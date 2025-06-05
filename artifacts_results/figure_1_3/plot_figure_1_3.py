import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Settings
prompt_lens = [1024, 2048, 4096, 8192, 16384, 32768]
x_ticks = ['1K', '2K', '4K', '8K', '16K', '32K']
impl_methods = [
   'baseline_multimem', 'no_ar1', 'no_ar2', 'no_ln1', 'no_ln2'
]

runtime_colors = {
    'Llama-3.3-70B': 'tab:green',
    'Qwen2.5-72B': 'tab:orange',
    'Mixtral-8x22B': 'tab:blue',
}

runtime_markers = {
    'Llama-3.3-70B': 'o',
    'Qwen2.5-72B': 's',
    'Mixtral-8x22B': '^',
}

runtime_ha_positions = {
    'Llama-3.3-70B': 'center',
    'Qwen2.5-72B': 'right',
    'Mixtral-8x22B': 'left',
}

runtime_va_positions = {
    'Llama-3.3-70B': 'top',
    'Qwen2.5-72B': 'top',
    'Mixtral-8x22B': 'bottom',
}

runtime_alpha = {label: 0.8 for label in runtime_colors.keys()}

def filter_and_aggregate(data, model_name, gpu_count):
    filtered = data[(data['model_name'] == model_name) & (data['gpu_count'] == gpu_count)]
    result = pd.DataFrame()
    for prompt_len in prompt_lens:
        subset = filtered[filtered['prompt_len'] == prompt_len]
        row = {'prompt_len': prompt_len}
        for impl in impl_methods[:5]:
            row[impl] = subset[subset['impl'] == impl]['avg_latency(ms)'].median()
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result


def plot_graphs_3(data, gpu_count, outdir):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=1200)
    x_pos = np.arange(len(x_ticks))
    models = ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct", "Mixtral-8x22B-Instruct-v0.1"]
    models_data = []
    for model in models:
        filtered_data = []
        for i in range(len(data)):
            flit_data = filter_and_aggregate(data[i], model, gpu_count)  
            filtered_data.append(flit_data)  
        models_data.append(filtered_data)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks, fontsize=25, rotation=45)
    models_layernorm_overhead_mean = []
    models_layernorm_overhead_std = []
    for model_data in models_data:
        overhead = []
        for i in range(len(model_data)):
            overhead.append((((model_data[i]['baseline_multimem'] - model_data[i]['no_ln1']) + (model_data[i]['baseline_multimem'] - model_data[i]['no_ln2'])) / model_data[i]['baseline_multimem']) * 100.0)
        models_layernorm_overhead_mean.append(np.mean(overhead, axis=0))
        models_layernorm_overhead_std.append(np.std(overhead, axis=0))
    
    for i, ln_overhead in enumerate(models_layernorm_overhead_mean):
        label = "Llama-3.3-70B" if i == 0 else "Qwen2.5-72B" if i == 1 else "Mixtral-8x22B"
        ax.errorbar(x_pos, ln_overhead, yerr=models_layernorm_overhead_std[i], label=label, color=runtime_colors[label], alpha=runtime_alpha[label], marker=runtime_markers[label], markersize=10, capsize=5)
        for j, val in enumerate(ln_overhead):
            x_offset = -0.2
            y_offset = 0.3 if i == 0 else -0.7 if i == 1 else -0.7
            y_offset = 0 if (i == 1 and j == 2) else y_offset
            ax.text(j + x_offset, val + y_offset, f'{val:.1f}%', fontsize=20, 
                    color=runtime_colors[label], alpha=runtime_alpha[label])
        
    handles, labels = ax.get_legend_handles_labels() 
    legend_fontsize = 20
    legend = fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1), fontsize=legend_fontsize, frameon=False)
    ax.set_ylabel('RMSNorm Overhead (%)', fontsize=28, fontweight='bold')
    ax.set_xlabel('Seq Length', fontsize=28, fontweight='bold')
    ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.set_yticks(ax.get_yticks()) 
    ax.set_yticklabels(ax.get_yticks(), fontsize=25)

    pdf_path = os.path.join(outdir, f'figure_3.pdf')
    plt.savefig(pdf_path, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_graphs_1(data, gpu_count, outdir):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=1200)
    x_pos = np.arange(len(x_ticks))
    models = ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct", "Mixtral-8x22B-Instruct-v0.1"]
    models_data = []
    for model in models:
        filtered_data = []
        for i in range(len(data)):
            flit_data = filter_and_aggregate(data[i], model, gpu_count)  
            filtered_data.append(flit_data)  
        models_data.append(filtered_data)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks, fontsize=25, rotation=45)
    models_ar_overhead_mean = []
    models_ar_overhead_std = []
    for model_data in models_data:
        overhead = []
        for i in range(len(model_data)):
            overhead.append((((model_data[i]['baseline_multimem'] - model_data[i]['no_ar1']) + (model_data[i]['baseline_multimem'] - model_data[i]['no_ar2'])) / model_data[i]['baseline_multimem']) * 100.0)
        models_ar_overhead_mean.append(np.mean(overhead, axis=0))
        models_ar_overhead_std.append(np.std(overhead, axis=0))
    
    for i, ar_overhead in enumerate(models_ar_overhead_mean):
        label = "Llama-3.3-70B" if i == 0 else "Qwen2.5-72B" if i == 1 else "Mixtral-8x22B"
        ax.errorbar(x_pos, ar_overhead, yerr=models_ar_overhead_std[i], label=label, color=runtime_colors[label], alpha=runtime_alpha[label], marker=runtime_markers[label], markersize=10, capsize=5)
        for j, val in enumerate(ar_overhead):
            y_offset = -1.2 if (i == 1  or i == 2) else 0.5
            y_offset = -1.4 if i == 1 else y_offset
            x_offset = -0.25
            y_offset = 0.5 if (i == 0 and j == 0) else y_offset
            ax.text(j + x_offset, val + y_offset, f'{val:.1f}%', fontsize=20, 
                    color=runtime_colors[label], alpha=runtime_alpha[label])
        
    handles, labels = ax.get_legend_handles_labels() 
    legend_fontsize = 20
    legend = fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1), fontsize=legend_fontsize, frameon=False)
    ax.set_ylabel('All Reduce Overhead (%)', fontsize=28, fontweight='bold')
    ax.set_xlabel('Seq Length', fontsize=28, fontweight='bold')
    ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.set_yticks(ax.get_yticks()) 
    ax.set_yticklabels(ax.get_yticks(), fontsize=25)

    pdf_path = os.path.join(outdir, f'figure_1.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Input and output directories
    indir = sys.argv[1]
    outdir = sys.argv[2]
    csvdir = sys.argv[3]
    NUM_RUNS = 5
    data_path = [os.path.join(indir, f'figure_1_3_summary_{i+1}.csv') for i in range(NUM_RUNS)]
    data = [pd.read_csv(path) for path in data_path]

    os.makedirs(outdir, exist_ok=True)
    plot_graphs_3(data, 8, outdir)
    plot_graphs_1(data, 8, outdir)


if __name__ == '__main__':
    main()