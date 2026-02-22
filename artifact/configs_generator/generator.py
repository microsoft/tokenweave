import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import json

sysname = "TokenWeave"
prompt_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

def filter_and_aggregate(data, model_name, gpu_count):
    filtered = data[(data['model_name'] == model_name) & (data['gpu_count'] == gpu_count)]
    result = pd.DataFrame()
    for prompt_len in prompt_lens:
        subset = filtered[filtered['prompt_len'] == prompt_len]
        row = {'prompt_len': prompt_len}
        for impl in impl_methods[:4]:
            row[impl] = subset[subset['impl'] == impl]['avg_latency(ms)'].median()
        fused = subset[subset['impl'] == 'baseline_fused']
        min_fused = fused['avg_latency(ms)'].min()
        row['baseline_fused'] = min_fused
        row['baseline_fused_attn_ctas'] = fused.loc[fused['avg_latency(ms)'].idxmin(), 'attn_ctas']
        row['baseline_fused_mlp_ctas'] = fused.loc[fused['avg_latency(ms)'].idxmin(), 'mlp_ctas']

        overlap = subset[subset['impl'] == 'overlap_fused']
        min_overlap = overlap['avg_latency(ms)'].min()
        row['overlap_fused'] = min_overlap
        row['overlap_fused_split_offset'] = overlap.loc[overlap['avg_latency(ms)'].idxmin(), 'split_offset']
        row['overlap_fused_attn_ctas'] = overlap.loc[overlap['avg_latency(ms)'].idxmin(), 'attn_ctas']
        row['overlap_fused_mlp_ctas'] = overlap.loc[overlap['avg_latency(ms)'].idxmin(), 'mlp_ctas']
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result

def model_name_to_short(model_name):
    """Convert full model name to short name for filename"""
    if "Llama" in model_name:
        return "llama"
    elif "Qwen" in model_name:
        return "qwen2"
    elif "Mixtral" in model_name:
        return "mixtral"
    else:
        return model_name.lower()

def create_json_config(data, model_name, gpu_count, outdir):
    """Create JSON configuration file by finding minimum latency configuration"""
    config = {}
    
    # Filter data for this model and GPU count
    filtered = data[(data['model_name'] == model_name) & (data['gpu_count'] == gpu_count)]
    
    for prompt_len in prompt_lens:
        prompt_len_str = str(prompt_len)
        
        # Get all rows for this prompt length
        subset = filtered[filtered['prompt_len'] == prompt_len]

        subset_baseline_fused = subset[subset['impl'] == 'baseline_fused']
        subset_overlap_fused = subset[subset['impl'] == 'overlap_fused']
        
        if len(subset) > 0:
            # Find the row with minimum latency
            min_idx_baseline_fused = subset_baseline_fused['avg_latency(ms)'].idxmin() if not subset_baseline_fused.empty else None
            min_idx_overlap_fused = subset_overlap_fused['avg_latency(ms)'].idxmin() if not subset_overlap_fused.empty else None
            min_row_baseline_fused = subset_baseline_fused.loc[min_idx_baseline_fused] if min_idx_baseline_fused is not None else pd.Series()
            min_row_overlap_fused = subset_overlap_fused.loc[min_idx_overlap_fused] if min_idx_overlap_fused is not None else pd.Series()
            
            # Extract the configuration parameters
            config[prompt_len_str] = {
                "baseline_ctas": int(max(min_row_baseline_fused['attn_ctas'], min_row_baseline_fused['mlp_ctas'])) if pd.notna(min_row_baseline_fused['attn_ctas']) and pd.notna(min_row_baseline_fused['mlp_ctas']) else 32,
                "attention_ctas": int(min_row_overlap_fused['attn_ctas']) if pd.notna(min_row_overlap_fused['attn_ctas']) else 8,
                "mlp_ctas": int(min_row_overlap_fused['mlp_ctas']) if pd.notna(min_row_overlap_fused['mlp_ctas']) else 8,
                "split_offset": int(min_row_overlap_fused['split_offset']) if pd.notna(min_row_overlap_fused['split_offset']) else 0
            }
        else:
            # Fallback to default values if no data found
            config[prompt_len_str] = {
                "baseline_ctas": 32,
                "attention_ctas": 8,
                "mlp_ctas": 8,
                "split_offset": 0
            }
    
    # Create JSON filename
    short_model_name = model_name_to_short(model_name)
    json_filename = f'{short_model_name}_config_{gpu_count}.json'
    json_path = os.path.join(outdir, json_filename)
    
    # Write JSON file
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created {json_filename}")

def main():
    # Input and output directories
    indir = sys.argv[1]
    outdir = sys.argv[2]
    data_path = os.path.join(indir, 'configs_generator_summary.csv')
    data = pd.read_csv(data_path)

    os.makedirs(outdir, exist_ok=True)

    # Process GPU count 8
    for model in ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct", "Mixtral-8x22B-Instruct-v0.1"]:
        create_json_config(data, model, 8, outdir)

if __name__ == '__main__':
    main()