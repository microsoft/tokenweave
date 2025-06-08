#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SHAREGPT_FILE_NAME="ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_FILE_PATH="$SCRIPT_DIR/$SHAREGPT_FILE_NAME"

# Download the file if it doesn't exist
if [ ! -f "$SHAREGPT_FILE_PATH" ]; then
    echo "Downloading $SHAREGPT_FILE_NAME..."
    wget -O "$SHAREGPT_FILE_PATH" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/$SHAREGPT_FILE_NAME
else
    echo "File already exists. Skipping download."
fi

# ------------------ Configuration ------------------
EVAL_DIR=$1
MASTER_CSV="$EVAL_DIR/figure_13_summary.csv"
DATASET_NAME=("random" "sharegpt")
DATASET_PATH="$SHAREGPT_FILE_PATH"
INPUT_LENS=(512 1024 2048)
OUTPUT_LEN=128
RANDOM_RANGE_RATIO=0.0 # FIXED P:D
NUM_GPUS_LIST=(8)
MODEL_NAME_LIST=("Llama-3.3-70B-Instruct")
BASELINE_IMPL_LIST=("no_ar" "baseline_multimem")
OVERLAP_FUSED_IMPL_LIST=("overlap_fused")
CHUNKED_PREFILL_SIZES_LIST=(1024 2048 4096 8192)
extra_args=""

# ------------------ Environment Setup ------------------
nvidia-smi -pm ENABLED
nvidia-smi -lgc tdp
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ------------------ Utility Functions ------------------

log_info() {
    echo "[INFO] $1"
}

init_benchmark() {
    mkdir -p "$EVAL_DIR"
    echo "model_name,gpu_count,cs,dataset,input_len,output_len,impl,requests_per_sec,total_tokens_per_sec,output_tokens_per_sec,total_prompt_tokens,total_output_tokens" > "$MASTER_CSV"
}

copy_model_files() {
    local model=$1 impl=$2 src_dir=$3 prefix=$4 dst_file_path=$5
    log_info "Copying model files for $model, implementation: $impl"
    cp "$src_dir"/"$prefix""$impl".py "$dst_file_path"
}

extract_throughput() {
    local output_log=$1 model=$2 gpus=$3 impl=$4 dataset=$5 input_len=$6 output_len=$7 chunked_prefill_size=$8

    # Extract throughput and token information in a single pass
    while read -r line; do
        if [[ "$line" == *"Throughput:"* ]]; then
            requests_per_sec=$(echo "$line" | awk -F'[ :/,]' '{print $3}')
            total_tokens_per_sec=$(echo "$line" | awk -F'[ :/,]' '{print $7}')
            output_tokens_per_sec=$(echo "$line" | awk -F'[ :/,]' '{print $12}')
        elif [[ "$line" == *"Total num prompt tokens:"* ]]; then
            total_prompt_tokens=$(echo "$line" | awk -F'[:,]' '{print $2}' | xargs)
        elif [[ "$line" == *"Total num output tokens:"* ]]; then
            total_output_tokens=$(echo "$line" | awk -F'[:,]' '{print $2}' | xargs)
        fi
    done < <(grep -E "Throughput:|Total num prompt tokens:|Total num output tokens:" "$output_log")

    # Print the combined CSV row
    echo "$model,$gpus,$chunked_prefill_size,$dataset,$input_len,$output_len,$impl,$requests_per_sec,$total_tokens_per_sec,$output_tokens_per_sec,$total_prompt_tokens,$total_output_tokens" >> "$MASTER_CSV"
}


# ------------------ Benchmark Functions ------------------

run_benchmark_sharegpt() {
    local model_path=$1 model=$2 gpus=$3 impl=$4 dataset=$5 input_len=$6 output_len=$7 extra_args=$8 chunked_prefill_size=$9
    local dir="$EVAL_DIR/$model/${gpus}xh100"
    mkdir -p "$dir"
    local output_log="$dir/${impl}.txt"
    python "$SCRIPT_DIR/bench.py" \
        --dataset-name $dataset --dataset-path "$DATASET_PATH" $extra_args \
        --disable-detokenize --model "$model_path" \
        --tensor-parallel-size "$gpus" --enable-chunked-prefill --enforce-eager \
        --disable-custom-all-reduce --max-num-batched-tokens "$chunked_prefill_size" \
        --no-enable-prefix-caching $extra_args | tee -a "$output_log"

    extract_throughput "$output_log" "$model" "$gpus" "$impl" "$dataset" "$input_len" "$output_len" "$chunked_prefill_size"

    rm -rf "$dir"
}

run_benchmark_random() {
    local model_path=$1 model=$2 gpus=$3 impl=$4 dataset=$5 input_len=$6 output_len=$7 extra_args=$8 chunked_prefill_size=$9
    local dir="$EVAL_DIR/$model/${gpus}xh100"
    mkdir -p "$dir"
    local output_log="$dir/${impl}.txt"
    python "$SCRIPT_DIR/bench.py" \
        --dataset-name $dataset \
        --input-len "$input_len" --output-len "$output_len" --random-range-ratio "$RANDOM_RANGE_RATIO" $extra_args \
        --disable-detokenize --model "$model_path" \
        --tensor-parallel-size "$gpus" --enable-chunked-prefill --enforce-eager \
        --disable-custom-all-reduce --max-num-batched-tokens "$chunked_prefill_size" \
        --no-enable-prefix-caching $extra_args | tee -a "$output_log"

    extract_throughput "$output_log" "$model" "$gpus" "$impl" "$dataset" "$input_len" "$output_len" "$chunked_prefill_size"
    rm -rf "$dir"
}


# ------------------ Main Execution ------------------

log_info "Setting up the environment..."
init_benchmark

# Combine random input lengths and sharegpt into one loop
DATASET_CONFIGS=("${INPUT_LENS[@]}" "sharegpt")

for dataset_config in "${DATASET_CONFIGS[@]}"; do
    cp -f "$SCRIPT_DIR/hybrid_configs/llama_config_8_$dataset_config.json" "$SCRIPT_DIR/../../vllm/tokenweave_configs/llama_config_8.json"
    for model in "${MODEL_NAME_LIST[@]}"; do
        case "$model" in
            "Llama-3.3-70B-Instruct")
                model_path="meta-llama/Llama-3.3-70B-Instruct"
                prefix="pl_"
                src_dir="$SCRIPT_DIR/../llama_src_files"
                dst_file_path="${SCRIPT_DIR}/../../vllm/model_executor/models/llama.py"
                extra_args=""
                ;;
            "Qwen2.5-72B-Instruct")
                model_path="Qwen/Qwen2.5-72B-Instruct"
                prefix="pq_"
                src_dir="$SCRIPT_DIR/../qwen2_src_files"
                dst_file_path="${SCRIPT_DIR}/../../vllm/model_executor/models/qwen2.py"
                extra_args=""
                ;;
            "Mixtral-8x22B-Instruct-v0.1")
                model_path="mistralai/Mixtral-8x22B-Instruct-v0.1"
                prefix="pm_"
                src_dir="$SCRIPT_DIR/../mixtral_src_files"
                dst_file_path="${SCRIPT_DIR}/../../vllm/model_executor/models/mixtral.py"
                extra_args="--tokenizer-mode mistral"
                ;;
            *)
                log_info "Unknown model: $model"
                continue
                ;;
        esac

        for gpus in "${NUM_GPUS_LIST[@]}"; do
            for chunked_prefill_size in "${CHUNKED_PREFILL_SIZES_LIST[@]}"; do
                # Baseline Implementations
                for impl in "${BASELINE_IMPL_LIST[@]}"; do
                    copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
                    
                    if [[ "$dataset_config" == "sharegpt" ]]; then
                        run_benchmark_sharegpt "$model_path" "$model" "$gpus" "$impl" "sharegpt" "" "" "$extra_args" "$chunked_prefill_size"
                    else
                        run_benchmark_random "$model_path" "$model" "$gpus" "$impl" "random" "$dataset_config" "$OUTPUT_LEN" "$extra_args" "$chunked_prefill_size"
                    fi
                done

                # Overlap Fused Implementations
                for impl in "${OVERLAP_FUSED_IMPL_LIST[@]}"; do
                    copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
                    
                    if [[ "$dataset_config" == "sharegpt" ]]; then
                        run_benchmark_sharegpt "$model_path" "$model" "$gpus" "$impl" "sharegpt" "" "" "$extra_args" "$chunked_prefill_size"
                    else
                        run_benchmark_random "$model_path" "$model" "$gpus" "$impl" "random" "$dataset_config" "$OUTPUT_LEN" "$extra_args" "$chunked_prefill_size"
                    fi
                done
            done
        done
        rm -rf "$EVAL_DIR/$model"
    done
done
# ------------------ GPU Reset ------------------
log_info "Resetting GPU settings..."
nvidia-smi -pm ENABLED
nvidia-smi -rgc
rm -f "$SHAREGPT_FILE_PATH"
log_info "Benchmarking completed successfully."