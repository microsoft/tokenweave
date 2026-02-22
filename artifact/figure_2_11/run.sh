#!/bin/bash

# ------------------ Configuration ------------------
EVAL_DIR=$1
MASTER_CSV="$EVAL_DIR/figure_2_11_summary.csv"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_LEN=1
BATCH_SIZE=1
NUM_ITERS=30
NUM_ITERS_WARMUP=10
NUM_GPUS_LIST=(8)
MODEL_NAME_LIST=("Llama-3.3-70B-Instruct" "Qwen2.5-72B-Instruct" "Mixtral-8x22B-Instruct-v0.1")
BASELINE_IMPL_LIST=("baseline_multimem" "no_ar")
OVERLAP_FUSED_IMPL_LIST=("overlap_fused")

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
    echo "model_name,gpu_count,impl,prompt_len,avg_latency(ms)" > "$MASTER_CSV"
}

copy_model_files() {
    local model=$1 impl=$2 src_dir=$3 prefix=$4 dst_file_path=$5
    log_info "Copying model files for $model, implementation: $impl"
    cp "$src_dir"/"$prefix""$impl".py "$dst_file_path"
}

extract_latency() {
    local output_log=$1 model=$2 gpus=$3 impl=$4
    grep "Input length:" "$output_log" | while read -r line; do
        input_len=$(echo "$line" | awk -F'[:,]' '{print $2}' | xargs)
        avg_latency=$(echo "$line" | awk -F'[:,]' '{print $4}' | xargs)
        echo "$model,$gpus,$impl,$input_len,$avg_latency" >> "$MASTER_CSV"
    done
}

# ------------------ Benchmark Functions ------------------

run_benchmark() {
    local model_path=$1 model=$2 gpus=$3 impl=$4 extra_args=$5
    local dir="$EVAL_DIR/$model/${gpus}xh100/$impl"
    mkdir -p "$dir/latency_logs"
    local output_log="$dir/latency_logs/${impl}.txt"

    python "$SCRIPT_DIR/bench.py" \
        --output-len "$OUTPUT_LEN" --batch-size "$BATCH_SIZE" \
        --num-iters "$NUM_ITERS" --num-iters-warmup "$NUM_ITERS_WARMUP" \
        --output-log "$output_log" --disable-detokenize --model "$model_path" \
        --tensor-parallel-size "$gpus" --enable-chunked-prefill --enforce-eager \
        --disable-custom-all-reduce --max-num-batched-tokens 65536 --max-model-len 131072 \
        --no-enable-prefix-caching $extra_args | tee -a "$output_log"

    extract_latency "$output_log" "$model" "$gpus" "$impl"
    rm -rf "$dir"
}

# ------------------ Main Execution ------------------

log_info "Setting up the environment..."
init_benchmark

cp -rf "$SCRIPT_DIR/prefill_configs"/* "$SCRIPT_DIR/../../vllm/tokenweave_configs"

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
        log_info "Running benchmark for $model on $gpus GPUs..."
        # Baseline Implementations
        for impl in "${BASELINE_IMPL_LIST[@]}"; do
            copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
            run_benchmark "$model_path" "$model" "$gpus" "$impl" "$extra_args"
        done

        # Overlap Fused Implementations
        for impl in "${OVERLAP_FUSED_IMPL_LIST[@]}"; do
            copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
            run_benchmark "$model_path" "$model" "$gpus" "$impl" "$extra_args"
        done
    done
    rm -rf "$EVAL_DIR/$model"
done

# ------------------ GPU Reset ------------------
log_info "Resetting GPU settings..."
nvidia-smi -pm ENABLED
nvidia-smi -rgc
log_info "Benchmarking completed successfully."