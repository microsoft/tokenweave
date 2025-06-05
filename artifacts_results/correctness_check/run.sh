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
DATASET_NAME=("sharegpt")
DATASET_PATH="$SHAREGPT_FILE_PATH"
NUM_GPUS_LIST=(8)
MODEL_NAME_LIST=("Llama-3.3-70B-Instruct" "Qwen2.5-72B-Instruct" "Mixtral-8x22B-Instruct-v0.1")
BASELINE_IMPL_LIST=("baseline_multimem")
OVERLAP_FUSED_IMPL_LIST=("overlap_fused")
CHUNKED_PREFILL_SIZES_LIST=(1024 2048)
extra_args=""

# ------------------ Environment Setup ------------------
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ------------------ Utility Functions ------------------
copy_model_files() {
    local model=$1 impl=$2 src_dir=$3 prefix=$4 dst_file_path=$5
    cp "$src_dir"/"$prefix""$impl".py "$dst_file_path"
}

# ------------------ Benchmark Functions ------------------

run_benchmark_sharegpt() {
    local model_path=$1 model=$2 gpus=$3 impl=$4 dataset=$5 chunked_prefill_size=$6 extra_args=$7
    local dir="$EVAL_DIR/CORRECTNESS_$model/${gpus}xh100"
    mkdir -p "$dir"
    local output_log="$dir/${impl}.txt"
    python "$SCRIPT_DIR/bench.py" \
        --dataset-name $dataset --dataset-path "$DATASET_PATH" \
        --model "$model_path" \
        --tensor-parallel-size "$gpus" --enable-chunked-prefill --enforce-eager \
        --disable-custom-all-reduce --max-num-batched-tokens "$chunked_prefill_size" \
        --no-enable-prefix-caching $extra_args | tee -a "$output_log"
}

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
            # ShareGPT
            # Baseline Implementations
            for impl in "${BASELINE_IMPL_LIST[@]}"; do
                copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
                run_benchmark_sharegpt "$model_path" "$model" "$gpus" "$impl" "sharegpt" "$chunked_prefill_size" "$extra_args"
            done

            # Overlap Fused Implementations
            for impl in "${OVERLAP_FUSED_IMPL_LIST[@]}"; do
                copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
                run_benchmark_sharegpt "$model_path" "$model" "$gpus" "$impl" "sharegpt" "$chunked_prefill_size" "$extra_args"
            done
        done
    done
done
# ------------------ GPU Reset ------------------
rm -f "$SHAREGPT_FILE_PATH"