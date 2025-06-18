#!/bin/bash

# ------------------ Configuration ------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$1
MASTER_CSV="$EVAL_DIR/configs_generator_summary.csv"
OUTPUT_LEN=1
BATCH_SIZE=1
NUM_ITERS=30
NUM_ITERS_WARMUP=10
NUM_GPUS_LIST=(8 4)
CHUNK_OFFSET_LIST=(0 128 256)
MAX_CTAS_MLP_LIST=(2 4 8)
MAX_CTAS_ATTN_LIST=(2 4 8)
MAX_CTAS_LIST=(16 32)
MODEL_NAME_LIST=("Llama-3.3-70B-Instruct" "Qwen2.5-72B-Instruct" "Mixtral-8x22B-Instruct-v0.1")
BASELINE_FUSED_IMPL_LIST=("baseline_fused")
OVERLAP_FUSED_IMPL_LIST=("overlap_fused")
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
    echo "model_name,gpu_count,impl,chunk_offset,attn_ctas,mlp_ctas,prompt_len,avg_latency(ms)" > "$MASTER_CSV"
}

copy_model_files() {
    local model=$1 impl=$2 src_dir=$3 prefix=$4 dst_file_path=$5
    log_info "Copying model files for $model, implementation: $impl"
    cp "$src_dir"/"$prefix""$impl".py "$dst_file_path"
}

extract_latency() {
    local output_log=$1 model=$2 gpus=$3 impl=$4 offset=$5 attn_ctas=$6 mlp_ctas=$7
    grep "Input length:" "$output_log" | while read -r line; do
        input_len=$(echo "$line" | awk -F'[:,]' '{print $2}' | xargs)
        avg_latency=$(echo "$line" | awk -F'[:,]' '{print $4}' | xargs)
        echo "$model,$gpus,$impl,$offset,$attn_ctas,$mlp_ctas,$input_len,$avg_latency" >> "$MASTER_CSV"
    done
}

# ------------------ Benchmark Functions ------------------

run_benchmark() {
    local model_path=$1 model=$2 gpus=$3 impl=$4 offset=$5 attn_ctas=$6 mlp_ctas=$7 extra_args=$8
    local dir="$EVAL_DIR/$model/${gpus}xh100"
    mkdir -p "$dir"
    local output_log="$dir/${impl}.txt"

    python "$SCRIPT_DIR/bench.py" \
        --output-len "$OUTPUT_LEN" --batch-size "$BATCH_SIZE" \
        --num-iters "$NUM_ITERS" --num-iters-warmup "$NUM_ITERS_WARMUP" \
        --output-log "$output_log" --disable-detokenize --model "$model_path" \
        --tensor-parallel-size "$gpus" --enable-chunked-prefill --enforce-eager \
        --disable-custom-all-reduce --max-num-batched-tokens 65536 --max-model-len 131072 \
        --no-enable-prefix-caching $extra_args | tee -a "$output_log"

    extract_latency "$output_log" "$model" "$gpus" "$impl" "$offset" "$attn_ctas" "$mlp_ctas"
    rm -rf "$dir"
}

# ------------------ Main Execution ------------------

log_info "Setting up the environment..."
init_benchmark

cp "$SCRIPT_DIR/../../vllm/v1/worker/gpu_model_runner.py" "$SCRIPT_DIR/src_files/gpu_model_runner.py.orig"
cp "$SCRIPT_DIR/src_files/gpu_model_runner.py" "$SCRIPT_DIR/../../vllm/v1/worker/gpu_model_runner.py"

for model in "${MODEL_NAME_LIST[@]}"; do
    case "$model" in
        "Llama-3.3-70B-Instruct")
            model_path="meta-llama/Llama-3.3-70B-Instruct"
            prefix="llama_"
            src_dir="$SCRIPT_DIR/src_files"
            dst_file_path="$SCRIPT_DIR/../../vllm/model_executor/models/llama.py"
            extra_args=""
            ;;
        "Qwen2.5-72B-Instruct")
            model_path="Qwen/Qwen2.5-72B-Instruct"
            prefix="qwen2_"
            src_dir="$SCRIPT_DIR/src_files"
            dst_file_path="$SCRIPT_DIR/../../vllm/model_executor/models/qwen2.py"
            extra_args=""
            ;;
        "Mixtral-8x22B-Instruct-v0.1")
            model_path="mistralai/Mixtral-8x22B-Instruct-v0.1"
            prefix="mixtral_"
            src_dir="$SCRIPT_DIR/src_files"
            dst_file_path="$SCRIPT_DIR/../../vllm/model_executor/models/mixtral.py"
            extra_args="--tokenizer_mode mistral"
            ;;
        *)
            log_info "Unknown model: $model"
            continue
            ;;
    esac

    for gpus in "${NUM_GPUS_LIST[@]}"; do
        # Baseline Fused Implementations
        for ctas in "${MAX_CTAS_LIST[@]}"; do
            for impl in "${BASELINE_FUSED_IMPL_LIST[@]}"; do
                copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
                export MAX_CTAS_ATTN=$ctas
                export MAX_CTAS_MLP=$ctas
                run_benchmark "$model_path" "$model" "$gpus" "$impl" "" "$ctas" "$ctas" "$extra_args"
            done
        done

        # Overlap Fused Implementations
        for offset in "${CHUNK_OFFSET_LIST[@]}"; do
            for attn_ctas in "${MAX_CTAS_ATTN_LIST[@]}"; do
                for mlp_ctas in "${MAX_CTAS_MLP_LIST[@]}"; do
                    for impl in "${OVERLAP_FUSED_IMPL_LIST[@]}"; do
                        copy_model_files "$model" "$impl" "$src_dir" "$prefix" "$dst_file_path"
                        export CHUNK_OFFSET=$offset
                        export MAX_CTAS_ATTN=$attn_ctas
                        export MAX_CTAS_MLP=$mlp_ctas
                        run_benchmark "$model_path" "$model" "$gpus" "$impl" "$offset" "$attn_ctas" "$mlp_ctas" "$extra_args"
                    done
                done
            done
        done
    done
    rm -rf "$EVAL_DIR/$model"
done

# ------------------ GPU Reset ------------------
log_info "Resetting GPU settings..."
nvidia-smi -pm ENABLED
nvidia-smi -rgc
log_info "Benchmarking completed successfully."

mv "$SCRIPT_DIR/src_files/gpu_model_runner.py.orig" "$SCRIPT_DIR/../../vllm/v1/worker/gpu_model_runner.py"
CONFIGS_DIR="$SCRIPT_DIR/generated_tokenweave_configs"
mkdir -p "$CONFIGS_DIR"

python3 "$SCRIPT_DIR/generator.py" "$EVAL_DIR" "$CONFIGS_DIR"