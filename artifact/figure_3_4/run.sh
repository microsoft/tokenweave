#!/bin/bash
nvidia-smi -pm ENABLED
nvidia-smi -lgc tdp
RESULTS_DIR=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$SCRIPT_DIR/bench.py"
HIDDEN_SIZE=(8192)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 "$script_name" \
    --hidden-size "$HIDDEN_SIZE" \
    --output-dir "$RESULTS_DIR" > "$RESULTS_DIR/figure_3_4_hs_$HIDDEN_SIZE.txt" 2>&1

sleep 3
nvidia-smi -pm ENABLED
nvidia-smi -rgc