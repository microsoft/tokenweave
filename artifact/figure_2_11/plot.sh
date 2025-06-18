#!/bin/bash

RESULTS_DIR=$1
GRAPHS_DIR=$2
CSVS_DIR=$3
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 "$SCRIPT_DIR/plot_figure_2.py" "$RESULTS_DIR" "$GRAPHS_DIR" "$CSVS_DIR"
python3 "$SCRIPT_DIR/plot_figure_11.py" "$RESULTS_DIR" "$GRAPHS_DIR" "$CSVS_DIR"