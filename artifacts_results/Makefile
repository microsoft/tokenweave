# Configuration
OUTPUT ?= output/
GRAPHS ?= graphs/
CSVS   ?= csvs/
TMP    ?= tmp/

# Figure definitions
FIGURES = figure_2_11 figure_1_3 figure_5_6_7 figure_4_10 figure_9 figure_12 figure_13

# MODEL FILES
PY_FILES := llama.py qwen2.py mixtral.py

# Default target
.PHONY: all start clean install_dependencies install_miniconda correctness_check $(FIGURES) run_all gen_all
.DEFAULT_GOAL := all

# Create directories once
start:
	@mkdir -p $(OUTPUT) $(GRAPHS) $(CSVS) $(TMP)
	@for f in $(PY_FILES); do cp ../vllm/model_executor/models/$$f $(TMP)/$$f.orig; done

# Build all figures sequentially
all: $(FIGURES)

# Run all figure computations sequentially
run_all: start
	@echo "Running all figure computations sequentially..."
	@$(foreach fig,$(FIGURES),echo "Running $(fig)..."; bash $(fig)/run.sh $(OUTPUT);)
	@for f in $(PY_FILES); do mv $(TMP)/$$f.orig ../vllm/model_executor/models/$$f; done

# Generate all plots sequentially
gen_all: start
	@echo "Generating all plots sequentially..."
	@$(foreach fig,$(FIGURES),echo "Generating $(fig) plots..."; bash $(fig)/plot.sh $(OUTPUT) $(GRAPHS) $(CSVS);)

# Generic figure target - works for all figures
$(FIGURES): start
	@echo "Processing $@..."
	@bash $@/run.sh $(OUTPUT)
	@bash $@/plot.sh $(OUTPUT) $(GRAPHS) $(CSVS)
	@for f in $(PY_FILES); do mv $(TMP)/$$f.orig ../vllm/model_executor/models/$$f; done

# Correctness check
correctness_check: start
	@bash correctness_check/run.sh $(OUTPUT)
	@for f in $(PY_FILES); do mv $(TMP)/$$f.orig ../vllm/model_executor/models/$$f; done

# Configs Generator
configs_generator: start
	@bash configs_generator/run.sh $(OUTPUT)
	@for f in $(PY_FILES); do mv $(TMP)/$$f.orig ../vllm/model_executor/models/$$f; done

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf $(OUTPUT) $(GRAPHS) $(CSVS) $(TMP)

# Restore original files
restore_files:
	@cp llama_src_files/pl_overlap_fused.py ../vllm/model_executor/models/llama.py
	@cp qwen2_src_files/pq_overlap_fused.py ../vllm/model_executor/models/qwen2.py
	@cp mixtral_src_files/pm_overlap_fused.py ../vllm/model_executor/models/mixtral.py
	@cp gpu_model_runner.py ../vllm/v1/worker/gpu_model_runner.py

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build all figures sequentially (default)"
	@echo "  run_all          - Run all figure computations sequentially"
	@echo "  gen_all          - Generate all plots sequentially"
	@echo "  <figure_name>    - Build specific figure (e.g., make figure_5_6_7)"
	@echo "  correctness_check - Run correctness check"
	@echo "  clean            - Remove generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Available figures: $(FIGURES)"