.PHONY: help install clean run_llama3 run_qwen2 run_mixtral

# Default target
help:
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install project in editable mode"
	@echo "  make run_llama3     Run with model: llama3"
	@echo "  make run_qwen2      Run with model: qwen2"
	@echo "  make run_mixtral    Run with model: mixtral"
	@echo "  install_dependencies - Install Python dependencies"
	@echo "  install_miniconda - Install Miniconda"
	@echo "  create_env - create environment"
	@echo "  make clean          Remove build artifacts, __pycache__, .so files"
	@echo ""

install:
	@pip3 install -vvv -e .

run_llama3:
	@python3 tokenweave_offline_example.py --model llama3

run_qwen2:
	@python3 tokenweave_offline_example.py --model qwen2

run_mixtral:
	@python3 tokenweave_offline_example.py --model mixtral

# Dependency installation
install_dependencies:
	@echo "Installing Python dependencies..."
	@pip3 install matplotlib pandas seaborn
	@pip install --upgrade huggingface_hub
	@pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl

# Install Miniconda only
install_miniconda:
	@echo "Installing Miniconda..."
	@mkdir -p ~/miniconda3
	@if [ ! -f ~/miniconda3/miniconda.sh ]; then \
		echo "Downloading Miniconda..."; \
		wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh; \
	fi
	@if [ ! -d ~/miniconda3/bin ]; then \
		echo "Installing Miniconda..."; \
		bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3; \
		~/miniconda3/condabin/conda init; \
	fi

# Create tokenweave environment
create_env:
	@echo "Checking for tokenweave environment..."
	@if ! ~/miniconda3/bin/conda env list | grep -q tokenweave 2>/dev/null; then \
		echo "Creating tokenweave environment..."; \
		~/miniconda3/bin/conda create -n tokenweave python=3.12 -y; \
	else \
		echo "tokenweave environment already exists"; \
	fi

clean:
	@echo "Cleaning project..."
	@bash -c 'find . \( -type d -name "__pycache__" -o -type d -name ".deps" \) -exec rm -rf {} +'
	@bash -c 'find . -type f -name "*.so" -exec rm -f {} +'
	@rm -rf build dist *.egg-info raw_output*.txt debug*.txt
	@rm -rf .venv venv
	@rm -f vllm/_version.py
	@pip3 uninstall vllm -y
	@echo "Done."
