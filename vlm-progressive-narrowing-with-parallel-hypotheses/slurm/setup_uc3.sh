#!/bin/bash
# Setup for bwUniCluster 3 (uc3).
# Run ONCE on the login node.

# Prerequisites:
#   1. ws_allocate vlm-council-pn 30
#   2. ws_allocate hf-cache 30
#   3. cd $(ws_find vlm-council-pn)
#   4. Upload the project files into the workspace.
#   5. bash slurm/setup_uc3.sh

set -euo pipefail

echo "=== VLM Council: uc3 Setup ==="

PROJECT_DIR="$(pwd)"

# 1. Load modules
echo "Loading modules..."
module load devel/python/3.13.1
module load devel/cuda/12.8

echo "Python: $(python3 --version 2>&1)"

# 2. Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "Installing vLLM and dependencies..."
pip install --upgrade pip
pip cache purge 2>/dev/null || true

pip install "vllm>=0.8" \
    --no-cache-dir \
    --only-binary vllm \
    --extra-index-url https://wheels.vllm.ai/cu126/torch2.7.0

pip install langchain-core langchain-openai langgraph

# 4. HuggingFace cache → workspace
HF_WS="$(ws_find hf-cache 2>/dev/null || echo "$PROJECT_DIR/.cache")"
export HF_HOME="$HF_WS/huggingface"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

# 5. Check images
if [ ! -d "Images" ]; then
    echo ""
    echo "WARNING: No Images/ directory found."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run:  sbatch slurm/run_council_uc3.sh --limit 1"
