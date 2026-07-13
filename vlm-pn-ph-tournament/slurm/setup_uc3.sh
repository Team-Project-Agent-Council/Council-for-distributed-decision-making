#!/bin/bash
# Setup script for bwForCluster Helix.
# Run this once on the login node after cloning the repo into the workspace or uploading the project files.
#
# Prerequisites:
#   1. ws_allocate vlm-council 90
#   2. ws_allocate hf-cache 90
#   3. cd $(ws_find vlm-council)
#   4. Upload the project files into the workspace.
#   5. bash setup.sh

set -euo pipefail

echo "=== VLM Council Setup ==="

PROJECT_DIR="$(pwd)"

# 1. Load modules
echo "Loading modules..."
module load devel/python/3.13.1
module load devel/cuda/12.6

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "Installing vLLM and dependencies (this takes a few minutes)..."
pip install --upgrade pip
pip cache purge 2>/dev/null || true

# vLLM: install pre-built CUDA 12.6 wheel.
pip install "vllm>=0.8" \
    --no-cache-dir \
    --only-binary vllm \
    --extra-index-url https://wheels.vllm.ai/cu126/torch2.7.0

pip install langchain-core langchain-openai langgraph

# 4. Set up HuggingFace cache in workspace (home is too small)
HF_WS="$(ws_find hf-cache 2>/dev/null || echo "$PROJECT_DIR/.cache")"
export HF_HOME="$HF_WS/huggingface"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

# 5. Copy env template
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from template."
fi

# 6. Check for images
if [ ! -d "Images" ]; then
    echo ""
    echo "WARNING: No Images/ directory found."
    echo "  Copy or symlink your images:"
    echo "  ln -s /path/to/your/images Images"
    echo "  Or: cp -r /source/images Images/"
fi

echo ""
echo "=== Setup complete ==="
echo ""