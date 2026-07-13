#!/bin/bash
#SBATCH --job-name=vlm-initial-vision
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vision-%j.out
#SBATCH --error=logs/vision-%j.err

# Vision pipeline batch runner on bwUniCluster 3.
# Assumes ws_find datasets is symlinked as ./Images.

set -euo pipefail

mkdir -p logs results

module load devel/cuda/12.6 || true
source .venv/bin/activate

# --- Ollama on the compute node ---
export PATH="$(ws_find ollama_models)/bin:$PATH"

OLLAMA_HOST=0.0.0.0 \
OLLAMA_MAX_LOADED_MODELS=3 \
OLLAMA_NUM_PARALLEL=4 \
ollama serve &
OLLAMA_PID=$!

trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# wait for readiness
for i in {1..30}; do
  curl -s http://localhost:11434/api/tags > /dev/null && break
  sleep 2
done

# --- Model pulls (idempotent) ---
ollama pull qwen3-vl:32b
ollama pull gemma4:26b
ollama pull qwen2.5vl:32b

# --- Pipeline config ---
export OLLAMA_HOST="http://localhost:11434"
export VISION_MODEL="qwen3-vl:32b"
export TEXT_MODEL="gemma4:26b"
export GROUNDING_MODEL="qwen2.5vl:32b"
export MAX_DETAILS=5

# --- Batch run: first 100 images ---
python -m vision_pipeline.batch Images/ results/ --limit 100

echo "Done."
