#!/bin/bash
#SBATCH --job-name=baseline-eval
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Baseline single-model evaluation — sends images directly to a VLM.
#
# Usage:
#   sbatch run_baseline_gpu.sh                           # default: gemma4:27b
#   MODEL=qwen3-vl:32b sbatch run_baseline_gpu.sh       # different model
#
# Chain multiple:
#   for i in $(seq 1 5); do sbatch --parsable --dependency=singleton run_baseline_gpu.sh; done
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/llm_council_agent_orchestration_tests}"
cd "$PROJECT_DIR"
mkdir -p logs

MODEL="${MODEL:-gemma4:31b}"
OUTPUT_NAME="${OUTPUT_NAME:-baseline_${MODEL//:/_}_result.json}"
OUTPUT_CSV="${OUTPUT_CSV:-baseline_${MODEL//:/_}_results.csv}"

echo "=== Baseline Evaluation (GPU) ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "Model    : $MODEL"
echo "Output   : $OUTPUT_NAME"
echo ""

# ── GPU check ────────────────────────────────────────────────────────────────
echo "[gpu] nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Start Ollama with GPU ────────────────────────────────────────────────────
OLLAMA_PATH=$(ws_find ollama_models 2>/dev/null || echo "")
if [ -z "$OLLAMA_PATH" ]; then
    echo "[ERROR] ws_find ollama_models failed."
    exit 1
fi

export PATH=$OLLAMA_PATH/bin:$PATH
export OLLAMA_MODELS=$OLLAMA_PATH/models

pkill -u $USER ollama 2>/dev/null || true
sleep 1

echo "[ollama] Starting Ollama server ..."
OLLAMA_HOST=0.0.0.0 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 \
    ollama serve > logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

echo "[ollama] Waiting for server ..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; then
        echo "[ollama] Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; then
    echo "[ERROR] Ollama did not start."
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

echo "[ollama] Available models:"
curl -s "http://127.0.0.1:11434/api/tags" | python3 -c "import sys,json; [print(f'  {m[\"name\"]}') for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null || true

# Warm up the model
echo "[ollama] Warming up $MODEL ..."
curl -sf http://127.0.0.1:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" > /dev/null 2>&1 || true
sleep 2

GPU_LAYERS=$(grep -o "offloaded [0-9]*/[0-9]* layers to GPU" logs/ollama_${SLURM_JOB_ID}.log | tail -1)
echo "[ollama] $GPU_LAYERS"
echo ""

export OLLAMA_HOST=http://127.0.0.1:11434
export LLM_PROVIDER=ollama

# ── Python environment ───────────────────────────────────────────────────────
module load devel/python/3.12 2>/dev/null || true

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --quiet
PYTHON="uv run python"

export PYTHONUNBUFFERED=1

echo "[env] Python: $($PYTHON --version)"
echo ""

# ── Run baseline evaluation ──────────────────────────────────────────────────
echo "[run] Starting baseline eval with $MODEL ..."
$PYTHON baseline_eval.py --all \
    --model "$MODEL" \
    --results-dir "results GeoRC" \
    --output-name "$OUTPUT_NAME" \
    --clean-image-dir "clean_images_GeoRC" \
    --concurrency 1 \
    --verbose

EXIT_CODE=$?

# ── Cleanup ──────────────────────────────────────────────────────────────────
kill $OLLAMA_PID 2>/dev/null || true
wait $OLLAMA_PID 2>/dev/null || true

echo ""
echo "=== Done: $(date) (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
