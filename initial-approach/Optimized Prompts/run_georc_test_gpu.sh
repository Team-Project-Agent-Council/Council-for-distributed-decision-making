#!/bin/bash
#SBATCH --job-name=georc-test-gpu
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/georc_test_%j.out
#SBATCH --error=logs/georc_test_%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com   # uncomment and fill in

# =============================================================================
# GeoRC single-image test - runs the council on one pre-computed vision result.
#
# Usage:
#   LOCATION_ID=67gLC5CcGgkQIWEW_2 sbatch run_georc_test_gpu.sh
#
# Monitor: squeue -u $USER
# Logs:    tail -f logs/georc_test_<jobid>.out
# =============================================================================

set -euo pipefail

if [ -z "${LOCATION_ID:-}" ]; then
    echo "[ERROR] LOCATION_ID is not set."
    echo "Usage: LOCATION_ID=67gLC5CcGgkQIWEW_2 sbatch run_georc_test_gpu.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "$PROJECT_DIR"
# Shared vision_pipeline package lives one level up (monorepo root)
export PYTHONPATH="$(cd "$PROJECT_DIR/.." && pwd):${PYTHONPATH:-}"
mkdir -p logs

echo "=== GeoRC Council Test (GPU) ==="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "Work dir    : $PROJECT_DIR"
echo "Location ID : $LOCATION_ID"
echo ""

# -- GPU check ----------------------------------------------------------------
echo "[gpu] nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# -- Start Ollama with GPU ----------------------------------------------------
OLLAMA_PATH=$(ws_find ollama_models 2>/dev/null || echo "")
if [ -z "$OLLAMA_PATH" ]; then
    echo "[ERROR] ws_find ollama_models failed. Is the workspace allocated?"
    exit 1
fi
echo "[ollama] Ollama path: $OLLAMA_PATH"

export PATH=$OLLAMA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$OLLAMA_PATH/lib/ollama/cuda_v12:$OLLAMA_PATH/lib/ollama:${LD_LIBRARY_PATH:-}

# Tell Ollama where models live (workspace is shared across nodes)
export OLLAMA_MODELS=$OLLAMA_PATH/models

export OLLAMA_LLM_LIBRARY=cuda_v12

unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_NUM_PARALLEL=5

echo "[ollama] Starting Ollama server ..."
ollama serve > logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

echo "[ollama] Waiting for server to start ..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; then
        echo "[ollama] Server reachable after ${i}s [ok]"
        break
    fi
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "[ERROR] Ollama process died. Log:"
        tail -20 logs/ollama_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; then
    echo "[ERROR] Ollama did not start within 30s. Log:"
    tail -20 logs/ollama_${SLURM_JOB_ID}.log
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

echo "[ollama] Available models:"
curl -s "http://127.0.0.1:11434/api/tags" | python3 -c "import sys,json; [print(f'  {m[\"name\"]}') for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null || true

echo "[ollama] Triggering test request to verify GPU offload ..."
curl -sf http://127.0.0.1:11434/api/generate -d '{"model":"qwen3:32b","prompt":"hi","stream":false,"options":{"num_predict":1}}' > /dev/null 2>&1 || true
sleep 2

if grep -q "offloaded 0/65 layers to GPU" logs/ollama_${SLURM_JOB_ID}.log; then
    echo "[ERROR] Model running on CPU. Aborting."
    tail -30 logs/ollama_${SLURM_JOB_ID}.log
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

GPU_LAYERS=$(grep -o "offloaded [0-9]*/65 layers to GPU" logs/ollama_${SLURM_JOB_ID}.log | tail -1)
echo "[ollama] $GPU_LAYERS"
echo ""

# -- Point all agents at local Ollama -----------------------------------------
export ORCHESTRATOR_OLLAMA_HOST=http://127.0.0.1:11434
export LINGUISTIC_OLLAMA_HOST=http://127.0.0.1:11434
export LANDSCAPE_OLLAMA_HOST=http://127.0.0.1:11434
export BOTANICS_OLLAMA_HOST=http://127.0.0.1:11434
export REGULATORY_OLLAMA_HOST=http://127.0.0.1:11434
export META_OLLAMA_HOST=http://127.0.0.1:11434
export JUDGE_OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_HOST=http://127.0.0.1:11434
export VISION_OLLAMA_HOST=http://127.0.0.1:11434

# -- Python environment -------------------------------------------------------
module load devel/python/3.12 2>/dev/null || true

if ! command -v uv &>/dev/null; then
    echo "[env] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --quiet
uv pip install pysqlite3-binary --quiet
PYTHON="uv run python"

export CHROMA_SQLITE_PATCH=1
export PYTHONUNBUFFERED=1

echo "[env] Python: $($PYTHON --version)"
echo ""

# -- Smoke test ---------------------------------------------------------------
echo "[test] Verifying ChatOllama can reach Ollama ..."
$PYTHON -c "
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

async def test():
    llm = ChatOllama(model='qwen3:32b', base_url='http://127.0.0.1:11434', num_predict=5, client_kwargs={'timeout': 30.0})
    resp = await asyncio.wait_for(llm.ainvoke([HumanMessage(content='Say hi')]), timeout=60)
    print(f'[test] LLM responded: {resp.content[:50]!r}')

asyncio.run(test())
" || { echo "[ERROR] ChatOllama smoke test failed. Aborting."; kill $OLLAMA_PID 2>/dev/null; exit 1; }
echo ""

# -- Run council for single image ---------------------------------------------
OUTPUT_NAME="${OUTPUT_NAME:-council_optimized_qwen_result.json}"
echo "[run] Running council for location: $LOCATION_ID ..."
echo "[run] Output filename: $OUTPUT_NAME"
$PYTHON georc_test.py "$LOCATION_ID" \
    --results-dir ../results \
    --output-name "$OUTPUT_NAME" \
    --verbose

EXIT_CODE=$?

# -- Cleanup ------------------------------------------------------------------
echo ""
echo "[cleanup] Stopping Ollama ..."
kill $OLLAMA_PID 2>/dev/null || true
wait $OLLAMA_PID 2>/dev/null || true

echo ""
echo "=== Done: $(date) (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
