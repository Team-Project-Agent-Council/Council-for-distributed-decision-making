#!/bin/bash
#SBATCH --job-name=council-eval-gpu
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_gpu_%j.out
#SBATCH --error=logs/eval_gpu_%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com   # uncomment and fill in

# =============================================================================
# Self-contained council evaluation job - starts Ollama on the GPU node,
# runs the evaluation pipeline, then cleans up.
#
# Submit:  sbatch run_eval_gpu.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/eval_gpu_<jobid>.out
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "$PROJECT_DIR"
# Shared vision_pipeline package lives one level up (monorepo root)
export PYTHONPATH="$(cd "$PROJECT_DIR/.." && pwd):${PYTHONPATH:-}"
mkdir -p logs

echo "=== Council Evaluation (GPU) ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "Work dir : $PROJECT_DIR"
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

# Kill any leftover Ollama from previous jobs
pkill -u $USER ollama 2>/dev/null || true
sleep 1

# Start Ollama - match the interactive setup as closely as possible
# (OLLAMA_HOST=0.0.0.0 without explicit port = default 11434)
echo "[ollama] Starting Ollama server ..."
OLLAMA_HOST=0.0.0.0 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_NUM_PARALLEL=5 \
    ollama serve > logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready (up to 60s)
echo "[ollama] Waiting for server to start ..."
for i in $(seq 1 60); do
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
    echo "[ERROR] Ollama did not start within 60s. Log:"
    tail -20 logs/ollama_${SLURM_JOB_ID}.log
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# Show loaded models
echo "[ollama] Available models:"
curl -s "http://127.0.0.1:11434/api/tags" | python3 -c "import sys,json; [print(f'  {m[\"name\"]}') for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null || true
echo ""

# Show Ollama server log (for debugging model discovery)
echo "[ollama] Server log snippet:"
head -20 logs/ollama_${SLURM_JOB_ID}.log
echo ""

# Verify qwen3:32b is available
if ! curl -sf "http://127.0.0.1:11434/api/tags" | python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; sys.exit(0 if any('qwen3' in m and '32b' in m for m in models) else 1)"; then
    echo "[ERROR] qwen3:32b not found in model list. Available models:"
    curl -s "http://127.0.0.1:11434/api/tags" | python3 -c "import sys,json; [print(f'  {m[\"name\"]}') for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null || true
    echo ""
    echo "[debug] OLLAMA_MODELS dir:"
    echo "  HOME=$HOME"
    echo "  ~/.ollama is: $(ls -la ~/.ollama 2>/dev/null | head -1)"
    ls ~/.ollama/models/manifests/registry.ollama.ai/library/ 2>/dev/null || echo "  (not found)"
    echo ""
    echo "[debug] Workspace models dir:"
    ls $OLLAMA_PATH/models/manifests/registry.ollama.ai/library/ 2>/dev/null || echo "  (not found)"
    echo ""
    echo "[debug] Ollama log:"
    cat logs/ollama_${SLURM_JOB_ID}.log
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# Quick smoke test to warm up and verify GPU offload
echo "[ollama] Warming up qwen3:32b ..."
curl -sf http://127.0.0.1:11434/api/generate -d '{"model":"qwen3:32b","prompt":"hi","stream":false,"options":{"num_predict":1}}' > /dev/null 2>&1 || true
sleep 2

# Check GPU offload in log
if grep -q "offloaded 0/" logs/ollama_${SLURM_JOB_ID}.log; then
    echo "[WARN] Model may be running on CPU. Check log for details."
    grep "offloaded" logs/ollama_${SLURM_JOB_ID}.log || true
fi

GPU_LAYERS=$(grep -o "offloaded [0-9]*/[0-9]* layers to GPU" logs/ollama_${SLURM_JOB_ID}.log | tail -1)
echo "[ollama] $GPU_LAYERS"
echo ""

# -- Point all agents at local Ollama (use 127.0.0.1 to avoid IPv6) -----------
export ORCHESTRATOR_OLLAMA_HOST=http://127.0.0.1:11434
export LINGUISTIC_OLLAMA_HOST=http://127.0.0.1:11434
export LANDSCAPE_OLLAMA_HOST=http://127.0.0.1:11434
export BOTANICS_OLLAMA_HOST=http://127.0.0.1:11434
export REGULATORY_OLLAMA_HOST=http://127.0.0.1:11434
export INFRASTRUCTURE_OLLAMA_HOST=http://127.0.0.1:11434
export CLIMATE_OLLAMA_HOST=http://127.0.0.1:11434
export META_OLLAMA_HOST=http://127.0.0.1:11434
export JUDGE_OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_HOST=http://127.0.0.1:11434
export VISION_OLLAMA_HOST=http://127.0.0.1:11434

echo "[config] Agent Ollama hosts:"
env | grep _OLLAMA_HOST | sort
echo ""

# -- Python environment -------------------------------------------------------
module load devel/python/3.12 2>/dev/null || true

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "[env] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[env] uv: $(uv --version)"
echo "[env] Syncing dependencies ..."
uv sync --quiet
uv pip install pysqlite3-binary --quiet
PYTHON="uv run python"

# Fix: cluster sqlite3 is too old for chromadb
# Set env var so our sitecustomize patches it before chromadb loads
export CHROMA_SQLITE_PATCH=1

echo "[env] Python: $($PYTHON --version)"
echo ""

# -- Smoke test: verify LLM calls reach Ollama --------------------------------
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
" || { echo "[ERROR] ChatOllama smoke test failed - LLM calls cannot reach Ollama. Aborting."; kill $OLLAMA_PID 2>/dev/null; exit 1; }
echo ""

# -- Run evaluation -----------------------------------------------------------
export PYTHONUNBUFFERED=1
# Shared monorepo data lives two levels up (VLM Initial Approach root).
CONCURRENCY="${CONCURRENCY:-1}"
RESULTS_DIR="${RESULTS_DIR:-../results}"
MAPPING="${MAPPING:-../results/llm_council_evals/location-image-mapping.csv}"
OUTPUT="${OUTPUT:-additional_agents.csv}"
echo "[run] Starting evaluation (concurrency=$CONCURRENCY) ..."
echo "[run] results-dir=$RESULTS_DIR  mapping=$MAPPING  output=$OUTPUT"
$PYTHON evaluate.py \
    --results-dir "$RESULTS_DIR" \
    --mapping "$MAPPING" \
    --output "$OUTPUT" \
    --concurrency "$CONCURRENCY" \
    --verbose \
    --run-name "council-eval-$SLURM_JOB_ID"

EXIT_CODE=$?

# -- Cleanup ------------------------------------------------------------------
echo ""
echo "[cleanup] Stopping Ollama ..."
kill $OLLAMA_PID 2>/dev/null || true
wait $OLLAMA_PID 2>/dev/null || true

echo ""
echo "=== Done: $(date) (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
