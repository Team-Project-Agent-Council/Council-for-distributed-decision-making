#!/bin/bash
#SBATCH --job-name=baseline-gemma4
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/baseline_gemma4_%j.out
#SBATCH --error=logs/baseline_gemma4_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Baseline Gemma 4 evaluation with vLLM — proper vision budget (1120 tokens).
#
# Usage:
#   sbatch run_baseline_gemma4.sh
#
# Chain:
#   for i in $(seq 1 5); do sbatch --parsable --dependency=singleton run_baseline_gemma4.sh; done
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/llm_council_agent_orchestration_tests}"
cd "$PROJECT_DIR"
mkdir -p logs

MODEL_NAME="google/gemma-4-31b-it"
VISION_BUDGET="${VISION_BUDGET:-1120}"
VLLM_PORT=8234
THINKING="${THINKING:-false}"

# Set output name based on thinking mode
if [ "$THINKING" = "true" ]; then
    OUTPUT_NAME="${OUTPUT_NAME:-baseline_gemma4_thinking_result.json}"
else
    OUTPUT_NAME="${OUTPUT_NAME:-baseline_gemma4_vllm_result.json}"
fi

echo "=== Baseline Gemma 4 Evaluation (vLLM) ==="
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $(hostname)"
echo "Started       : $(date)"
echo "Model         : $MODEL_NAME"
echo "Thinking      : $THINKING"
echo "Vision budget : $VISION_BUDGET tokens"
echo "Output        : $OUTPUT_NAME"
echo ""

# ── GPU check ────────────────────────────────────────────────────────────────
echo "[gpu] nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Python environment ───────────────────────────────────────────────────────
module load devel/python/3.12 2>/dev/null || true

# Use separate vLLM venv (has correct CUDA libs from GPU node install)
VLLM_VENV="${HOME}/vllm_env"
if [ ! -f "$VLLM_VENV/bin/activate" ]; then
    echo "[env] Creating vLLM venv ..."
    python3 -m venv "$VLLM_VENV"
    source "$VLLM_VENV/bin/activate"
    pip install vllm openai --quiet
else
    source "$VLLM_VENV/bin/activate"
fi

# Also need project deps for baseline_eval.py
pip install langchain-openai langchain-core click python-dotenv --quiet 2>/dev/null

PYTHON="python"

echo "[env] Python: $($PYTHON --version)"
echo "[env] vLLM: $($PYTHON -c 'import vllm; print(vllm.__version__)')"
echo ""

# ── HuggingFace cache ────────────────────────────────────────────────────────
# Use workspace for model cache (HOME is too small)
HF_CACHE="${HF_HOME:-$(ws_find ollama_models 2>/dev/null || echo $HOME/.cache)/huggingface}"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
mkdir -p "$HF_CACHE"
echo "[hf] Cache dir: $HF_CACHE"

# ── Download chat template if needed ──────────────────────────────────────────
mkdir -p examples
if [ ! -f "examples/tool_chat_template_gemma4.jinja" ]; then
    echo "[vllm] Downloading Gemma 4 chat template..."
    curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja" \
        -o examples/tool_chat_template_gemma4.jinja || echo "[WARN] Failed to download chat template, thinking may not work"
fi

# ── Build vLLM server args ────────────────────────────────────────────────────
VLLM_ARGS=(
    --model "$MODEL_NAME"
    --mm-processor-kwargs "{\"max_soft_tokens\": $VISION_BUDGET}"
    --dtype bfloat16
    --max-model-len 65536
    --gpu-memory-utilization 0.9
    --port $VLLM_PORT
    --trust-remote-code
    --enforce-eager
)

if [ "$THINKING" = "true" ]; then
    echo "[vllm] Thinking mode ENABLED (via prompt)"
else
    echo "[vllm] Thinking mode DISABLED"
fi

# ── Start vLLM server ────────────────────────────────────────────────────────
# Clear stale Triton/inductor caches from previous jobs
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "[vllm] Starting vLLM server with $MODEL_NAME (vision budget: $VISION_BUDGET) ..."

$PYTHON -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
    > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "[vllm] Waiting for server to start (this may take a few minutes for model download) ..."
for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[vllm] Server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[ERROR] vLLM process died. Log:"
        tail -30 logs/vllm_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "[ERROR] vLLM did not start within 600s. Log:"
    tail -50 logs/vllm_${SLURM_JOB_ID}.log
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "[vllm] Models available:"
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" | python3 -c "import sys,json; [print(f'  {m[\"id\"]}') for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null || true
echo ""

# ── Run baseline evaluation ──────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export BASELINE_VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
export BASELINE_MODEL="$MODEL_NAME"
export BASELINE_THINKING="$THINKING"

echo "[run] Starting baseline eval ..."
echo "[run] Thinking: $THINKING"
$PYTHON baseline_eval.py --all \
    --model "$MODEL_NAME" \
    --results-dir "results GeoRC" \
    --output-name "$OUTPUT_NAME" \
    --clean-image-dir "clean_images_GeoRC" \
    --concurrency 1 \
    --verbose

EXIT_CODE=$?

# ── Cleanup ──────────────────────────────────────────────────────────────────
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=== Done: $(date) (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
