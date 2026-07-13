#!/bin/bash
#SBATCH --job-name=vlm-eval-tournament-qwen
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VLM Council Tournament-only : LLM-as-a-judge eval, Qwen3.6-35B-A3B-FP8.
# Runs eval/judge_tournament.py + eval/judge_aggregate_tournament.py
# against the 500-image tournament-only results.
#
# Usage (from workspace):
#   sbatch slurm/run_eval_uc3_qwen.sh
#   sbatch slurm/run_eval_uc3_qwen.sh --limit 3    # smoke
#
# Resume-capable: judge verdicts are skipped when they already exist.
# Chained-resubmit: if verdicts remain after this slot, another slot is
# submitted automatically (unless --limit was passed).

set -uo pipefail

PROJECT_DIR="$(ws_find vlm-council-tournament)"
cd "$PROJECT_DIR"
mkdir -p logs

EXTRA_ARGS="${@}"

echo "============================================="
echo "  VLM Council Tournament : Judge (Qwen3.6 MoE)"
echo "============================================="
echo "Working dir: $(pwd)"
echo "Job-ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date -Iseconds)"
echo ""

module load devel/python/3.12 || { echo "ERROR: cannot load Python module"; exit 1; }
module load devel/cuda/12.8 || { echo "ERROR: cannot load CUDA module"; exit 1; }

set -e

source "$HOME/vllm_env/bin/activate"

# Cache HuggingFace weights in the workspace (persistent across slots) so
# the 35B Qwen download only happens once.
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Allow online mode by default so first slot can download; user can force
# offline once weights are cached by exporting HF_HUB_OFFLINE=1 externally.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

rm -rf "$HOME/.cache/vllm/torch_compile_cache" 2>/dev/null || true

if nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi
echo ""

# Inputs / outputs
VLM_RESULTS_DIR="${VLM_RESULTS_DIR:-$PROJECT_DIR/results_tournament_500}"
VLM_GT_CSV="${VLM_GT_CSV:-/pfs/data6/home/ma/ma_ma/ma_lweisssc/llm_council_agent_orchestration_tests/clean_images_GeoRC_500/georc_locations.csv}"
VLM_IMAGE_ROOT="${VLM_IMAGE_ROOT:-/pfs/data6/home/ma/ma_ma/ma_lweisssc/llm_council_agent_orchestration_tests/clean_images_GeoRC_500}"
VLM_EVAL_OUT="${VLM_EVAL_OUT:-$PROJECT_DIR/eval_outputs_tournament_qwen}"
JUDGE_CONCURRENCY="${VLM_JUDGE_CONCURRENCY:-4}"

if [[ ! -d "$VLM_RESULTS_DIR" ]]; then
    echo "ERROR: results dir $VLM_RESULTS_DIR does not exist"; exit 1
fi
if [[ ! -f "$VLM_GT_CSV" ]]; then
    echo "ERROR: ground-truth CSV $VLM_GT_CSV does not exist"; exit 1
fi
if [[ ! -d "$VLM_IMAGE_ROOT" ]]; then
    echo "ERROR: image dir $VLM_IMAGE_ROOT does not exist"; exit 1
fi

mkdir -p "$VLM_EVAL_OUT"

echo "Eval config:"
echo "  results dir:  $VLM_RESULTS_DIR"
echo "  ground truth: $VLM_GT_CSV  ($(wc -l < "$VLM_GT_CSV") lines)"
echo "  image root:   $VLM_IMAGE_ROOT"
echo "  output dir:   $VLM_EVAL_OUT"
echo "  concurrency:  $JUDGE_CONCURRENCY"
echo ""

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# vLLM server for the judge
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-40960}"
export VLM_JUDGE_MAX_TOKENS="${VLM_JUDGE_MAX_TOKENS:-8000}"
VLM_GPU_MEMORY_UTIL="${VLM_GPU_MEMORY_UTIL:-0.88}"
VLLM_PORT=$((8100 + (${SLURM_JOB_ID:-$$} % 900)))

echo "=== Booting vLLM (judge) ==="
echo "  Model: $VLM_MODEL"
echo "  Port:  $VLLM_PORT"
echo "  HF cache: $HF_HOME"

# For Qwen MoE vision models the mm-processor-kwargs are model-specific;
# we do not override unless the user asks.
python -u -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --dtype auto \
    --gpu-memory-utilization "$VLM_GPU_MEMORY_UTIL" \
    --max-model-len "$VLM_MAX_MODEL_LEN" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    ${VLM_MM_PROCESSOR_KWARGS:+--mm-processor-kwargs "$VLM_MM_PROCESSOR_KWARGS"} \
    &> "logs/vllm-eval-qwen-${SLURM_JOB_ID:-local}.log" &
VLLM_PID=$!

VLLM_URL="http://localhost:$VLLM_PORT"
echo "Waiting for vLLM (PID: $VLLM_PID)..."
# Longer patience: first-time download of a 35B FP8 model can take a while.
for i in $(seq 1 360); do
    if curl -s "$VLLM_URL/health" &>/dev/null; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died. Log tail:" >&2
        tail -60 "logs/vllm-eval-qwen-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -s "$VLLM_URL/health" &>/dev/null; then
    echo "ERROR: vLLM failed to start after 1800s" >&2
    tail -60 "logs/vllm-eval-qwen-${SLURM_JOB_ID:-local}.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Judge env
export VLM_JUDGE_LLM_MODEL="$VLM_MODEL"
export VLM_JUDGE_LLM_API_BASE="$VLLM_URL/v1"

echo ""
echo "=== Stage 2: judge_tournament ==="
python -u -m eval.judge_tournament \
    --results "$VLM_RESULTS_DIR" \
    --gt "$VLM_GT_CSV" \
    --image-root "$VLM_IMAGE_ROOT" \
    --out "$VLM_EVAL_OUT" \
    --concurrency "$JUDGE_CONCURRENCY" \
    $EXTRA_ARGS
JUDGE_EXIT=$?

echo ""
echo "=== Stage 2: aggregate ==="
python -u -m eval.judge_aggregate_tournament --out "$VLM_EVAL_OUT" --results "$VLM_RESULTS_DIR" || \
    echo "[warn] aggregation failed (safe to ignore on the first slot)"

echo ""
echo "Finished at $(date -Iseconds) (judge exit code: $JUDGE_EXIT)"
echo "Outputs in: $VLM_EVAL_OUT"
ls -la "$VLM_EVAL_OUT" | head -20

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# Chained-resubmit
N_RECORDS=$(ls -1d "$VLM_RESULTS_DIR"/*/ 2>/dev/null | wc -l)
N_VERDICTS=$(ls -1 "$VLM_EVAL_OUT/judge/"*.json 2>/dev/null | wc -l)
echo "Progress: $N_VERDICTS / $N_RECORDS verdicts written"
if [[ ! "$EXTRA_ARGS" =~ --limit ]] && [[ "$N_VERDICTS" -lt "$N_RECORDS" ]]; then
    echo "Chaining next slot..."
    sbatch "$PROJECT_DIR/slurm/run_eval_uc3_qwen.sh"
fi

exit $JUDGE_EXIT
