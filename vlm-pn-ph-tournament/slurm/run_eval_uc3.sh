#!/bin/bash
#SBATCH --job-name=vlm-eval-v12
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VLM Council v12 : full evaluation suite (Stage 1 + Stage 2 + report)
# : vLLM on bwUniCluster 3 (uc3, H100 80GB), gpu_h100_short partition
#
# Stage 1 (CPU): geo / agents / influence : runs in seconds against
#   results_v12_pn/*/result.json + georc_locations.csv
# Stage 2 (GPU): LLM-as-judge : boots vLLM (same model as the council)
#   and writes a per-image verdict under <out>/judge/<image_id>.json.
#   The judge sees ground truth + image + the full agent discussion
#   trace and produces a structured Pydantic verdict.
# Report: composes everything into a single markdown file.
#
# Usage:
#   cd $(ws_find vlm-council-pnt)
#
#   # Smoke (3 images, judge only)
#   sbatch slurm/run_eval_uc3.sh --limit 3
#
#   # Full eval against results_v12_pn/
#   sbatch slurm/run_eval_uc3.sh
#
# Resume-capable: judge/<image_id>.json files are skipped, so re-submit if
# the slot expires.
#
# Override defaults from the command line:
#   VLM_RESULTS_DIR=results_v12_pn_qwen \
#   VLM_EVAL_OUT=eval_outputs_qwen \
#       sbatch slurm/run_eval_uc3.sh

set -uo pipefail

PROJECT_DIR="$(ws_find vlm-council-pnt)"
cd "$PROJECT_DIR"
mkdir -p logs

EXTRA_ARGS="${@}"

echo "============================================="
echo "  VLM Council v12 : Evaluation Suite (uc3)"
echo "============================================="
echo "Working dir: $(pwd)"
echo "Job-ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date -Iseconds)"
echo ""

# Environment
module load devel/python/3.12 || { echo "ERROR: cannot load Python module"; exit 1; }
module load devel/cuda/12.8 || { echo "ERROR: cannot load CUDA module"; exit 1; }

set -e

source "$HOME/vllm_env/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

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
VLM_RESULTS_DIR="${VLM_RESULTS_DIR:-$PROJECT_DIR/results_v12_pn_500}"
VLM_GT_CSV="${VLM_GT_CSV:-$PROJECT_DIR/georc_locations.csv}"
VLM_IMAGE_ROOT="${VLM_IMAGE_ROOT:-$PROJECT_DIR/Images}"
VLM_EVAL_OUT="${VLM_EVAL_OUT:-$PROJECT_DIR/eval_outputs}"
JUDGE_CONCURRENCY="${VLM_JUDGE_CONCURRENCY:-1}"

# Auto-link georc_locations.csv from the standard llm_council_agent_orchestration_tests/
# checkout if it isn't already in the workspace.
if [[ ! -e "$VLM_GT_CSV" ]]; then
    SRC_GT="$HOME/llm_council_agent_orchestration_tests/georc_locations.csv"
    if [[ -f "$SRC_GT" ]]; then
        ln -sfn "$SRC_GT" "$VLM_GT_CSV"
        echo "[setup] linked $VLM_GT_CSV -> $SRC_GT"
    fi
fi

if [[ ! -d "$VLM_RESULTS_DIR" ]]; then
    echo "ERROR: results dir $VLM_RESULTS_DIR does not exist"
    exit 1
fi
if [[ ! -f "$VLM_GT_CSV" ]]; then
    echo "ERROR: ground-truth CSV $VLM_GT_CSV does not exist"
    exit 1
fi

mkdir -p "$VLM_EVAL_OUT"

echo "Eval config:"
echo "  results dir:  $VLM_RESULTS_DIR"
echo "  ground truth: $VLM_GT_CSV"
echo "  image root:   $VLM_IMAGE_ROOT"
echo "  output dir:   $VLM_EVAL_OUT"
echo "  concurrency:  $JUDGE_CONCURRENCY"
echo ""

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Stage 1 : deterministic, no GPU needed
echo "=== Stage 1: geo ==="
python -u -m eval geo \
    --results "$VLM_RESULTS_DIR" \
    --gt "$VLM_GT_CSV" \
    --out "$VLM_EVAL_OUT"

echo "=== Stage 1: agents ==="
python -u -m eval agents \
    --results "$VLM_RESULTS_DIR" \
    --gt "$VLM_GT_CSV" \
    --out "$VLM_EVAL_OUT"

echo "=== Stage 1: influence ==="
python -u -m eval influence \
    --results "$VLM_RESULTS_DIR" \
    --gt "$VLM_GT_CSV" \
    --out "$VLM_EVAL_OUT"

# Stage 2 : boot vLLM, run judge
VLM_MODEL="${VLM_MODEL:-google/gemma-4-31b-it}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-65536}"
VLM_GPU_MEMORY_UTIL="${VLM_GPU_MEMORY_UTIL:-0.85}"
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-$$} % 1000)))

echo "=== Booting vLLM for judge ==="
echo "  Model: $VLM_MODEL"
echo "  Port:  $VLLM_PORT"

VLM_MM_PROCESSOR_KWARGS="${VLM_MM_PROCESSOR_KWARGS:-{\"max_soft_tokens\": 1120}}"

python -u -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --dtype auto \
    --gpu-memory-utilization "$VLM_GPU_MEMORY_UTIL" \
    --max-model-len "$VLM_MAX_MODEL_LEN" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    ${VLM_MM_PROCESSOR_KWARGS:+--mm-processor-kwargs "$VLM_MM_PROCESSOR_KWARGS"} \
    &> "logs/vllm-eval-${SLURM_JOB_ID:-local}.log" &
VLLM_PID=$!

VLLM_URL="http://localhost:$VLLM_PORT"
echo "Waiting for vLLM (PID: $VLLM_PID)..."
for i in $(seq 1 250); do
    if curl -s "$VLLM_URL/health" &>/dev/null; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died. Log:" >&2
        tail -30 "logs/vllm-eval-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -s "$VLLM_URL/health" &>/dev/null; then
    echo "ERROR: vLLM failed to start after 1250s" >&2
    tail -30 "logs/vllm-eval-${SLURM_JOB_ID:-local}.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Judge LLM env (loader.get_vlm reads VLM_JUDGE_LLM_*)
export VLM_JUDGE_LLM_MODEL="$VLM_MODEL"
export VLM_JUDGE_LLM_API_BASE="$VLLM_URL/v1"

echo "=== Stage 2: judge ==="
python -u -m eval judge \
    --results "$VLM_RESULTS_DIR" \
    --gt "$VLM_GT_CSV" \
    --image-root "$VLM_IMAGE_ROOT" \
    --out "$VLM_EVAL_OUT" \
    --concurrency "$JUDGE_CONCURRENCY" \
    $EXTRA_ARGS
JUDGE_EXIT=$?

echo "=== Stage 2: aggregate ==="
python -u -m eval aggregate --out "$VLM_EVAL_OUT"

echo "=== Report ==="
python -u -m eval report --out "$VLM_EVAL_OUT"

echo ""
echo "Eval finished at $(date -Iseconds) (judge exit code: $JUDGE_EXIT)"
echo "Outputs in: $VLM_EVAL_OUT"
ls -la "$VLM_EVAL_OUT"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# Chained-resubmit: if there are still un-judged images and we are NOT in
# --limit smoke-test mode, queue the next 30-min slot. The judge skips
# images whose verdict file already exists, so it picks up where this slot
# left off.
N_RECORDS=$(ls -1d "$VLM_RESULTS_DIR"/*/ 2>/dev/null | wc -l)
N_VERDICTS=$(ls -1 "$VLM_EVAL_OUT/judge/"*.json 2>/dev/null | wc -l)
echo "Progress: $N_VERDICTS / $N_RECORDS verdicts written"
if [[ ! "$EXTRA_ARGS" =~ --limit ]] && [[ "$N_VERDICTS" -lt "$N_RECORDS" ]]; then
    echo "Chaining next slot (run_eval_uc3.sh)..."
    sbatch "$PROJECT_DIR/slurm/run_eval_uc3.sh"
fi

exit $JUDGE_EXIT
