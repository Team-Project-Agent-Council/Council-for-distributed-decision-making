#!/bin/bash
#SBATCH --job-name=vlm-eval-v12-qwen
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VLM Council PN+PH+Tournament (v12_pn) : LLM-as-a-judge eval with
# Qwen3.6-35B-A3B-FP8. Boots vLLM (H100-only, FP8) and runs
# `python -m eval judge` against the 500-image v12_pn results.
#
# Usage (from cluster workspace):
#   sbatch slurm/run_eval_uc3_qwen.sh
#   sbatch slurm/run_eval_uc3_qwen.sh --limit 3    # smoke
#
# Resume-capable: existing verdict JSONs are skipped.
# Chained-resubmit: another slot is submitted if verdicts remain.

set -uo pipefail

PROJECT_DIR="$(ws_find vlm-council-pnt)"
cd "$PROJECT_DIR"
mkdir -p logs

EXTRA_ARGS="${@}"

echo "============================================="
echo "  VLM Council v12_pn : Judge (Qwen3.6 MoE)"
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

export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

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
VLM_RESULTS_DIR="${VLM_RESULTS_DIR:-$PROJECT_DIR/results_v12_pn_500}"
VLM_GT_CSV="${VLM_GT_CSV:-$HOME/llm_council_agent_orchestration_tests/clean_images_GeoRC_500/georc_locations.csv}"
VLM_IMAGE_ROOT="${VLM_IMAGE_ROOT:-$HOME/llm_council_agent_orchestration_tests/clean_images_GeoRC_500}"
VLM_EVAL_OUT="${VLM_EVAL_OUT:-$PROJECT_DIR/eval_outputs_v12_pn_500_qwen}"
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

# vLLM server for the judge (Qwen MoE FP8 needs H100)
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-49152}"
export VLM_JUDGE_MAX_TOKENS="${VLM_JUDGE_MAX_TOKENS:-32000}"
VLM_GPU_MEMORY_UTIL="${VLM_GPU_MEMORY_UTIL:-0.88}"
VLLM_PORT=$((8100 + (${SLURM_JOB_ID:-$$} % 900)))

echo "=== Booting vLLM (judge) ==="
echo "  Model: $VLM_MODEL"
echo "  Port:  $VLLM_PORT"
echo "  HF cache: $HF_HOME"

python -u -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --dtype auto \
    --gpu-memory-utilization "$VLM_GPU_MEMORY_UTIL" \
    --max-model-len "$VLM_MAX_MODEL_LEN" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    ${VLM_MM_PROCESSOR_KWARGS:+--mm-processor-kwargs "$VLM_MM_PROCESSOR_KWARGS"} \
    &> "logs/vllm-eval-v12-qwen-${SLURM_JOB_ID:-local}.log" &
VLLM_PID=$!

VLLM_URL="http://localhost:$VLLM_PORT"
echo "Waiting for vLLM (PID: $VLLM_PID)..."
for i in $(seq 1 360); do
    if curl -s "$VLLM_URL/health" &>/dev/null; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died. Log tail:" >&2
        tail -60 "logs/vllm-eval-v12-qwen-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -s "$VLLM_URL/health" &>/dev/null; then
    echo "ERROR: vLLM failed to start after 1800s" >&2
    tail -60 "logs/vllm-eval-v12-qwen-${SLURM_JOB_ID:-local}.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Judge env
export VLM_MODEL="$VLM_MODEL"
export VLM_API_BASE="$VLLM_URL/v1"

echo ""
echo "=== Stage 2: judge ==="
python -u -m eval judge \
    --results "$VLM_RESULTS_DIR" \
    --gt "$VLM_GT_CSV" \
    --image-root "$VLM_IMAGE_ROOT" \
    --out "$VLM_EVAL_OUT" \
    --model "$VLM_MODEL" \
    --api-base "$VLLM_URL/v1" \
    --concurrency "$JUDGE_CONCURRENCY" \
    $EXTRA_ARGS
JUDGE_EXIT=$?

echo ""
echo "=== Stage 2: aggregate ==="
python -u -m eval aggregate --out "$VLM_EVAL_OUT" || \
    echo "[warn] aggregation failed (safe to ignore on the first slot)"

echo ""
echo "=== Report ==="
python -u -m eval report --out "$VLM_EVAL_OUT" || true

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
