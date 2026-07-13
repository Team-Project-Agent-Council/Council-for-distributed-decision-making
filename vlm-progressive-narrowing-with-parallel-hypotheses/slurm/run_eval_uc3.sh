#!/bin/bash
#SBATCH --job-name=vlm-council-eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VLM Council: Full evaluation pipeline (Stage 1 + LLM-as-Judge Stage 2).
# Resume-capable: per-image judge/<image_id>.json files are skipped if present.
#
# Usage:
#   cd $(ws_find vlm-judge-eval) && sbatch slurm/run_eval_uc3.sh
#   cd $(ws_find vlm-judge-eval) && sbatch slurm/run_eval_uc3.sh --limit 3  (smoke test)
#
# Environment overrides (export before sbatch or pass inline):
#   VLM_MODEL          , judge model (default: Qwen/Qwen3.6-35B-A3B-FP8)
#   VLM_RESULTS_DIR    , results directory (default: results)
#   VLM_GT_CSV         , ground-truth CSV (default: Images/georc_locations.csv)
#   VLM_IMAGE_ROOT     , image directory (default: Images)
#   VLM_EVAL_OUT       , output directory (default: eval_outputs)
#   VLM_JUDGE_CONCURRENCY, parallel judge calls (default: 4)

set -uo pipefail

PROJECT_DIR="$(ws_find vlm-judge-eval)"
cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================="
echo "  VLM Council Eval: uc3 (H100)"
echo "============================================="
echo "Working dir: $(pwd)"
echo "Job-ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

EXTRA_ARGS="${@}"

# Environment
module load devel/python/3.13.1 || { echo "ERROR: Cannot load Python module"; exit 1; }
module load devel/cuda/12.8 || { echo "ERROR: Cannot load CUDA module"; exit 1; }
set -e

source "$PROJECT_DIR/.venv/bin/activate"

HF_WS="$(ws_find hf-cache 2>/dev/null || echo "$PROJECT_DIR/.cache")"
export HF_HOME="$HF_WS/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"

export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
rm -rf "$HOME/.cache/vllm/torch_compile_cache" 2>/dev/null || true

# Config
RESULTS_DIR="${VLM_RESULTS_DIR:-results}"
GT_CSV="${VLM_GT_CSV:-Images/georc_locations.csv}"
IMAGE_ROOT="${VLM_IMAGE_ROOT:-Images}"
EVAL_OUT="${VLM_EVAL_OUT:-eval_outputs}"
CONCURRENCY="${VLM_JUDGE_CONCURRENCY:-4}"
mkdir -p "$EVAL_OUT"

# Verify inputs
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: results dir $RESULTS_DIR does not exist" >&2
    exit 1
fi
if [[ ! -f "$GT_CSV" ]]; then
    echo "ERROR: ground-truth CSV $GT_CSV does not exist" >&2
    exit 1
fi

echo "Results:    $RESULTS_DIR"
echo "GT CSV:     $GT_CSV"
echo "Image root: $IMAGE_ROOT"
echo "Eval out:   $EVAL_OUT"
echo ""

# GPU info
if nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        if [ -n "$pid" ]; then
            echo "  Killing leftover GPU process: $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi
echo ""

# vLLM Judge Server
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-8192}"
VLM_GPU_MEMORY_UTIL="${VLM_GPU_MEMORY_UTIL:-0.85}"
VLLM_PORT=$((8100 + (${SLURM_JOB_ID:-$$} % 1000)))

echo "Starting vLLM judge server..."
echo "  Model: $VLM_MODEL"
echo "  Port:  $VLLM_PORT"

python -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --dtype auto \
    --gpu-memory-utilization "$VLM_GPU_MEMORY_UTIL" \
    --max-model-len "$VLM_MAX_MODEL_LEN" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    &> "logs/vllm-eval-${SLURM_JOB_ID:-local}.log" &
VLLM_PID=$!

VLLM_URL="http://localhost:$VLLM_PORT"
echo "Waiting for vLLM (PID: $VLLM_PID)..."
for i in $(seq 1 120); do
    if curl -s "$VLLM_URL/health" &>/dev/null; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Log:" >&2
        tail -30 "logs/vllm-eval-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -s "$VLLM_URL/health" &>/dev/null; then
    echo "ERROR: vLLM failed to start after 600s" >&2
    tail -30 "logs/vllm-eval-${SLURM_JOB_ID:-local}.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi
echo ""

export VLM_JUDGE_LLM_MODEL="$VLM_MODEL"
export VLM_JUDGE_LLM_API_BASE="$VLLM_URL/v1"

# Chunked mode: a parallel launcher set VLM_JUDGE_FILE_LIST → only run the
# judge stage restricted to those image_ids. Stage 1 / aggregate / report
# stay the responsibility of a separate full-run job (or a final cleanup).
if [[ -n "${VLM_JUDGE_FILE_LIST:-}" ]]; then
    echo "=== Chunked mode: judge only ==="
    echo "  File list: $VLM_JUDGE_FILE_LIST"
    echo "  Lines:     $(wc -l < "$VLM_JUDGE_FILE_LIST")"

    LIMIT_ARG=""
    if [[ "$EXTRA_ARGS" == *"--limit"* ]]; then
        LIMIT_ARG="$EXTRA_ARGS"
    fi

    python -m eval judge \
        --results "$RESULTS_DIR" \
        --gt "$GT_CSV" \
        --out "$EVAL_OUT" \
        --image-root "$IMAGE_ROOT" \
        --concurrency "$CONCURRENCY" \
        --file-list "$VLM_JUDGE_FILE_LIST" \
        $LIMIT_ARG

    echo ""
    echo "Done (chunked). Verdicts in: $EVAL_OUT/judge/"

    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    exit 0
fi

# Stage 1: deterministic metrics (no GPU needed but run here for convenience)
echo "=== Stage 1: geo ==="
python -m eval geo --results "$RESULTS_DIR" --gt "$GT_CSV" --out "$EVAL_OUT"

echo "=== Stage 1: agents ==="
python -m eval agents --results "$RESULTS_DIR" --gt "$GT_CSV" --out "$EVAL_OUT"

echo "=== Stage 1: funnel ==="
python -m eval funnel --results "$RESULTS_DIR" --gt "$GT_CSV" --out "$EVAL_OUT"

# Stage 2: LLM-as-judge
echo "=== Stage 2: judge ==="
LIMIT_ARG=""
if [[ "$EXTRA_ARGS" == *"--limit"* ]]; then
    LIMIT_ARG="$EXTRA_ARGS"
fi

python -m eval judge \
    --results "$RESULTS_DIR" \
    --gt "$GT_CSV" \
    --out "$EVAL_OUT" \
    --image-root "$IMAGE_ROOT" \
    --concurrency "$CONCURRENCY" \
    $LIMIT_ARG

echo "=== Stage 2: aggregate ==="
python -m eval aggregate --out "$EVAL_OUT"

echo "=== Report ==="
python -m eval report --out "$EVAL_OUT"

echo ""
echo "Done. Report: $EVAL_OUT/report.md"

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
exit 0
