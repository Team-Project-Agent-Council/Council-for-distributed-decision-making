#!/bin/bash
#SBATCH --job-name=vlm-judge
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VLM Council : LLM-as-Judge on bwUniCluster 3 (uc3, H100 80GB).
#
# Loads a single VLM (default: google/gemma-4-31b-it), starts a local vLLM
# OpenAI-compatible server, then runs `python -m eval_debate judge` over the
# image IDs in --file-list (one image per line : name or stem).
#
# Resume-capable: images that already have <out>/judge/<image_id>.json are
# skipped automatically by judge.run().
#
# Usage (typically launched by slurm/launch_judge_short_uc3.sh):
#   sbatch slurm/run_judge_uc3.sh --file-list .vlm_judge_lists/job_0.txt

set -uo pipefail

PROJECT_DIR="$(ws_find vlm-council-debate)"
cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================="
echo "  VLM Council : LLM-as-Judge (uc3 H100)"
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

# HuggingFace cache → workspace (one-time download already done by council run)
HF_WS="$(ws_find hf-cache 2>/dev/null || echo "$PROJECT_DIR/.cache")"
export HF_HOME="$HF_WS"
export HF_TOKEN="${HF_TOKEN:-<TOKEN>}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"

export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

rm -rf "$HOME/.cache/vllm/torch_compile_cache" 2>/dev/null || true

# GPU check + cleanup
if nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

    echo "Cleaning GPU memory..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        if [ -n "$pid" ]; then
            echo "  Killing leftover GPU process: $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    echo "GPU after cleanup:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader
else
    echo "WARNING: nvidia-smi not available"
fi
echo ""

# vLLM Server
VLM_MODEL="${VLM_JUDGE_LLM_MODEL:-${VLM_MODEL:-google/gemma-4-31b-it}}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-16384}"
VLM_GPU_MEMORY_UTIL="${VLM_GPU_MEMORY_UTIL:-0.85}"

VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-$$} % 1000)))

echo "Starting vLLM server..."
echo "  Model: $VLM_MODEL"
echo "  Port: $VLLM_PORT"
echo "  Max model len: $VLM_MAX_MODEL_LEN"
echo "  GPU memory util: $VLM_GPU_MEMORY_UTIL"

VLM_MM_PROCESSOR_KWARGS="${VLM_MM_PROCESSOR_KWARGS:-{\"max_soft_tokens\": 1120}}"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --dtype auto \
    --gpu-memory-utilization "$VLM_GPU_MEMORY_UTIL" \
    --max-model-len "$VLM_MAX_MODEL_LEN" \
    --trust-remote-code \
    --enforce-eager \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    ${VLM_MM_PROCESSOR_KWARGS:+--mm-processor-kwargs "$VLM_MM_PROCESSOR_KWARGS"} \
    &> "logs/vllm-judge-${SLURM_JOB_ID:-local}.log" &
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
        tail -30 "logs/vllm-judge-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -s "$VLLM_URL/health" &>/dev/null; then
    echo "ERROR: vLLM failed to start after 600s" >&2
    tail -30 "logs/vllm-judge-${SLURM_JOB_ID:-local}.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "vLLM models:"
curl -s "$VLLM_URL/v1/models" | python3 -m json.tool 2>/dev/null || true
echo ""

# Judge inputs / outputs
VLM_RESULTS_DIR="${VLM_RESULTS_DIR:-results_debate_adversarial}"
VLM_GT_CSV="${VLM_GT_CSV:-Images/georc_locations.csv}"
VLM_EVAL_OUT="${VLM_EVAL_OUT:-eval_out/debate_adversarial}"
VLM_IMAGE_ROOT="${VLM_IMAGE_ROOT:-Images}"
VLM_JUDGE_CONCURRENCY="${VLM_JUDGE_CONCURRENCY:-8}"

mkdir -p "$VLM_EVAL_OUT/judge"

if [ ! -d "$VLM_RESULTS_DIR" ]; then
    echo "ERROR: Results dir not found: $VLM_RESULTS_DIR" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi
if [ ! -f "$VLM_GT_CSV" ]; then
    echo "ERROR: GT CSV not found: $VLM_GT_CSV" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

export VLM_JUDGE_LLM_MODEL="$VLM_MODEL"
export VLM_JUDGE_LLM_API_BASE="$VLLM_URL/v1"
export VLM_IMAGE_TOKEN_BUDGET="${VLM_IMAGE_TOKEN_BUDGET:-1120}"

echo "=== Starting LLM-as-Judge ==="
echo "  Results dir:   $VLM_RESULTS_DIR"
echo "  GT CSV:        $VLM_GT_CSV"
echo "  Eval out:      $VLM_EVAL_OUT"
echo "  Image root:    $VLM_IMAGE_ROOT"
echo "  Model:         $VLM_JUDGE_LLM_MODEL"
echo "  API base:      $VLM_JUDGE_LLM_API_BASE"
echo "  Concurrency:   $VLM_JUDGE_CONCURRENCY"
echo "  Extra args:    $EXTRA_ARGS"
echo ""

python -m eval_debate judge \
    --results "$VLM_RESULTS_DIR" \
    --gt "$VLM_GT_CSV" \
    --out "$VLM_EVAL_OUT" \
    --image-root "$VLM_IMAGE_ROOT" \
    --concurrency "$VLM_JUDGE_CONCURRENCY" \
    $EXTRA_ARGS
EXIT_CODE=$?

echo ""
echo "Judge finished (exit code: $EXIT_CODE)"
echo "Verdicts: $(find "$VLM_EVAL_OUT/judge" -name '*.json' 2>/dev/null | wc -l) files"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
exit $EXIT_CODE
