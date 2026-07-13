#!/bin/bash
#SBATCH --job-name=vlm-council-hs
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VLM Council : vLLM on bwUniCluster 3 (uc3, H100 80GB).
# Resume-capable: already processed images are skipped.
#
# Usage:
#   cd $(ws_find vlm-council-hs) && sbatch slurm/run_council_uc3.sh
#   cd $(ws_find vlm-council-hs) && sbatch slurm/run_council_uc3.sh --limit 5

set -uo pipefail

# Workspace paths 
PROJECT_DIR="$(ws_find vlm-council-hs)"
cd "$PROJECT_DIR"
mkdir -p logs results

echo "============================================="
echo "  VLM Council : uc3 vLLM (H100)"
echo "============================================="
echo "Working dir: $(pwd)"
echo "Job-ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

EXTRA_ARGS="${@}"

# Environment
module load devel/python/3.13.1 || { echo "ERROR: Cannot load Python module"; exit 1; }
module load devel/cuda/12.8 || { echo "ERROR: Cannot load CUDA module"; exit 1; }

# Re-enable strict error handling after modules
set -e

# Activate venv
source "$PROJECT_DIR/.venv/bin/activate"

# HuggingFace cache → workspace
HF_WS="$(ws_find hf-cache 2>/dev/null || echo "$PROJECT_DIR/.cache")"
export HF_HOME="$HF_WS/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"

export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Clear stale vLLM compile cache that references old job tmp dirs
rm -rf "$HOME/.cache/vllm/torch_compile_cache" 2>/dev/null || true

# GPU check + cleanup
if nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

    # Kill any leftover GPU processes from previous jobs on this node
    echo "Cleaning GPU memory..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        if [ -n "$pid" ]; then
            echo "  Killing leftover GPU process: $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Clear CUDA cache
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    echo "GPU after cleanup:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader
else
    echo "WARNING: nvidia-smi not available"
fi
echo ""

# vLLM Server 
VLM_MODEL="${VLM_MODEL:-google/gemma-4-31b-it}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-8192}"
VLM_GPU_MEMORY_UTIL="${VLM_GPU_MEMORY_UTIL:-0.85}"

# Unique port per job
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-$$} % 1000)))

echo "Starting vLLM server..."
echo "  Model: $VLM_MODEL"
echo "  Port: $VLLM_PORT"
echo "  Max model len: $VLM_MAX_MODEL_LEN"
echo "  GPU memory util: $VLM_GPU_MEMORY_UTIL"

VLM_MM_PROCESSOR_KWARGS="${VLM_MM_PROCESSOR_KWARGS:-{\"max_soft_tokens\": 1120}}"

python -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --dtype auto \
    --gpu-memory-utilization "$VLM_GPU_MEMORY_UTIL" \
    --max-model-len "$VLM_MAX_MODEL_LEN" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    ${VLM_MM_PROCESSOR_KWARGS:+--mm-processor-kwargs "$VLM_MM_PROCESSOR_KWARGS"} \
    &> "logs/vllm-${SLURM_JOB_ID:-local}.log" &
VLLM_PID=$!

# Wait for vLLM to be ready
VLLM_URL="http://localhost:$VLLM_PORT"
echo "Waiting for vLLM (PID: $VLLM_PID)..."
for i in $(seq 1 120); do
    if curl -s "$VLLM_URL/health" &>/dev/null; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Log:" >&2
        tail -30 "logs/vllm-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -s "$VLLM_URL/health" &>/dev/null; then
    echo "ERROR: vLLM failed to start after 600s" >&2
    tail -30 "logs/vllm-${SLURM_JOB_ID:-local}.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "vLLM models:"
curl -s "$VLLM_URL/v1/models" | python3 -m json.tool 2>/dev/null || true
echo ""

# Run VLM Council
export VLM_MODEL
export VLM_API_BASE="$VLLM_URL/v1"
export VLM_MAX_MODEL_LEN
export VLM_MAX_DISCUSSION_ROUNDS="${VLM_MAX_DISCUSSION_ROUNDS:-3}"
export VLM_JUDGE_MODEL="${VLM_JUDGE_MODEL:-$VLM_MODEL}"

OUTPUT_DIR="${VLM_OUTPUT_DIR:-results}"
mkdir -p "$OUTPUT_DIR"

echo "=== Starting VLM Council ==="
echo "  VLM_MODEL: $VLM_MODEL"
echo "  VLM_JUDGE_MODEL: $VLM_JUDGE_MODEL"
echo "  VLM_API_BASE: $VLM_API_BASE"
echo "  VLM_MAX_DISCUSSION_ROUNDS: $VLM_MAX_DISCUSSION_ROUNDS"
echo "  Output: $OUTPUT_DIR/"
echo ""

python -m vlm_council.batch Images/ "$OUTPUT_DIR/" $EXTRA_ARGS
EXIT_CODE=$?

echo ""
echo "Pipeline finished (exit code: $EXIT_CODE)"
echo "Results: $(find "$OUTPUT_DIR/" -name 'result.json' 2>/dev/null | wc -l) images processed"

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
exit $EXIT_CODE
