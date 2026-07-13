#!/bin/bash
#SBATCH --job-name=vlm-council-v12-pn
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VLM Council v12: Progressive Narrowing + RAG Pre-Filter + Tournament
# : vLLM on bwUniCluster 3
# (uc3, H100 80GB), gpu_h100_short partition, 30 min slot.
#
# Sequential: one job processes images one after another via vlm_council.batch.
# Resume-capable: already processed images are skipped on each re-submission,
# so just re-run sbatch until everything is done.
#
# Output layout:
#   results_v12_pn/<image_name>/result.json   ← per-image
#   council_v12_pn_result.json                ← aggregated (when batch finishes)
#
# Usage:
#   cd $(ws_find vlm-council-pnt)
#
#   # Smoke test (5 images)
#   sbatch slurm/run_council_uc3.sh --limit 5
#
#   # Full batch : sequential, re-submit when the 30 min slot expires
#   sbatch slurm/run_council_uc3.sh
#   # …after it finishes / times out, just sbatch again. Done images are skipped.
#
# Override defaults from the command line:
#   VLM_OUTPUT_DIR=results_v12_pn_qwen sbatch slurm/run_council_uc3.sh

set -uo pipefail

# Workspace paths
PROJECT_DIR="$(ws_find vlm-council-pnt)"
cd "$PROJECT_DIR"
mkdir -p logs results

echo "============================================="
echo "  VLM Council v12 (PN + Tournament + RAG) : uc3 vLLM (H100)"
echo "============================================="
echo "Working dir: $(pwd)"
echo "Job-ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date -Iseconds)"
echo ""

EXTRA_ARGS="${@}"

# Environment
module load devel/python/3.12 || { echo "ERROR: Cannot load Python module"; exit 1; }
module load devel/cuda/12.8 || { echo "ERROR: Cannot load CUDA module"; exit 1; }

# Re-enable strict error handling after modules
set -e

# Activate venv (use the working ~/vllm_env from llm_council_agent_orchestration_tests
# : Python 3.12 + vllm 0.19.1 + torch 2.10.0+cu128. The workspace-local .venv
# is broken: built against Python 3.13 / CUDA 13 which is not on this node.)
source "$HOME/vllm_env/bin/activate"

# HuggingFace cache → workspace
# HuggingFace cache → use ~/.cache/huggingface where gemma-4-31b-it is already
# downloaded (59 GB). The hf-cache workspace would re-trigger a 7+ min download.
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Offline mode: weights are already cached, never call the HF API. Without
# this, vLLM's snapshot_download issues a list_repo_files() probe and gets
# rate-limited (429) on shared cluster IPs.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Clear stale vLLM compile cache that references old job tmp dirs
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

# RAG data tree
# Default: vendored geoguessr_rag/ subfolder of the project. Override with
# VLM_DATA_DIR if the cluster has a different mount point.
export VLM_DATA_DIR="${VLM_DATA_DIR:-$PROJECT_DIR/geoguessr_rag}"
export VLM_RAG_MAX_REFS_PER_ROUND="${VLM_RAG_MAX_REFS_PER_ROUND:-6}"
export VLM_RAG_MAX_REFS_PER_COUNTRY="${VLM_RAG_MAX_REFS_PER_COUNTRY:-3}"

if [[ ! -d "$VLM_DATA_DIR" ]]; then
    echo "ERROR: VLM_DATA_DIR=$VLM_DATA_DIR does not exist."
    echo "       v12 requires the RAG data tree (driving_side, road_lines, plonkit_images)."
    exit 1
elif [[ ! -f "$VLM_DATA_DIR/driving_side.json" ]]; then
    echo "ERROR: $VLM_DATA_DIR/driving_side.json missing : pre-filters cannot run."
    exit 1
fi

echo "RAG config:"
echo "  VLM_DATA_DIR: $VLM_DATA_DIR"
echo "  VLM_RAG_MAX_REFS_PER_ROUND: $VLM_RAG_MAX_REFS_PER_ROUND"
echo "  VLM_RAG_MAX_REFS_PER_COUNTRY: $VLM_RAG_MAX_REFS_PER_COUNTRY"
echo ""

# vLLM Server : gemma-4-31b-it (same as v10)
VLM_MODEL="${VLM_MODEL:-google/gemma-4-31b-it}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-65536}"
VLM_GPU_MEMORY_UTIL="${VLM_GPU_MEMORY_UTIL:-0.85}"

# Unique port per job
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-$$} % 1000)))

echo "Starting vLLM server..."
echo "  Model: $VLM_MODEL"
echo "  Port: $VLLM_PORT"
echo "  Max model len: $VLM_MAX_MODEL_LEN"
echo "  GPU memory util: $VLM_GPU_MEMORY_UTIL"

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
    &> "logs/vllm-${SLURM_JOB_ID:-local}.log" &
VLLM_PID=$!

# Wait for vLLM to be ready (250 × 5s = 1250s, accommodates first-run download)
VLLM_URL="http://localhost:$VLLM_PORT"
echo "Waiting for vLLM (PID: $VLLM_PID)..."
for i in $(seq 1 250); do
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
    echo "ERROR: vLLM failed to start after 1250s" >&2
    tail -30 "logs/vllm-${SLURM_JOB_ID:-local}.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "vLLM models:"
curl -s "$VLLM_URL/v1/models" | python3 -m json.tool 2>/dev/null || true
echo ""

# Run VLM Council v12
export VLM_MODEL
export VLM_API_BASE="$VLLM_URL/v1"
export VLM_MAX_MODEL_LEN
export VLM_JUDGE_MODEL="${VLM_JUDGE_MODEL:-$VLM_MODEL}"
export VLM_JUDGE_THINKING="${VLM_JUDGE_THINKING:-false}"
# v12 doesn't run discussion rounds : Tournament is the disambiguation step.
# VLM_TOURNAMENT_FINALISTS controls bracket size (default 4 → 4-2-1)
export VLM_TOURNAMENT_FINALISTS="${VLM_TOURNAMENT_FINALISTS:-4}"
export VLM_PREFILTER_TOPK="${VLM_PREFILTER_TOPK:-8}"
# Force unbuffered output so per-image agent logs stream live to the .out file
export PYTHONUNBUFFERED=1

# LangSmith tracing : pull keys from ~/llm_council_agent_orchestration_tests/.env
ENV_FILE="$HOME/llm_council_agent_orchestration_tests/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
if [ -n "${LANGCHAIN_API_KEY:-}" ]; then
    export LANGCHAIN_TRACING_V2=true
    # Dedicated v12 project bucket
    export LANGCHAIN_PROJECT="vlm-council-v12-uc3"
    export LANGCHAIN_ENDPOINT="${LANGCHAIN_ENDPOINT:-https://api.smith.langchain.com}"
    echo "[langsmith] Tracing enabled → project: $LANGCHAIN_PROJECT"
else
    echo "[langsmith] LANGCHAIN_API_KEY not set : tracing disabled"
fi

OUTPUT_DIR="${VLM_OUTPUT_DIR:-results_v12_pn_500}"
RESULT_NAME="${VLM_RESULT_NAME:-council_v12_pn_result.json}"
export OUTPUT_DIR RESULT_NAME
mkdir -p "$OUTPUT_DIR"

echo "=== Starting VLM Council v12 (PN + Tournament + RAG) ==="
echo "  VLM_MODEL: $VLM_MODEL"
echo "  VLM_JUDGE_MODEL: $VLM_JUDGE_MODEL"
echo "  VLM_API_BASE: $VLM_API_BASE"
echo "  VLM_TOURNAMENT_FINALISTS: $VLM_TOURNAMENT_FINALISTS"
echo "  VLM_PREFILTER_TOPK: $VLM_PREFILTER_TOPK"
echo "  VLM_JUDGE_THINKING: $VLM_JUDGE_THINKING"
echo "  VLM_DATA_DIR: $VLM_DATA_DIR"
echo "  LANGCHAIN_TRACING_V2: ${LANGCHAIN_TRACING_V2:-(off)}"
echo "  LANGCHAIN_PROJECT: ${LANGCHAIN_PROJECT:-(none)}"
echo "  Output dir: $OUTPUT_DIR/"
echo "  Aggregated result: $RESULT_NAME"
echo ""

python -u -m vlm_council.batch Images/ "$OUTPUT_DIR/" $EXTRA_ARGS
EXIT_CODE=$?

echo ""
echo "Pipeline finished (exit code: $EXIT_CODE) at $(date -Iseconds)"
DONE_COUNT=$(find "$OUTPUT_DIR/" -name 'result.json' 2>/dev/null | wc -l)
echo "Results: $DONE_COUNT images processed in $OUTPUT_DIR/"

# Aggregate per-image results into a single JSON, mirroring v10_result naming
if [ "$DONE_COUNT" -gt 0 ]; then
    AGG_PATH="$OUTPUT_DIR/$RESULT_NAME"
    echo "Aggregating into $AGG_PATH ..."
    python3 -c "
import json, os, sys
out_dir = os.environ['OUTPUT_DIR']
agg = {}
for entry in sorted(os.listdir(out_dir)):
    rp = os.path.join(out_dir, entry, 'result.json')
    if os.path.isfile(rp):
        try:
            with open(rp) as f:
                agg[entry] = json.load(f)
        except json.JSONDecodeError as e:
            print(f'  skip {entry}: {e}', file=sys.stderr)
with open(os.path.join(out_dir, '$RESULT_NAME'), 'w') as f:
    json.dump(agg, f, indent=2, ensure_ascii=False)
print(f'  aggregated {len(agg)} entries')
" || echo "  (aggregation failed, per-image result.json files are intact)"
fi

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
exit $EXIT_CODE
