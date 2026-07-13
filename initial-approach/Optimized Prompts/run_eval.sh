#!/bin/bash
#SBATCH --job-name=council-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com   # uncomment and fill in

# =============================================================================
# Council evaluation job - BwUniCluster 3.0
#
# This job runs the council pipeline on all pre-computed result.json files.
# It does NOT need a GPU - all LLM inference happens via HTTP to an Ollama
# server whose address you set in OLLAMA_HOST below.
#
# Submit:  sbatch run_eval.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/eval_<jobid>.out
# =============================================================================

set -euo pipefail

# -- Project root --------------------------------------------------------------
# Adjust this to the absolute path on the cluster
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "$PROJECT_DIR"
# Shared vision_pipeline package lives one level up (monorepo root)
export PYTHONPATH="$(cd "$PROJECT_DIR/.." && pwd):${PYTHONPATH:-}"

mkdir -p logs

echo "=== Council Evaluation ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "Work dir : $PROJECT_DIR"
echo ""

# -- Python environment --------------------------------------------------------
# Option A: uv venv (recommended - fastest install)
if command -v uv &>/dev/null; then
    echo "[env] Using uv"
    uv sync --quiet
    PYTHON="uv run python"
# Option B: existing .venv
elif [ -f ".venv/bin/python" ]; then
    echo "[env] Using .venv"
    source .venv/bin/activate
    PYTHON="python"
# Option C: module system + venv
else
    echo "[env] Loading module + creating venv"
    module load devel/python/3.12
    if [ ! -f ".venv_cluster/bin/activate" ]; then
        python -m venv .venv_cluster
        source .venv_cluster/bin/activate
        pip install --quiet -e ".[dev]" 2>/dev/null || pip install --quiet -e .
    else
        source .venv_cluster/bin/activate
    fi
    PYTHON="python"
fi

echo "[env] Python: $($PYTHON --version)"
echo ""

# -- Ollama / LLM connectivity -------------------------------------------------
# Set OLLAMA_HOST to wherever your Ollama server is running.
# Options:
#   A) Ollama on a different cluster node/login node accessible via hostname
#   B) Ollama on your local machine exposed via SSH tunnel (see below)
#   C) Ollama running on the same node (start it before this job)
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# Check connectivity before wasting job time
echo "[check] Testing Ollama at $OLLAMA_HOST ..."
if curl -sf "$OLLAMA_HOST/api/tags" > /dev/null; then
    echo "[check] Ollama reachable [ok]"
else
    echo "[ERROR] Cannot reach Ollama at $OLLAMA_HOST - aborting."
    echo "        If using an SSH tunnel, set up the tunnel first (see README)."
    exit 1
fi
echo ""

# -- LangSmith (optional) ------------------------------------------------------
# Set these in your environment or a .env file if you want cloud tracing.
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY=ls-...
# export LANGCHAIN_PROJECT=council-eval

# -- Run evaluation ------------------------------------------------------------
echo "[run] Starting evaluation pipeline ..."
$PYTHON evaluate.py \
    --results-dir ../results \
    --mapping ../results/llm_council_evals/location-image-mapping.csv \
    --output optimized_prompts.csv \
    --concurrency 1 \
    --run-name "council-eval-$SLURM_JOB_ID"

echo ""
echo "=== Done: $(date) ==="
