#!/bin/bash
#SBATCH --job-name=vlm-eval-debate
#SBATCH --partition=single
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Evaluate the Debate approach results on bwUniCluster uc3.
# No GPU needed : CPU-only metrics + plots.
#
# Usage (login node):
#   cd $(ws_find vlm-council-debate)
#
#   # Stage 1 only (no LLM judge):
#   sbatch slurm/run_eval_uc3.sh
#
#   # With LLM judge (requires vLLM running separately or external API):
#   VLM_JUDGE_LLM_MODEL=google/gemma-4-31b-it \
#   VLM_JUDGE_LLM_API_BASE=http://localhost:8000/v1 \
#       sbatch slurm/run_eval_uc3.sh --with-judge
#
# Key env vars (all optional, have defaults):
#   VLM_RESULTS_DIR   : path to results dir     (default: results_debate_adversarial)
#   VLM_GT_CSV        : path to ground-truth CSV (default: gt.csv)
#   VLM_EVAL_OUT      : output dir for reports   (default: eval_out/debate)
#   VLM_JUDGE_LLM_MODEL      : judge LLM model name
#   VLM_JUDGE_LLM_API_BASE   : judge LLM API base URL
#   VLM_JUDGE_CONCURRENCY    : parallel judge calls (default: 8)
#   VLM_JUDGE_LIMIT          : limit N images for judge (default: all)

set -uo pipefail

WITH_JUDGE=false
for arg in "$@"; do
    [[ "$arg" == "--with-judge" ]] && WITH_JUDGE=true
done

PROJECT_DIR="$(ws_find vlm-council-debate)"
cd "$PROJECT_DIR"

echo "============================================="
echo "  VLM Council : Debate Evaluation (uc3)"
echo "============================================="
echo "  Job-ID:   ${SLURM_JOB_ID:-local}"
echo "  Node:     $(hostname)"
echo "  With LLM judge: $WITH_JUDGE"
echo ""

# Environment
module load devel/python/3.13.1 || { echo "ERROR: Cannot load Python module"; exit 1; }
set -e
source "$PROJECT_DIR/.venv/bin/activate"

# Paths
RESULTS_DIR="${VLM_RESULTS_DIR:-results_debate_adversarial}"
GT_CSV="${VLM_GT_CSV:-gt.csv}"
EVAL_OUT="${VLM_EVAL_OUT:-eval_out/debate}"
mkdir -p "$EVAL_OUT"

echo "  Results dir: $RESULTS_DIR"
echo "  GT CSV:      $GT_CSV"
echo "  Eval output: $EVAL_OUT"
echo ""

# Check inputs
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results dir not found: $RESULTS_DIR" >&2
    exit 1
fi
if [ ! -f "$GT_CSV" ]; then
    echo "ERROR: GT CSV not found: $GT_CSV" >&2
    exit 1
fi

N_RESULTS=$(find "$RESULTS_DIR" -name "result.json" 2>/dev/null | wc -l)
echo "Found $N_RESULTS result.json files in $RESULTS_DIR"
echo ""

# Stage 1: geo + agents metrics (no GPU, fast)
echo "=== Stage 1: Geo + Agent metrics ==="
python -m eval_debate geo \
    --results "$RESULTS_DIR" \
    --gt "$GT_CSV" \
    --out "$EVAL_OUT"

python -m eval_debate agents \
    --results "$RESULTS_DIR" \
    --gt "$GT_CSV" \
    --out "$EVAL_OUT"

# Stage 2: LLM judge (optional, requires API)
if [ "$WITH_JUDGE" = true ]; then
    echo ""
    echo "=== Stage 2: LLM Judge ==="

    export VLM_JUDGE_LLM_MODEL="${VLM_JUDGE_LLM_MODEL:-}"
    export VLM_JUDGE_LLM_API_BASE="${VLM_JUDGE_LLM_API_BASE:-}"
    JUDGE_CONCURRENCY="${VLM_JUDGE_CONCURRENCY:-8}"
    JUDGE_LIMIT="${VLM_JUDGE_LIMIT:-}"

    JUDGE_ARGS="--results $RESULTS_DIR --gt $GT_CSV --out $EVAL_OUT --concurrency $JUDGE_CONCURRENCY"
    [ -n "$JUDGE_LIMIT" ] && JUDGE_ARGS="$JUDGE_ARGS --limit $JUDGE_LIMIT"
    [ -n "$VLM_JUDGE_LLM_MODEL" ] && JUDGE_ARGS="$JUDGE_ARGS --model $VLM_JUDGE_LLM_MODEL"
    [ -n "$VLM_JUDGE_LLM_API_BASE" ] && JUDGE_ARGS="$JUDGE_ARGS --api-base $VLM_JUDGE_LLM_API_BASE"

    python -m eval_debate judge $JUDGE_ARGS

    echo ""
    echo "=== Stage 3: Aggregate judge results ==="
    python -m eval_debate aggregate --out "$EVAL_OUT"
fi

# Always: generate report
echo ""
echo "=== Generating report ==="
python -m eval_debate report --out "$EVAL_OUT"

echo ""
echo "============================================="
echo "  Evaluation complete"
echo "  Report: $EVAL_OUT/report.html"
echo "  Metrics:"
echo "    $EVAL_OUT/geo_metrics.json"
echo "    $EVAL_OUT/agent_metrics.json"
echo "    $EVAL_OUT/debate_stats.json"
[ "$WITH_JUDGE" = true ] && echo "    $EVAL_OUT/judge_summary.json"
echo "============================================="
