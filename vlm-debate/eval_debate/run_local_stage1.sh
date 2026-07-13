#!/usr/bin/env bash
# Run the Debate evaluation pipeline (geo + agents metrics; skip LLM judge).
# Usage: bash eval_debate/run_local_stage1.sh <results_dir> <gt_csv> <out_dir>
#
# Example:
#   bash eval_debate/run_local_stage1.sh \
#       results_debate_adversarial \
#       /path/to/gt.csv \
#       eval_out/debate_adversarial

set -euo pipefail

RESULTS="${1:?Usage: $0 <results_dir> <gt_csv> <out_dir>}"
GT="${2:?Usage: $0 <results_dir> <gt_csv> <out_dir>}"
OUT="${3:?Usage: $0 <results_dir> <gt_csv> <out_dir>}"

echo "[run_local_stage1] results=$RESULTS  gt=$GT  out=$OUT"

python -m eval_debate all \
    --results "$RESULTS" \
    --gt "$GT" \
    --out "$OUT" \
    --skip-judge

echo "[run_local_stage1] done - report at $OUT/report.html"
