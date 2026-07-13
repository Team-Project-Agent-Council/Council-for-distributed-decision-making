#!/usr/bin/env bash
# Local Stage-1 only -- no GPU, no LLM.
# Runs geo / agents / report against an existing results_hub_and_spoke/ tree.
#
# Usage:
#   bash eval_hubspoke/run_local_stage1.sh \
#       results_hub_and_spoke \
#       Images/georc_locations.csv \
#       /tmp/eval_hubspoke_out

set -euo pipefail

RESULTS_DIR="${1:-results_hub_and_spoke}"
GT_CSV="${2:-Images/georc_locations.csv}"
OUT_DIR="${3:-eval_hubspoke_outputs}"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: results dir $RESULTS_DIR does not exist" >&2
    exit 1
fi
if [[ ! -f "$GT_CSV" ]]; then
    echo "ERROR: ground-truth CSV $GT_CSV does not exist" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[stage1] results: $RESULTS_DIR"
echo "[stage1] gt:      $GT_CSV"
echo "[stage1] out:     $OUT_DIR"

python -m eval_hubspoke all --skip-judge \
    --results "$RESULTS_DIR" \
    --gt "$GT_CSV" \
    --out "$OUT_DIR"

echo "[stage1] done -- see $OUT_DIR/report.md"
