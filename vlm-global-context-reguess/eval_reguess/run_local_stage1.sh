#!/usr/bin/env bash
# Local Stage-1 only - no GPU, no LLM judge. Runs geo / agents + report
# against an existing results_global_context_re_guess_*/ tree.
# Use this on a laptop to iterate on deterministic metrics;
# the judge stage requires a GPU or API endpoint.
#
# Usage (from repo root):
#   eval_reguess/run_local_stage1.sh \
#       results_global_context_re_guess_1 \
#       Images/georc_locations.csv \
#       /tmp/eval_reguess_out

set -euo pipefail

RESULTS_DIR="${1:-results_global_context_re_guess_1}"
GT_CSV="${2:-Images/georc_locations.csv}"
OUT_DIR="${3:-eval_reguess_outputs}"

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

python -m eval_reguess all --skip-judge \
    --results "$RESULTS_DIR" \
    --gt "$GT_CSV" \
    --out "$OUT_DIR"

echo "[stage1] done - see $OUT_DIR/report.md"
