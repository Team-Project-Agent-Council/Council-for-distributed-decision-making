#!/usr/bin/env bash
# Local Stage-1 only - no GPU, no vLLM. Runs geo / agents / influence + report
# against an existing results_v12_pn/ tree. Use this on a laptop to iterate on
# the deterministic metrics; the judge stage requires the cluster.
#
# Usage:
#   vlm_council_v12/eval/run_local_stage1.sh \
#       vlm_council_v12/results_v12_pn \
#       georc_locations.csv \
#       /tmp/eval_test

set -euo pipefail

RESULTS_DIR="${1:-vlm_council_v12/results_v12_pn}"
GT_CSV="${2:-georc_locations.csv}"
OUT_DIR="${3:-eval_outputs}"

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

python -m eval all --skip-judge \
    --results "$RESULTS_DIR" \
    --gt "$GT_CSV" \
    --out "$OUT_DIR"

echo "[stage1] done - see $OUT_DIR/report.md"
