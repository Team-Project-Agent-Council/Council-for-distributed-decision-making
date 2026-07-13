#!/bin/bash
# SLURM dependency-chain launcher for run_council_uc3.sh on uc3.
#
# Pre-submits N copies of the 30-min job at once. Each job is wired with
# --dependency=afterany:<previous_jobid> so SLURM serializes them: only one
# runs at a time, and the next starts after the previous finishes (success
# OR failure : `afterany`, not `afterok`, so a single timeout doesn't kill
# the chain).
#
# Each job runs the same run_council_uc3.sh, which auto-skips
# images whose results/<image>/result.json already exists. So each link in
# the chain just continues from where the previous left off.
#
# Usage:
#   cd $(ws_find vlm-council-tournament)
#
#   # Smoke test : 1 link, 1 image
#   bash slurm/chain_uc3.sh 1 --limit 1
#
#   # Real run : 4 sequential 30-min slots
#   bash slurm/chain_uc3.sh 4
#
# Monitor:
#   squeue -u $USER
#   find results_tournament/ -name 'result.json' | wc -l

set -euo pipefail

NUM_LINKS="${1:-5}"
shift || true
EXTRA_ARGS="$@"

PROJECT_DIR="$(ws_find vlm-council-tournament)"
cd "$PROJECT_DIR"

SCRIPT="slurm/run_council_uc3.sh"
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: $SCRIPT not found in $PROJECT_DIR" >&2
    exit 1
fi

# Forward the same env vars the script reads
ENV_EXPORTS=""
for var in VLM_MODEL VLM_OUTPUT_DIR VLM_RESULT_NAME VLM_IMAGE_DIR VLM_MAX_MODEL_LEN \
           VLM_GPU_MEMORY_UTIL VLM_JUDGE_MODEL \
           VLM_JUDGE_THINKING VLM_CALL_TIMEOUT VLM_MM_PROCESSOR_KWARGS \
           VLM_DATA_DIR VLM_RAG_MAX_REFS_PER_ROUND VLM_RAG_MAX_REFS_PER_COUNTRY \
           VLM_TOURNAMENT_FINALISTS; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        ENV_EXPORTS="$ENV_EXPORTS,$var=$val"
    fi
done

echo "============================================="
echo "  Dependency chain : uc3 gpu_h100_short (tournament-only)"
echo "============================================="
echo "  Script:       $SCRIPT"
echo "  Links:        $NUM_LINKS  (each = one 30-min slot)"
echo "  Extra args:   ${EXTRA_ARGS:-<none>}"
echo "  Model:        ${VLM_MODEL:-google/gemma-4-31b-it}"
echo "  Output dir:   ${VLM_OUTPUT_DIR:-results_tournament_500}"
echo ""

PREV_JOBID=""
for ((i=1; i<=NUM_LINKS; i++)); do
    if [ -z "$PREV_JOBID" ]; then
        DEP_ARG=""
    else
        DEP_ARG="--dependency=afterany:$PREV_JOBID"
    fi

    JOBID=$(sbatch \
        --parsable \
        --job-name="vlm-tourney-${i}" \
        ${DEP_ARG} \
        --export=ALL${ENV_EXPORTS} \
        "$SCRIPT" $EXTRA_ARGS)

    echo "  Link $i: JOBID=$JOBID  ${DEP_ARG:+(after $PREV_JOBID)}"
    PREV_JOBID="$JOBID"
done

echo ""
echo "Submitted $NUM_LINKS links. SLURM will run them one after another."
echo "Monitor: squeue -u \$USER"
echo "Cancel chain: scancel -u \$USER --name=vlm-tourney"
