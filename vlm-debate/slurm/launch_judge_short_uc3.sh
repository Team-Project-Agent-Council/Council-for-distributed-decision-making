#!/bin/bash
# Parallel launcher for the LLM-as-Judge stage on gpu_h100_short (uc3).
#
# 1. Finds image IDs that have a council result.json but NOT yet a judge verdict
#    in $VLM_EVAL_OUT/judge/<image_id>.json
# 2. Splits them into chunks of IMAGES_PER_JOB
# 3. Submits each chunk as a separate SLURM job that runs slurm/run_judge_uc3.sh
#    with --file-list pointing at the chunk
#
# All relevant env vars (VLM_RESULTS_DIR, VLM_GT_CSV, VLM_EVAL_OUT, VLM_IMAGE_ROOT,
# VLM_JUDGE_LLM_MODEL, VLM_JUDGE_CONCURRENCY, VLM_MAX_MODEL_LEN, ...) are forwarded.
#
# Usage (on the cluster):
#   cd $(ws_find vlm-council-debate)
#
#   # Default: judge results_debate_adversarial → eval_out/debate_adversarial
#   #          5 jobs x 20 images = 100 images per launch (re-run to drain the rest)
#   bash slurm/launch_judge_short_uc3.sh 5 20
#
#   # Custom results dir / output dir
#   VLM_RESULTS_DIR=results_debate_adversarial \
#   VLM_EVAL_OUT=eval_out/debate_adversarial \
#       bash slurm/launch_judge_short_uc3.sh 10 25

set -euo pipefail

NUM_JOBS="${1:-5}"
IMAGES_PER_JOB="${2:-20}"
PARTITION="${PARTITION:-gpu_h100_short}"
TIME="${TIME:-0:30:00}"
MEM="${MEM:-80G}"

PROJECT_DIR="$(ws_find vlm-council-debate)"
cd "$PROJECT_DIR"

RESULTS_DIR="${VLM_RESULTS_DIR:-results_debate_adversarial}"
GT_CSV="${VLM_GT_CSV:-Images/georc_locations.csv}"
EVAL_OUT="${VLM_EVAL_OUT:-eval_out/debate_adversarial}"
IMAGE_ROOT="${VLM_IMAGE_ROOT:-Images}"
FILE_LIST_DIR="$PROJECT_DIR/.vlm_judge_lists"

JUDGE_DIR="$EVAL_OUT/judge"
mkdir -p "$JUDGE_DIR" "$EVAL_OUT"
rm -rf "$FILE_LIST_DIR"
mkdir -p "$FILE_LIST_DIR"

echo "============================================="
echo "  Parallel LLM-as-Judge : uc3 ($PARTITION)"
echo "============================================="
echo "  Results dir:   $RESULTS_DIR"
echo "  GT CSV:        $GT_CSV"
echo "  Eval out:      $EVAL_OUT"
echo "  Image root:    $IMAGE_ROOT"
echo "  Judge model:   ${VLM_JUDGE_LLM_MODEL:-google/gemma-4-31b-it}"
echo "  Partition:     $PARTITION"
echo "  Time per job:  $TIME"
echo ""

# Sanity-check inputs
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results dir not found: $RESULTS_DIR" >&2
    exit 1
fi
if [ ! -f "$GT_CSV" ]; then
    echo "ERROR: GT CSV not found: $GT_CSV" >&2
    exit 1
fi
if [ ! -d "$IMAGE_ROOT" ]; then
    echo "ERROR: Image root not found: $IMAGE_ROOT" >&2
    exit 1
fi

# Build the remaining-work list: image IDs with a council result.json but no judge verdict yet
REMAINING_LIST="$FILE_LIST_DIR/remaining.txt"

python3 -c "
import json, os, sys

results_dir = '$RESULTS_DIR'
judge_dir   = '$JUDGE_DIR'

# All image IDs with a valid (non-error) council trace
all_ids = []
for d in sorted(os.listdir(results_dir)):
    rj = os.path.join(results_dir, d, 'result.json')
    if not os.path.isfile(rj):
        continue
    try:
        with open(rj) as f:
            data = json.load(f)
        if data.get('error'):
            continue
    except (json.JSONDecodeError, OSError):
        continue
    all_ids.append(d)

# Already judged: any non-empty <id>.json in judge dir (skip would happen anyway)
done = set()
if os.path.isdir(judge_dir):
    for f in os.listdir(judge_dir):
        if not f.endswith('.json'):
            continue
        # Skip empty / zero-byte placeholder files
        full = os.path.join(judge_dir, f)
        try:
            if os.path.getsize(full) <= 2:  # essentially empty
                continue
        except OSError:
            continue
        done.add(f[:-5])

remaining = [i for i in all_ids if i not in done]

print(f'Council traces: {len(all_ids)}', file=sys.stderr)
print(f'Already judged: {len(done)}', file=sys.stderr)
print(f'Remaining:      {len(remaining)}', file=sys.stderr)

for i in remaining:
    print(i)
" > "$REMAINING_LIST"

TOTAL_REMAINING=$(wc -l < "$REMAINING_LIST")

if [ "$TOTAL_REMAINING" -eq 0 ]; then
    echo "All images already judged!"
    rm -rf "$FILE_LIST_DIR"
    exit 0
fi

# Cap jobs if fewer images than requested
if [ "$NUM_JOBS" -gt "$TOTAL_REMAINING" ]; then
    NUM_JOBS=$((TOTAL_REMAINING / IMAGES_PER_JOB + 1))
    if [ "$NUM_JOBS" -gt "$TOTAL_REMAINING" ]; then
        NUM_JOBS="$TOTAL_REMAINING"
        IMAGES_PER_JOB=1
    fi
fi

MAX_TOTAL=$((NUM_JOBS * IMAGES_PER_JOB))

echo "  Remaining:      $TOTAL_REMAINING image IDs"
echo "  Jobs:           $NUM_JOBS"
echo "  Images per job: $IMAGES_PER_JOB"
echo "  Will judge:     $MAX_TOTAL images (this launch)"
echo ""

# Build env var exports to forward to sbatch
ENV_EXPORTS=""
for var in VLM_RESULTS_DIR VLM_GT_CSV VLM_EVAL_OUT VLM_IMAGE_ROOT \
           VLM_JUDGE_LLM_MODEL VLM_JUDGE_CONCURRENCY VLM_JUDGE_MAX_TOKENS \
           VLM_MAX_MODEL_LEN VLM_GPU_MEMORY_UTIL VLM_MM_PROCESSOR_KWARGS \
           VLM_IMAGE_TOKEN_BUDGET; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        ENV_EXPORTS="$ENV_EXPORTS --export=ALL,$var=$val"
    fi
done

# Split remaining IDs into per-job file lists and submit
JOB_NUM=0
LINE_NUM=0
TOTAL_ASSIGNED=0
CURRENT_LIST="$FILE_LIST_DIR/job_${JOB_NUM}.txt"
> "$CURRENT_LIST"

while IFS= read -r image_id; do
    if [ "$TOTAL_ASSIGNED" -ge "$MAX_TOTAL" ]; then
        break
    fi

    echo "$image_id" >> "$CURRENT_LIST"
    LINE_NUM=$((LINE_NUM + 1))
    TOTAL_ASSIGNED=$((TOTAL_ASSIGNED + 1))

    if [ "$LINE_NUM" -ge "$IMAGES_PER_JOB" ] && [ "$JOB_NUM" -lt "$((NUM_JOBS - 1))" ]; then
        JOB_ID=$(sbatch \
            --partition="$PARTITION" \
            --time="$TIME" \
            --mem="$MEM" \
            --exclusive \
            --job-name="judge-${JOB_NUM}" \
            --parsable \
            $ENV_EXPORTS \
            slurm/run_judge_uc3.sh --file-list "$CURRENT_LIST")

        COUNT=$(wc -l < "$CURRENT_LIST")
        echo "  Job $JOB_NUM: $COUNT images → JOBID=$JOB_ID"

        JOB_NUM=$((JOB_NUM + 1))
        LINE_NUM=0
        CURRENT_LIST="$FILE_LIST_DIR/job_${JOB_NUM}.txt"
        > "$CURRENT_LIST"
    fi
done < "$REMAINING_LIST"

# Submit last job if it has images
if [ -s "$CURRENT_LIST" ]; then
    JOB_ID=$(sbatch \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --exclusive \
        --job-name="judge-${JOB_NUM}" \
        --parsable \
        $ENV_EXPORTS \
        slurm/run_judge_uc3.sh --file-list "$CURRENT_LIST")

    COUNT=$(wc -l < "$CURRENT_LIST")
    echo "  Job $JOB_NUM: $COUNT images → JOBID=$JOB_ID"
fi

echo ""
echo "Submitted $((JOB_NUM + 1)) judge jobs on $PARTITION."
echo "Monitor:  squeue -u \$USER"
echo "Verdicts: find $JUDGE_DIR -name '*.json' | wc -l"
echo ""
echo "Re-run this launcher after the queue drains to pick up any failed / unjudged images."
