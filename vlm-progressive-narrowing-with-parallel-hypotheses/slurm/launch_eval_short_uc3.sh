#!/bin/bash
# Parallel launcher for the LLM-as-judge eval on gpu_h100_short.
#
# Analogous to launch_short_uc3.sh, but for `python -m eval judge` instead of
# the council runner.
#
# 1. Finds image_ids in $VLM_RESULTS_DIR that don't yet have a verdict in
#    $VLM_EVAL_OUT/judge/<image_id>.json
# 2. Splits the remaining IDs into NUM_JOBS chunks
# 3. Submits one short-gpu sbatch per chunk, each with VLM_JUDGE_FILE_LIST set
#
# Usage:
#   cd $(ws_find vlm-judge-eval)
#
#   # 5 jobs x 10 images
#   bash slurm/launch_eval_short_uc3.sh 5 10
#
#   # custom results dir / output dir
#   VLM_RESULTS_DIR=results VLM_EVAL_OUT=eval_outputs_main \
#       bash slurm/launch_eval_short_uc3.sh 4 15

set -euo pipefail

NUM_JOBS="${1:-5}"
IMAGES_PER_JOB="${2:-10}"
PARTITION="${PARTITION:-gpu_h100_short}"
TIME="${TIME:-0:30:00}"
MEM="${MEM:-80G}"

PROJECT_DIR="$(ws_find vlm-judge-eval)"
cd "$PROJECT_DIR"

RESULTS_DIR="${VLM_RESULTS_DIR:-results_shuffle}"
EVAL_OUT="${VLM_EVAL_OUT:-eval_outputs_shuffle}"
JUDGE_OUT="$EVAL_OUT/judge"
FILE_LIST_DIR="$PROJECT_DIR/.vlm_judge_lists"

rm -rf "$FILE_LIST_DIR"
mkdir -p "$FILE_LIST_DIR" "$JUDGE_OUT"

echo "============================================="
echo "  Parallel LLM-as-Judge: uc3 ($PARTITION)"
echo "============================================="
echo "  Results dir:  $RESULTS_DIR"
echo "  Eval out:     $EVAL_OUT"
echo "  Judge model:  ${VLM_MODEL:-(default in run_eval_uc3.sh)}"
echo ""

# Find image_ids that don't yet have a judge verdict
REMAINING_LIST="$FILE_LIST_DIR/remaining.txt"

python3 -c "
import os, sys
results = '$RESULTS_DIR'
judge_out = '$JUDGE_OUT'

# Already-judged image_ids (filename without .json)
done = set()
if os.path.isdir(judge_out):
    for f in os.listdir(judge_out):
        if f.endswith('.json'):
            done.add(os.path.splitext(f)[0])

# Candidate image_ids: every subdir of results that contains a usable result.json
import json
all_ids = []
if os.path.isdir(results):
    for d in sorted(os.listdir(results)):
        rd = os.path.join(results, d)
        rj = os.path.join(rd, 'result.json')
        if not os.path.isdir(rd) or not os.path.isfile(rj):
            continue
        try:
            with open(rj) as f:
                data = json.load(f)
            if data.get('error'):
                continue
        except (json.JSONDecodeError, OSError):
            continue
        all_ids.append(d)

remaining = [i for i in all_ids if i not in done]
print(f'Total result.jsons:  {len(all_ids)}', file=sys.stderr)
print(f'Already judged:      {len(done)}', file=sys.stderr)
print(f'Remaining to judge:  {len(remaining)}', file=sys.stderr)

for i in remaining:
    print(i)
" > "$REMAINING_LIST"

TOTAL_REMAINING=$(wc -l < "$REMAINING_LIST")

if [ "$TOTAL_REMAINING" -eq 0 ]; then
    echo "All result.jsons already have a verdict. Nothing to do."
    rm -rf "$FILE_LIST_DIR"
    exit 0
fi

# Cap jobs if fewer images than requested
TOTAL_REQUESTED=$((NUM_JOBS * IMAGES_PER_JOB))
if [ "$NUM_JOBS" -gt "$TOTAL_REMAINING" ]; then
    NUM_JOBS=$((TOTAL_REMAINING / IMAGES_PER_JOB + 1))
    if [ "$NUM_JOBS" -gt "$TOTAL_REMAINING" ]; then
        NUM_JOBS="$TOTAL_REMAINING"
        IMAGES_PER_JOB=1
    fi
fi

echo "  Remaining:      $TOTAL_REMAINING image_ids"
echo "  Jobs:           $NUM_JOBS"
echo "  Images per job: $IMAGES_PER_JOB"
echo "  Will process:   $((NUM_JOBS * IMAGES_PER_JOB)) image_ids"
echo "  Partition:      $PARTITION"
echo "  Time per job:   $TIME"
echo ""

# Forward relevant env vars to each sbatch
ENV_EXPORTS=""
for var in VLM_MODEL VLM_RESULTS_DIR VLM_GT_CSV VLM_IMAGE_ROOT VLM_EVAL_OUT \
           VLM_MAX_MODEL_LEN VLM_GPU_MEMORY_UTIL VLM_JUDGE_CONCURRENCY \
           VLM_JUDGE_MAX_TOKENS VLM_MM_PROCESSOR_KWARGS; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        ENV_EXPORTS="$ENV_EXPORTS,$var=$val"
    fi
done

# Make sure RESULTS_DIR / EVAL_OUT are exported (they may have come from defaults)
ENV_EXPORTS="$ENV_EXPORTS,VLM_RESULTS_DIR=$RESULTS_DIR,VLM_EVAL_OUT=$EVAL_OUT"

# Split remaining IDs into per-job chunks and submit
JOB_NUM=0
LINE_NUM=0
TOTAL_ASSIGNED=0
MAX_TOTAL=$((NUM_JOBS * IMAGES_PER_JOB))
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
            --gres=gpu:1 \
            --cpus-per-task=16 \
            --exclusive \
            --job-name="vlm-judge-${JOB_NUM}" \
            --output="vlm-judge-${JOB_NUM}-%j.out" \
            --error="vlm-judge-${JOB_NUM}-%j.err" \
            --parsable \
            --export=ALL${ENV_EXPORTS},VLM_JUDGE_FILE_LIST="$CURRENT_LIST" \
            slurm/run_eval_uc3.sh)

        COUNT=$(wc -l < "$CURRENT_LIST")
        echo "  Job $JOB_NUM: $COUNT image_ids → JOBID=$JOB_ID"

        JOB_NUM=$((JOB_NUM + 1))
        LINE_NUM=0
        CURRENT_LIST="$FILE_LIST_DIR/job_${JOB_NUM}.txt"
        > "$CURRENT_LIST"
    fi
done < "$REMAINING_LIST"

# Submit last job if it has image_ids
if [ -s "$CURRENT_LIST" ]; then
    JOB_ID=$(sbatch \
        --partition="$PARTITION" \
        --time="$TIME" \
        --mem="$MEM" \
        --gres=gpu:1 \
        --cpus-per-task=16 \
        --exclusive \
        --job-name="vlm-judge-${JOB_NUM}" \
        --output="vlm-judge-${JOB_NUM}-%j.out" \
        --error="vlm-judge-${JOB_NUM}-%j.err" \
        --parsable \
        --export=ALL${ENV_EXPORTS},VLM_JUDGE_FILE_LIST="$CURRENT_LIST" \
        slurm/run_eval_uc3.sh)

    COUNT=$(wc -l < "$CURRENT_LIST")
    echo "  Job $JOB_NUM: $COUNT image_ids → JOBID=$JOB_ID"
fi

echo ""
echo "Submitted $((JOB_NUM + 1)) jobs on $PARTITION."
echo "Monitor: squeue -u \$USER"
echo "Verdicts so far: find $JUDGE_OUT/ -name '*.json' | wc -l"
echo ""
echo "Once everything is done, run aggregation + report:"
echo "  python -m eval aggregate --out $EVAL_OUT"
echo "  python -m eval report    --out $EVAL_OUT"
