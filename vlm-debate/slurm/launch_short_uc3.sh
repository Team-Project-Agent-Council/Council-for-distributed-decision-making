#!/bin/bash
# Parallel launcher for gpu_h100_short on uc3.
#
# 1. Finds images NOT yet processed in the output dir
# 2. Splits them into chunks of IMAGES_PER_JOB
# 3. Submits each chunk as a separate SLURM job with --file-list
#
# All env vars (VLM_MODEL, VLM_OUTPUT_DIR, etc.) are forwarded to the jobs.
#
# Usage:
#   cd $(ws_find vlm-council-debate)
#
#   # Qwen Instruct : 5 jobs x 10 images
#   bash slurm/launch_short_uc3.sh 5 10
#
#   # Gemma 4 with judge thinking : 5 jobs x 5 images
#   VLM_MODEL=google/gemma-4-31b-it VLM_JUDGE_THINKING=true VLM_OUTPUT_DIR=results_gemma4 VLM_MAX_MODEL_LEN=16384 \
#       bash slurm/launch_short_uc3.sh 5 5
#
#   # Qwen Thinking : 5 jobs x 2 images
#   VLM_MODEL=Qwen/Qwen3-VL-32B-Thinking VLM_OUTPUT_DIR=results_thinking VLM_MAX_MODEL_LEN=32768 \
#       bash slurm/launch_short_uc3.sh 5 2

set -euo pipefail

NUM_JOBS="${1:-5}"
IMAGES_PER_JOB="${2:-5}"
PARTITION="${PARTITION:-gpu_h100_short}"
TIME="${TIME:-0:30:00}"
MEM="${MEM:-80G}"

PROJECT_DIR="$(ws_find vlm-council-debate)"
cd "$PROJECT_DIR"

IMAGE_DIR="Images"
OUTPUT_DIR="${VLM_OUTPUT_DIR:-results}"
FILE_LIST_DIR="$PROJECT_DIR/.vlm_job_lists"

rm -rf "$FILE_LIST_DIR"
mkdir -p "$FILE_LIST_DIR" "$OUTPUT_DIR"

echo "============================================="
echo "  Parallel VLM Council : uc3 ($PARTITION)"
echo "============================================="
echo "  Model: ${VLM_MODEL:-google/gemma-4-31b-it}"
echo "  Output: $OUTPUT_DIR"
echo ""

# Find remaining images
REMAINING_LIST="$FILE_LIST_DIR/remaining.txt"

python3 -c "
import json, os, sys

image_dir = '$IMAGE_DIR'
output_dir = '$OUTPUT_DIR'
extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

images = sorted(f for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))
                and os.path.splitext(f)[1].lower() in extensions)

done = set()
if os.path.isdir(output_dir):
    for d in os.listdir(output_dir):
        result_path = os.path.join(output_dir, d, 'result.json')
        if os.path.isfile(result_path):
            try:
                with open(result_path) as f:
                    data = json.load(f)
                if not data.get('error'):
                    done.add(d)
            except (json.JSONDecodeError, OSError):
                pass

remaining = [img for img in images if os.path.splitext(img)[0] not in done]

print(f'Total images: {len(images)}', file=sys.stderr)
print(f'Already done: {len(done)}', file=sys.stderr)
print(f'Remaining:    {len(remaining)}', file=sys.stderr)

for img in remaining:
    print(img)
" > "$REMAINING_LIST"

TOTAL_REMAINING=$(wc -l < "$REMAINING_LIST")

if [ "$TOTAL_REMAINING" -eq 0 ]; then
    echo "All images already processed!"
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

echo "  Remaining:      $TOTAL_REMAINING images"
echo "  Jobs:           $NUM_JOBS"
echo "  Images per job: $IMAGES_PER_JOB"
echo "  Will process:   $((NUM_JOBS * IMAGES_PER_JOB)) images"
echo "  Partition:      $PARTITION"
echo "  Time per job:   $TIME"
echo ""

# Build env var exports to forward to sbatch
ENV_EXPORTS=""
for var in VLM_MODEL VLM_OUTPUT_DIR VLM_MAX_MODEL_LEN VLM_GPU_MEMORY_UTIL VLM_JUDGE_THINKING VLM_IMAGE_TOKEN_BUDGET VLM_CALL_TIMEOUT VLM_MM_PROCESSOR_KWARGS DEBATE_MAX_ROUNDS DEBATE_MAX_EXCHANGES DEBATE_MIN_CONFIDENCE; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        ENV_EXPORTS="$ENV_EXPORTS --export=ALL,$var=$val"
    fi
done

# Split remaining images into per-job file lists and submit
JOB_NUM=0
LINE_NUM=0
TOTAL_ASSIGNED=0
MAX_TOTAL=$((NUM_JOBS * IMAGES_PER_JOB))
CURRENT_LIST="$FILE_LIST_DIR/job_${JOB_NUM}.txt"
> "$CURRENT_LIST"

while IFS= read -r image_name; do
    if [ "$TOTAL_ASSIGNED" -ge "$MAX_TOTAL" ]; then
        break
    fi

    echo "$image_name" >> "$CURRENT_LIST"
    LINE_NUM=$((LINE_NUM + 1))
    TOTAL_ASSIGNED=$((TOTAL_ASSIGNED + 1))

    if [ "$LINE_NUM" -ge "$IMAGES_PER_JOB" ] && [ "$JOB_NUM" -lt "$((NUM_JOBS - 1))" ]; then
        JOB_ID=$(sbatch \
            --partition="$PARTITION" \
            --time="$TIME" \
            --mem="$MEM" \
            --exclusive \
            --job-name="vlm-s-${JOB_NUM}" \
            --parsable \
            $ENV_EXPORTS \
            slurm/run_council_uc3.sh --file-list "$CURRENT_LIST")

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
        --job-name="vlm-s-${JOB_NUM}" \
        --parsable \
        $ENV_EXPORTS \
        slurm/run_council_uc3.sh --file-list "$CURRENT_LIST")

    COUNT=$(wc -l < "$CURRENT_LIST")
    echo "  Job $JOB_NUM: $COUNT images → JOBID=$JOB_ID"
fi

echo ""
echo "Submitted $((JOB_NUM + 1)) jobs on $PARTITION."
echo "Monitor: squeue -u \$USER"
echo "Results: find $OUTPUT_DIR/ -name 'result.json' | wc -l"
