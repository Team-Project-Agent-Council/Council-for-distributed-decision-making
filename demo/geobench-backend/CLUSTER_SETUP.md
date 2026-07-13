# Cluster Setup - Progressive Narrowing Demo

Step-by-step guide to expose Gemma 4 from bwUniCluster 3 to your local
GeoBench backend.

---

## 1. SSH into the login node

```bash
ssh <kit-username>@uc3.scc.kit.edu
```

## 2. One-time setup

The cluster only needs to serve `google/gemma-4-31b-it` over vLLM's
OpenAI-compatible endpoint. The council pipeline itself runs in the
GeoBench backend on your local machine; nothing council-specific has to
be installed on the cluster.

Pick any workspace name you like (referenced as `<workspace-name>`
below). Anything descriptive works.

```bash
ws_allocate <workspace-name> 30   # 30-day workspace, extend later with ws_extend
cd $(ws_find <workspace-name>)

# Any setup script that provisions a Python venv with vLLM + its CUDA
# dependencies is sufficient here.
```

## 3. Submit a long-lived vLLM serving job

For the demo we only need vLLM to be up - the GeoBench adapter does the
orchestration. Submit a manual sbatch with the model + thinking
settings:

```bash
cd $(ws_find <workspace-name>)

sbatch \
  --partition=gpu_h100 \
  --time=08:00:00 \
  --gres=gpu:1 \
  --mem=80G \
  --job-name=vlm-demo-server \
  --output=logs/demo-server-%j.out \
  --error=logs/demo-server-%j.err \
  --export=ALL,VLM_MODEL=google/gemma-4-31b-it,VLM_MAX_MODEL_LEN=16384,VLM_JUDGE_THINKING=true,VLM_MM_PROCESSOR_KWARGS='{"max_soft_tokens": 1120}' \
  --wrap="
    set -e
    module load devel/python/3.13.1
    module load devel/cuda/12.8
    source .venv/bin/activate

    PORT=\$((8000 + (\$SLURM_JOB_ID % 1000)))
    echo \"vLLM port: \$PORT\"
    echo \"vLLM node: \$(hostname)\"

    python -m vllm.entrypoints.openai.api_server \\
      --model \$VLM_MODEL \\
      --dtype auto \\
      --gpu-memory-utilization 0.9 \\
      --max-model-len \$VLM_MAX_MODEL_LEN \\
      --trust-remote-code \\
      --host 0.0.0.0 \\
      --port \$PORT \\
      --mm-processor-kwargs \"\$VLM_MM_PROCESSOR_KWARGS\"
  "
```

> **Why the dynamic port?** The shared cluster forbids a fixed port across
> nodes. The job ID is mixed into the port number. You'll read the actual
> port out of the log file in step 4.

Watch the queue and grab the assigned node + port:

```bash
squeue -u $USER

# Once the job is RUNNING, grep the port from the log:
JOB_ID=<from squeue>
NODE=$(scontrol show job $JOB_ID | grep -oP 'NodeList=\K\S+')
PORT=$(grep 'vLLM port:' logs/demo-server-${JOB_ID}.out | awk '{print $NF}')
echo "Tunnel target: $NODE:$PORT"
```

Wait until the log shows `Application startup complete` before opening
the tunnel. On first run, Gemma 4 needs to load weights and compile its
KV cache before it becomes responsive.

## 4. Open the SSH tunnel from your laptop

In a **second terminal on your local machine**:

```bash
# Replace NODE and PORT with the values from step 3.
ssh -N -L 8001:NODE:PORT <kit-username>@uc3.scc.kit.edu
```

Leave that terminal running. From your laptop, `http://localhost:8001/v1`
now points at the cluster's vLLM endpoint.

Verify:

```bash
curl -s http://localhost:8001/v1/models | jq
# -> { "data": [ { "id": "google/gemma-4-31b-it", … } ] }
```

## 5. Point GeoBench at the tunnel

In `geobench-backend/.env`:

```
VLM_API_BASE=http://localhost:8001/v1
VLM_MODEL=google/gemma-4-31b-it
VLM_JUDGE_MODEL=google/gemma-4-31b-it
VLM_JUDGE_THINKING=true
VLM_MAX_MODEL_LEN=16384
VLM_CALL_TIMEOUT=600
```

Then start the backend as usual:

```bash
cd geobench-backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

## 6. Smoke test

From `geobench-backend/`:

```bash
source .venv/bin/activate

PYTHONPATH=. python -c "
import asyncio
from dotenv import load_dotenv
load_dotenv()
from council_adapters.progressive_narrowing_adapter import run_progressive_narrowing
async def emit(t, d):
    print(t, str(d)[:120])
asyncio.run(run_progressive_narrowing(
    open('$DEMO_DATASET_DIR/6fGwHxCTvCbaK77Q_5.png', 'rb').read(),
    'image/png',
    emit,
))
"
```

If you see `agent_assessment` events streaming and a `final_result` at
the end, you're good - start the frontend with `npm run dev` and the
demo runs against the real cluster.

## 7. Tear-down

```bash
# On the cluster - cancel the vLLM job to free the GPU:
scancel <JOB_ID>

# On your laptop - close the tunnel terminal (Ctrl-C).
```
