# GeoBench: Progressive Narrowing with Parallel Hypotheses

Team project, University of Mannheim, on the topic
**"Multi-LLM Council for Distributed Decision Making"**. Over the course
of the project we analysed several council architectures for
vision-language geolocation. This repository contains the visualisation
for a single one of those architectures, **Progressive Narrowing with
Parallel Hypotheses**, and was used during the final presentation.

**🌐 Interactive demo:** <https://team-project-agent-council.github.io/Council-for-distributed-decision-making/demo/>

The site replays a pre-recorded Progressive Narrowing run on a Uruguay
Street View image at 3× speed, showing every phase of the council's
deliberation. No setup or cluster access required - click through the
timeline and inspect the reasoning traces at your own pace.

---

## Table of contents

- [Scope and context](#scope-and-context)
- [Architecture](#architecture)
- [Pipeline phases](#pipeline-phases)
- [Repository layout](#repository-layout)
- [Local setup](#local-setup)
  - [1. Backend](#1-backend)
  - [2. Frontend](#2-frontend)
  - [3. vLLM on the cluster](#3-vllm-on-the-cluster)
  - [4. SSH tunnel](#4-ssh-tunnel)
  - [5. Point the backend at the tunnel](#5-point-the-backend-at-the-tunnel)
  - [6. End-to-end smoke test](#6-end-to-end-smoke-test)

---

## Scope and context

This repository realises one specific council design:

- **Five specialists** run in parallel, each restricted to a single
  visual domain (language, landscape, botanics, regulatory, meta).
- **A judge agent** aggregates their independent assessments and drives
  a hierarchical narrowing: region consensus, then region hypotheses,
  then constrained country hypotheses, then final country.

However the focus of this repository was the visualization of this approach to be able to provide more detailed insights within the final presentation. The actual approach which was deployed on the cluster and also used to produce the results presented in the reports can be found in the for this approach dedicated GitHub repository.

## Architecture

```
       Browser ──► Next.js frontend ──► FastAPI backend ──► vendored vlm_council
                                                                     │
                                                                     ▼
                                                             vLLM (SSH-tunnelled)
                                                                     │
                                                                     ▼
                                                         GPU cluster (Gemma 4-31B-IT)
```

The vendored `vlm_council` package sits under
`geobench-backend/vendor/vlm_council/`. Extensions specific to this
visualisation, namely per-agent SSE tracing, rescue parsing for
malformed JSON returns, and country-name canonicalisation, live in the
adapter layer at `council_adapters/progressive_narrowing_adapter.py` as
monkey-patches applied per run.

## Pipeline phases

The `/test-council` view renders each phase in real time via SSE:

| # | Phase |
| --- | --- |
| 1 | Specialist Assessment |
| 2 | Region Consensus Check |
| 3 | Region Hypothesis Evaluation (Path B) |
| 4 | Region-Bound Assessment (Path B) |
| 5 | Country Hypothesis Evaluation |
| 6 | Country Determination |

Path selection happens at step 2. If all five specialists agree on the
same world region, steps 3 and 4 are skipped and the pipeline proceeds
directly to step 5 (Path A). Otherwise the judge synthesises region
hypotheses, evaluates them, commits to the winning region, and re-runs
the specialists inside that region before continuing (Path B).

## Repository layout

```
Team Projekt GeoBench/
├── geobench-frontend/     Next.js 16 UI (Progressive Narrowing demo, council info)
├── geobench-backend/      FastAPI service + vendored vlm_council LangGraph pipeline
└── README.md              This file
```

### Dataset

The Google Street View dataset (500 images + `georc_locations.csv`
ground truth) is not distributed with the repository. The images are
covered by the Google Maps Platform terms of service and are also too
large for a public repo. To enable the `Random` control on
`/test-council`, download the dataset from the project OneDrive and set
`DEMO_DATASET_DIR` in `geobench-backend/.env` to the extracted folder:

```env
DEMO_DATASET_DIR=/absolute/path/to/VLM-Council/Images
```

Without this the backend still runs and the local demo works for
uploaded images; only the `Random` picker is disabled.

## Local setup

### Prerequisites

- Python 3.13+
- Node.js 20+ with npm
- A vLLM endpoint capable of serving `google/gemma-4-31b-it` on the
  OpenAI-compatible protocol (in our case an SSH-tunnelled port from
  bwUniCluster 3)
- A KIT account for `uc3.scc.kit.edu` if reproducing the cluster path

### 1. Backend

```bash
cd geobench-backend
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

Sanity checks:

```bash
curl -s http://localhost:8000/health          # -> {"status":"ok"}
curl -s http://localhost:8000/api/council/agents | jq
```

### 2. Frontend

```bash
cd geobench-frontend
npm install
cp .env.example .env.local
npm run dev
```

Open <http://localhost:3000>:

- `/`, the landing page with pipeline overview
- `/council`, council metadata (5 specialists + 6 collaboration steps)
- `/test-council`, the Progressive Narrowing demo

Without a running backend and a live vLLM endpoint, `/test-council`
raises network errors. The `/council` metadata page is fully static and
works standalone.

### 3. vLLM on the cluster

The cluster only needs to serve `google/gemma-4-31b-it` over vLLM's
OpenAI-compatible endpoint. The council pipeline itself runs in the
GeoBench backend on your local machine; nothing council-specific has to
be installed on the cluster.

The example below submits a job on the `gpu_h100` partition. Adjust
`--partition`, `--time`, and (optionally) `--begin` to your needs.

> **Partition choice.** The regular `gpu_h100` partition often has long
> queue times on bwUniCluster 3, so for a quick demo it is usually more
> practical to request a short job on `dev_gpu_h100` (or the equivalent
> short partition) with `--time=00:30:00`. The dev and short partitions
> have significantly shorter waits, at the cost of a hard 30-minute
> ceiling on wall-clock time.

**One-time cluster setup**. Pick any workspace name you like (referenced
as `<workspace-name>` below); anything descriptive works.

```bash
ssh <kit-username>@uc3.scc.kit.edu

# Allocate a workspace for the vLLM install. The second argument is the
# lifetime in days; extend later with `ws_extend` if needed.
ws_allocate <workspace-name> 30

cd $(ws_find <workspace-name>)

# Any setup script that provisions a Python venv with vLLM + its CUDA
# dependencies is sufficient here. The GeoBench project used
# `bash setup_uc3.sh` from the vlm_council upstream repository, which
# also happens to install langchain (harmless but unused in this path).
```

**Job submission:**

```bash
cd $(ws_find <workspace-name>)

sbatch \
  --partition=gpu_h100 \
  --time=03:00:00 \
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

    # Route compile / model caches into the workspace so a stale
    # /scratch/slurm_tmpdir/job_XXX from a previous run cannot be inherited.
    export WORKSPACE=\$(pwd)
    export TORCHINDUCTOR_CACHE_DIR=\$WORKSPACE/.cache/torch_inductor
    export TRITON_CACHE_DIR=\$WORKSPACE/.cache/triton
    export HF_HOME=\$WORKSPACE/.cache/hf
    export XDG_CACHE_HOME=\$WORKSPACE/.cache
    mkdir -p \$TORCHINDUCTOR_CACHE_DIR \$TRITON_CACHE_DIR \$HF_HOME

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

The dynamic port (`$((8000 + SLURM_JOB_ID % 1000))`) is required because
the shared cluster forbids a fixed port across nodes. The port is
derived from the job ID and read back from the log.

**Once the job is `R` (RUNNING)**, resolve node and port:

```bash
JOB_ID=$(squeue -u $USER -h -o "%i" | head -1)
NODE=$(scontrol show job $JOB_ID | grep -oP 'NodeList=\K\S+')
PORT=$(grep 'vLLM port:' logs/demo-server-${JOB_ID}.out | awk '{print $NF}')
echo "$NODE:$PORT"
```

Wait until the log reports `INFO: Application startup complete` before
opening the tunnel.

### 4. SSH tunnel

On the local machine, in a terminal that stays open for the session:

```bash
ssh -N -L 8001:<NODE>:<PORT> <kit-username>@uc3.scc.kit.edu
```

Verify the tunnel:

```bash
curl -s http://localhost:8001/v1/models | jq
# -> { "data": [ { "id": "google/gemma-4-31b-it", … } ] }
```

### 5. Point the backend at the tunnel

The `.env.example` already defaults to the tunnel port. If the tunnel
runs on a different port, adjust `geobench-backend/.env`:

```env
VLM_API_BASE=http://localhost:8001/v1
VLM_MODEL=google/gemma-4-31b-it
VLM_JUDGE_MODEL=google/gemma-4-31b-it
VLM_JUDGE_THINKING=true
VLM_MAX_MODEL_LEN=16384
VLM_CALL_TIMEOUT=600
```

### 6. End-to-end smoke test

From the repo root:

```bash
cd geobench-backend
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

The stream should produce `agent_assessment`, `country_hypotheses`,
`country_evaluation`, and `final_result` events, ending with `done`.

This can also done via the UI under <http://localhost:3000/test-council>, click `Random` or select an image from the local file system and wait for the cluster to process the image. 
