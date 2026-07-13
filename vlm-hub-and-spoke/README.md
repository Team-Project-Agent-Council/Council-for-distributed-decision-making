# VLM Hub-and-Spoke

> **Results Overview:** [https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Hub_and_Spoke/report.html](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Hub_and_Spoke/report.html)  
> GT-based statistics and the LLM-as-Judge (Qwen 3.6) report for this approach, rendered as a GitHub Page.

A multi-agent council of five specialised Vision-Language Model agents
that collaboratively geolocate images. The council follows a
**Hub-and-Spoke** topology with a **dynamic judge dialogue**: the
specialist agents each produce an independent assessment; a Judge
(the "hub") reviews all five outputs and, when it finds contradictions,
sends **targeted follow-up questions to individual agents**. Only after
the Judge is satisfied does it produce the final country + coordinates.

This is the approach as it ran on bwUniCluster 3 (H100 80GB). The
results from that cluster run are included under `results/` (per-image
outputs) and under `../results-overview/VLM Hub and Spoke/` (consolidated
results + full LLM-as-Judge evaluation).

---

## 1. Approach

### Five specialists (running in parallel)

| Agent | Focus |
|---|---|
| `linguistic`  | Script, language, signage |
| `landscape`   | Topography, climate, vegetation, geology |
| `botanics`    | Plants, endemic species |
| `regulatory`  | Road signs, license plates, infrastructure standards |
| `meta`        | Camera artefacts, image style, Google Street View signatures |

Each agent returns a list of **country candidates** with confidence and
supporting evidence.

### Hub-and-Spoke pipeline (LangGraph)

```
prepare_image
    -> [5 agents: initial assessment]  (parallel)
    -> judge_review  <---+
         |               |
         | (finalize)    | (question)
         v               |
    judge_final          v
         |          discussion  (per targeted agent, in parallel)
         v               |
        END              +---> back to judge_review
```

- **Round 0 (initial assessment):** every agent sees only the image and
  its own domain prompt. No cross-agent chatter, no anchoring. Round 0
  is the same clean-slate step as in other approaches.
- **judge_review:** the Judge sees all five Round-0 assessments and
  decides either
  - **`finalize`** if the picture is clear (majority consensus, all
    high-confidence evidence pointing the same way), or
  - **`questions`** with a list of targeted follow-ups directed at
    specific agents ("Regulatory: you said the plate looks Turkish,
    but Linguistic sees Cyrillic; can you re-examine the plate?"). The
    Judge can ask up to `VLM_MAX_DISCUSSION_ROUNDS` rounds of questions
    before it is forced to finalize.
- **discussion:** each targeted agent re-examines the image with the
  Judge's specific question in mind and produces a refined assessment.
  The dialogue trace is preserved verbatim in `discussion_log`.
- **judge_final:** the Judge synthesises the final country +
  coordinates + reasoning from all traces (Round 0 + every discussion
  round).

The dialogue is **variable-length**: on 22 % of the 500 benchmark
images the Judge finalised immediately (0 discussion rounds); on 20 %
it used all 3 allowed rounds. The mean is 1.3 rounds.

### End-of-run outputs

- `assessments`: the Round-0 outputs of all 5 agents.
- `discussion_log`: full ordered list of Judge questions and agent
  responses (target_agent, judge_question, agent_response, round_number
  per entry).
- `discussion_rounds`: how many rounds the Judge used before
  finalising.
- `country_result`: raw judge text (verbatim, for auditing).
- `coordinates`: structured `{"lat", "lng"}` or `null` (see Section 6).
- `final_reasoning`: the judge's `<think>` chain, when the model
  emitted one.
- `error`: optional; set only if the judge failed hard (timeout /
  exception). No fake `(0, 0)` fallback is emitted.

---

## 2. Directory layout

```
.
├── vlm_council/               # Python package (pipeline + agents)
│   ├── agents/                # 5 specialists + judge
│   ├── graph.py               # LangGraph pipeline (Hub-and-Spoke)
│   ├── batch.py               # Batch processor with resume + --file-list
│   ├── config.py              # Env-var-based configuration
│   ├── state.py               # LangGraph state (assessments, discussion, final)
│   ├── coordinates.py         # Shared judge-output coordinate parser
│   ├── llm.py                 # vLLM / OpenAI-compatible client
│   ├── image_utils.py         # base64 encoding for the VLM
│   ├── run.py                 # Single-image CLI
│   └── evaluate.py            # Stage-1 evaluation (country accuracy + distance to GT)
├── eval_hubspoke/             # LLM-as-a-Judge evaluation (Stage 2, Qwen 3.6 as judge)
│   └── cluster_results/       # Cluster-run eval outputs (report.md/pdf, plots, judge/)
├── slurm/                     # bwUniCluster 3 launch scripts
│   ├── run_council_uc3.sh     # Single job: starts vLLM + batch
│   └── launch_short_uc3.sh    # Splits Images/ into parallel jobs
├── scripts/
│   └── download_dataset.py    # Local downloader (identical across approaches, run once)
├── results/                   # Per-image result.json (500 images from the cluster run)
├── slurm/setup_uc3.sh               # One-off setup on uc3
├── pyproject.toml
└── requirements.txt
```

---

## 3. Setup

### Preparing the dataset (once, then reused across all approaches)

**You only ever need to do this once, across the entire multi-approach
project.** The GeoRC benchmark (~3-4 GB of PNGs plus the
`georc_locations.csv` ground truth) is downloaded locally, rsynced to
the shared `datasets` workspace on the cluster, and from then on
**every approach reuses the same dataset via symlinks**. There is no
need to re-download or re-upload the images for the Hub-and-Spoke,
Progressive Narrowing, Tournament, or any future approach: each
approach workspace just points at the same `datasets` workspace with
`ln -s`.

The download script `scripts/download_dataset.py` is bundled with
every approach for self-containedness, but the three approaches ship
**identical copies** of it. Running it once from any one approach
folder is enough. **If the dataset is already present in the shared
`datasets` workspace (because another approach has already prepared
it), skip this section entirely** and go straight to *Recommended
workspace layout* below.

```bash
# On your local machine, one-time only (from ANY approach folder)
mkdir -p dataset
python3 scripts/download_dataset.py --output-dir dataset/Images --workers 16
cp /path/to/georc_locations.csv dataset/

# Rsync into the shared datasets workspace (one-time only)
rsync -avh --progress dataset/ <user>@uc3.scc.kit.edu:$(ssh <user>@uc3.scc.kit.edu 'ws_find datasets')/
```

After this one-time upload, the shared `datasets` workspace on the
cluster contains `Images/` and `georc_locations.csv`. Any approach
workspace can then symlink them in (see next section), avoiding
per-approach duplication of the multi-gigabyte image set.

### Recommended workspace layout on bwUniCluster 3

| Workspace | Purpose | Shared? |
|---|---|---|
| `vlm-council-hs` | Repo checkout + `results/` + `.venv` for this approach | No |
| `hf-cache` | HuggingFace model cache | Yes, reused across approaches |
| `datasets` | Images + `georc_locations.csv` | Yes, reused across approaches |

### On bwUniCluster 3 (H100 80GB)

```bash
# On the login node, allocate the three workspaces once
ws_allocate datasets 60
ws_allocate hf-cache 60
ws_allocate vlm-council-hs 30

cd $(ws_find vlm-council-hs)

# Upload the project files into the workspace, then symlink the dataset
ln -s "$(ws_find datasets)/Images" Images
ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv

bash slurm/setup_uc3.sh
```

What `slurm/setup_uc3.sh` does:
- Loads `devel/python/3.13.1` + `devel/cuda/12.8`
- Creates `.venv/` and installs `vllm>=0.8`, `langchain-core`,
  `langchain-openai`, `langgraph`
- Points the HuggingFace cache at the `hf-cache` workspace

---

## 4. Running the pipeline

### Cluster: single job

```bash
cd $(ws_find vlm-council-hs)
sbatch slurm/run_council_uc3.sh
```

Resume-capable: image folders that already contain a valid
`result.json` are skipped.

### Cluster: parallel jobs

```bash
# Default (Gemma 4), 5 jobs x 10 images each
bash slurm/launch_short_uc3.sh 5 10

# Judge thinking + larger context
VLM_JUDGE_THINKING=true VLM_OUTPUT_DIR=results_gemma4_thinking VLM_MAX_MODEL_LEN=16384 \
    bash slurm/launch_short_uc3.sh 5 5
```

The launcher scans `results/*/result.json`, finds unprocessed images
and splits them across `NUM_JOBS` sbatch submissions via `--file-list`.

---

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VLM_MODEL` | `google/gemma-4-31b-it` | Vision model (all 5 agents + judge) |
| `VLM_JUDGE_MODEL` | = `VLM_MODEL` | Judge model (optional separate model) |
| `VLM_JUDGE_THINKING` | `false` | Judge in thinking mode |
| `VLM_MAX_DISCUSSION_ROUNDS` | `3` | Cap on judge follow-up rounds |
| `VLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLM_MAX_MODEL_LEN` | `8192` | Max context length |
| `VLM_GPU_MEMORY_UTIL` | `0.9` | vLLM GPU utilisation |
| `VLM_CALL_TIMEOUT` | `600` | Per-VLM-call timeout (seconds) |
| `VLM_OUTPUT_DIR` | `results` | Output directory |

Tested model (council): `google/gemma-4-31b-it`
Tested judge (eval Stage 2): `Qwen/Qwen3.6-35B-A3B-FP8`

---

## 6. Output format

For every image `<output_dir>/<image-stem>/result.json` is written:

```json
{
  "image_path": "...",
  "model": "google/gemma-4-31b-it",
  "judge_model": "google/gemma-4-31b-it",
  "timing": { "total_seconds": 82.3 },
  "assessments": {
    "linguistic":  { "candidates": [...], "evidence": [...] },
    "landscape":   { ... },
    "botanics":    { ... },
    "regulatory":  { ... },
    "meta":        { ... }
  },
  "discussion_log": [
    {
      "round_number": 1,
      "judge_question": "You said the sign is Cyrillic ...",
      "target_agent": "linguistic",
      "agent_response": "..."
    },
    ...
  ],
  "discussion_rounds": 2,
  "country_result": "Country: Portugal\nCoordinates: 39.5, -8.0\nReasoning: ...",
  "coordinates": { "lat": 39.5, "lng": -8.0 },
  "final_reasoning": "..."
}
```

### `coordinates` field

The judge emits its final answer as free text of the form `Country:
<name>\nCoordinates: <lat>, <lon>\nReasoning: <...>` which is
preserved verbatim in `country_result`. The graph additionally parses
the coordinates into a structured object and writes them to
`coordinates`:

- **Success:** `coordinates` is `{"lat": <float>, "lng": <float>}`,
  values in valid geographic range.
- **Judge failure or unparseable output:** `coordinates` is `null`,
  and, if the judge failed hard (timeout / exception), a top-level
  `"error"` field is added. We deliberately do **not** substitute a
  `(0, 0)` fallback: that would silently poison downstream distance
  metrics with ~15 000 km outliers.

---

## 7. Results

The 500-image benchmark ran on Gemma 4 31B IT. Full evaluation
artefacts are stored in two locations for convenience:

- `eval_hubspoke/cluster_results/` (inside this approach folder)
- `../results-overview/VLM Hub and Spoke/evaluation/` (consolidated across approaches)

Both trees contain the same content:

- `council_run/` (only under `../results-overview/`): 500 per-image folders,
  each with `result.json` (initial assessments, full judge dialogue,
  final country + coordinates)
- Evaluation artefacts:
  - `report.md`, `report.html`, `report.pdf`: full evaluation report
  - `report_compact.html`, `report_compact.pdf`: condensed version
  - `agent_metrics.json`, `geo_metrics.json`, `judge_summary.json`:
    machine-readable metrics
  - `plots/`: error distribution, confusion matrix, agent scores,
    discussion analysis, ...
  - `judge/`: per-image LLM-as-judge verdicts (500 JSONs)

**Headline numbers (500 images, Hub-and-Spoke, Gemma 4 31B IT, judged
by Qwen 3.6):**

| Metric | Value |
|---|---|
| Country accuracy | 64.6 % (323 / 500) |
| Median haversine error | 435 km |
| Mean haversine error   | 1 598 km |
| Mean discussion rounds | 1.3 (of max 3) |
| Runs finalised immediately (0 rounds) | 22 % |
| Runs using all 3 rounds | 20 % |

Re-run the Stage 1 evaluation on new results with:

```bash
python -m vlm_council.evaluate results/ georc_locations.csv
```

---

## 8. Dependencies

See `pyproject.toml` and `requirements.txt`. Core stack:
- Python 3.11+
- vLLM >= 0.8 (CUDA 12.6/12.8 wheel)
- LangGraph + LangChain-OpenAI
- Pillow (image encoding)
