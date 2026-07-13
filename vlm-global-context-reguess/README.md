# VLM Global Context Re-guess

> **Results Overview:** [https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Global_Context_Reguess/report.html](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Global_Context_Reguess/report.html)  
> GT-based statistics and the LLM-as-Judge (Qwen 3.6) report for this approach, rendered as a GitHub Page.

A multi-agent council of five specialised Vision-Language Model agents
that collaboratively geolocate images. The council follows a
**Global Context Re-guess** topology with two synchronised rounds: every
agent first assesses the image independently in Round 1, then in Round 2
every agent sees **all five Round-1 outputs** and re-assesses. A Judge
finally synthesises the country + coordinates from every trace.

This is the approach as it ran on bwUniCluster 3 (H100 80GB). The
results from that cluster run are included under `results/` (per-image
outputs) and under `../results-overview/VLM Global Context Reguess/`
(consolidated results + full LLM-as-Judge evaluation).

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

### Global Context Re-guess pipeline (LangGraph)

```
prepare_image
    -> [5 agents Round 1: independent assessment]  (parallel)
    -> round1_complete  (barrier: collects all Round-1 outputs)
    -> [5 agents Round 2: re-assessment with global context]  (parallel)
    -> judge_final -> END
```

- **Round 1** is a clean-slate assessment: every agent sees only the
  image and its own domain expertise. No cross-agent chatter, no
  anchoring.
- **`round1_complete`** is a barrier node: it fires only once every
  Round-1 output is in place, no LLM call.
- **Round 2** is the "re-guess" step: every agent receives all five
  Round-1 assessments (its own included) and produces a refined answer.
  An agent that initially said "high confidence: Argentina" may see the
  botanics agent's crop evidence pointing to Uruguay and update
  accordingly.
- **Judge** receives every trace (Round 1 + Round 2 from all five
  agents) and emits the final country + coordinates + reasoning.

### Phase-by-phase walkthrough

Every box in the diagram maps to a node in `vlm_council/graph.py`.

1. **`prepare_image`** reads the image from disk, base64-encodes it once
   and stores it in the LangGraph state.
2. **`round1_<name>`** (5 nodes, parallel) applies each agent's
   expertise to the image in isolation. Result: an
   `AgentAssessment(candidates=[...], evidence=[...])`.
3. **`round1_complete`** is a no-op barrier. Downstream Round 2 nodes
   only fire once every Round 1 assessment is complete.
4. **`round2_<name>`** (5 nodes, parallel) applies the same domain
   expertise, but the prompt also carries all five Round-1 assessments.
   Agents can either stand their ground or revise.
5. **`judge_final`** receives all ten assessments plus the image and
   produces free text of the form
   `Country: <name>\nCoordinates: <lat>, <lon>\nReasoning: <...>`. The
   node then parses that text and writes `country_result` (raw text)
   and `coordinates` (`{"lat": float, "lng": float}` or `null`) into
   the state.

### End-of-run outputs

- `round_1_assessments`, `round_2_assessments`: per-agent
  `AgentAssessment` for both rounds.
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
│   ├── graph.py               # LangGraph pipeline (Global Context Re-guess)
│   ├── batch.py               # Batch processor with resume + --file-list
│   ├── config.py              # Env-var-based configuration
│   ├── state.py               # LangGraph state (round 1, round 2, final)
│   ├── coordinates.py         # Shared judge-output coordinate parser
│   ├── llm.py                 # vLLM / OpenAI-compatible client
│   ├── image_utils.py         # base64 encoding for the VLM
│   ├── run.py                 # Single-image CLI
│   ├── evaluate.py            # Stage-1 evaluation (country accuracy + shift analysis)
│   └── analyze_rounds.py      # Round-1 vs Round-2 shift analytics
├── eval_reguess/              # LLM-as-a-Judge evaluation (Stage 2, Qwen 3.6 as judge)
│   └── cluster_results/       # Cluster-run eval outputs (report.md/html, plots, judge/)
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
need to re-download or re-upload the images for the Global Context
Re-guess, Hub-and-Spoke, Progressive Narrowing, or any future
approach: each approach workspace just points at the same `datasets`
workspace with `ln -s`.

The download script `scripts/download_dataset.py` is bundled with
every approach for self-containedness, but the approaches ship
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

### Recommended workspace layout on bwUniCluster 3

| Workspace | Purpose | Shared? |
|---|---|---|
| `vlm-council-gcr` | Repo checkout + `results/` + `.venv` for this approach | No |
| `hf-cache` | HuggingFace model cache | Yes, reused across approaches |
| `datasets` | Images + `georc_locations.csv` | Yes, reused across approaches |

### On bwUniCluster 3 (H100 80GB)

```bash
# On the login node, allocate the three workspaces once
ws_allocate datasets 60
ws_allocate hf-cache 60
ws_allocate vlm-council-gcr 30

cd $(ws_find vlm-council-gcr)

# Upload the project files into the workspace, then symlink the dataset
ln -s "$(ws_find datasets)/Images" Images
ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv

bash slurm/setup_uc3.sh
```

---

## 4. Running the pipeline

### Cluster: single job

```bash
cd $(ws_find vlm-council-gcr)
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
| `VLM_JUDGE_THINKING` | `true` | Judge in thinking mode |
| `VLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLM_MAX_MODEL_LEN` | `16384` | Max context length |
| `VLM_GPU_MEMORY_UTIL` | `0.9` | vLLM GPU utilisation |
| `VLM_IMAGE_TOKEN_BUDGET` | `1120` | Per-image token budget for vLLM multi-modal |
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
  "judge_thinking": true,
  "image_token_budget": 1120,
  "timing": { "total_seconds": 82.3 },
  "round_1_assessments": {
    "linguistic":  { "agent_name": "linguistic", "candidates": [...], "evidence": [...] },
    "landscape":   { ... },
    "botanics":    { ... },
    "regulatory":  { ... },
    "meta":        { ... }
  },
  "round_2_assessments": {
    "linguistic":  { "agent_name": "linguistic", "candidates": [...], "evidence": [...] },
    ...
  },
  "country_result": "Country: Portugal\nCoordinates: 39.5, -8.0\nReasoning: ...",
  "coordinates": { "lat": 39.5, "lng": -8.0 },
  "final_reasoning": "..."
}
```

### `coordinates` field

The judge emits its final answer as free text of the form `Country:
<name>\nCoordinates: <lat>, <lon>\nReasoning: <...>` which is
preserved verbatim in `country_result`. The graph additionally parses
the coordinates into a structured object:

- **Success:** `coordinates` is `{"lat": <float>, "lng": <float>}`,
  values in valid geographic range.
- **Judge failure or unparseable output:** `coordinates` is `null`,
  and, if the judge failed hard (timeout / exception), a top-level
  `"error"` field is added. We deliberately do **not** substitute a
  `(0, 0)` fallback: that would silently poison downstream distance
  metrics with ~15 000 km outliers.

---

## 7. Results

The 500-image benchmark ran on Gemma 4 31B IT with judge thinking
enabled. Full evaluation artefacts are stored in two locations for
convenience:

- `eval_reguess/cluster_results/` (inside this approach folder)
- `../results-overview/VLM Global Context Reguess/evaluation/` (consolidated
  across approaches)

Both trees contain the same LLM-as-Judge (Qwen 3.6) evaluation
artefacts:

- `report.md`, `report.html`: evaluation report (judge synthesis
  quality, per-agent Round 2 improvement, hallucination scores)
- `judge_summary.json`: machine-readable metrics
- `plots/`: role adherence, hallucination, Round 2 improvement,
  judge synthesis distribution
- `judge/`: per-image LLM-as-judge verdicts (500 JSONs)

The council-run outputs (500 `result.json` files) are mirrored under
`../results-overview/VLM Global Context Reguess/council_run/`.

**Headline numbers (500 images, Global Context Re-guess, Gemma 4 31B
IT):**

| Metric | Value |
|---|---|
| Country accuracy | 65.0 % (325 / 500) |
| Median haversine error | 410 km |
| Mean haversine error   | 1 546 km |
| Judge synthesis quality (Qwen 3.6) | 0.683 mean, 0.750 median |
| Mean Round-2 improvement (Qwen 3.6) | 0.509 (1 = genuine synthesis, 0 = rubber-stamp) |

**Round-1 -> Round-2 top-prediction shift** per agent (how often each
agent changed its top country after seeing the global context):

| Agent | Shift rate |
|---|---|
| linguistic | 10.3 % |
| landscape | 8.8 % |
| botanics | 11.6 % |
| regulatory | 15.6 % |
| meta | 9.4 % |

Re-run the Stage 1 evaluation with:

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
