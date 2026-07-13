# VLM Progressive Narrowing with Parallel Hypotheses

> **Results Overview:** [https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Progressive_Narrowing_with_Parallel_Hypotheses/report.html](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Progressive_Narrowing_with_Parallel_Hypotheses/report.html)  
> GT-based statistics and the LLM-as-Judge (Qwen 3.6) report for this approach, rendered as a GitHub Page.

A multi-agent council of five specialised Vision-Language Model agents that
collaboratively geolocate images. The approach tackles the problem in two
stages, first **Region**, then **Country**, and in each stage lets all agents
independently evaluate several **parallel hypotheses** before a Judge
synthesises the final answer.

This is the approach as it ran on bwUniCluster 3 (H100 80GB). The results
from that cluster run are included under `results/`.

---

## 1. Approach

### Five specialists (running in parallel)

| Agent | Focus |
| --- | --- |
| `linguistic` | Script, language, signage |
| `landscape` | Topography, climate, vegetation, geology |
| `botanics` | Plants, endemic species |
| `regulatory` | Road signs, license plates, infrastructure standards |
| `meta` | Camera artefacts, image style, Google Street View signatures |

Each agent returns a list of **country candidates** with confidence and
supporting evidence.

### Progressive Narrowing (LangGraph pipeline)

```
prepare_image
    -> [5x specialists: initial assess]  (parallel)
    -> region_consensus_check
        ├── Path A (consensus):
        │       -> country_hypotheses
        │       -> [5x evaluate]  (parallel)
        │       -> country_decision -> END
        │
        └── Path B (no consensus):
                -> region_hypotheses
                -> [5x evaluate]  (parallel)
                -> region_decision
                -> [5x constrained assess]  (parallel)
                -> country_hypotheses
                -> [5x evaluate]  (parallel)
                -> country_decision -> END
```

- **Region consensus check**: if enough agents propose the same region, the
  region is considered confirmed and the pipeline jumps straight to the
  country stage (Path A). Otherwise the region itself is first evaluated
  as a hypothesis set (Path B).
- **Parallel hypotheses**: in each evaluation stage the Judge proposes up to
  `max_region_hypotheses` / `max_country_hypotheses` candidates, and all five
  agents evaluate **all** hypotheses in parallel. The order is shuffled
  per agent (`random`) to avoid position bias.
- **Judge aggregation**: the Judge condenses the parallel evaluations and
  picks the final answer (country + coordinates + reasoning).

### Phase-by-phase walkthrough

Every box in the diagram maps to a node in `vlm_council/graph.py`. Below is
what each node actually does, what state it produces, and (where relevant)
which prompt drives it.

**Shared prelude (always runs)**

1. **`prepare_image`** reads the image from disk, base64-encodes it once, and
   stores it in the LangGraph state (`image_b64`, `image_mime`). This
   guarantees that every downstream agent receives the exact same bytes,
   without repeated disk I/O.
2. **`[5x specialists: initial assess]`** dispatches the same image to all
   five agents in parallel (`asyncio.gather`, backed by vLLM continuous
   batching). Each agent applies **only its own expertise** and returns
   `AgentAssessment(candidates=[{country, confidence, reasoning}, ...],
   evidence=[...])`. The confidence label is one of
   `high | medium | low | speculative`. Agents are actively encouraged to
   assign the same confidence to multiple plausible countries instead of
   inventing an artificial ranking.
3. **`region_consensus_check`** is the first Judge call. The Judge maps
   every candidate from every agent to a world region (`Europe`,
   `East Asia`, `South America`, ...) and returns a `region_candidates`
   object of the form `{region: {country: agent_count}}`. Consensus is
   **strict**: even a single candidate outside the majority region breaks
   consensus. The output routes the graph into Path A or Path B.

**Path A: region consensus (fast path)**

4. **`country_hypotheses` (Path A)** aggregates every country any agent
   proposed initially and ranks them by how many agents mentioned each
   one. Any country that 3+ agents named is always kept (a "strong local
   consensus" safety net). The top `max_country_hypotheses` become
   `Hypothesis` objects with an id like `country_germany`.
5. **`[5x evaluate]` (country)** hands the same hypothesis list back to
   every specialist, but this time as a scoring task: each agent rates
   every hypothesis on the five-level confidence scale
   `strongly_support (+2) / support (+1) / neutral (0) / contradicts (-1) /
   strongly_contradicts (-2)`. The order of the hypotheses is
   **independently shuffled per agent** so that positional bias
   (primacy/recency) doesn't correlate across the council.
6. **`country_decision`** is the final Judge call. It receives the raw
   assessments *plus* all 5 x N per-hypothesis evaluations, applies weighted
   aggregation with heavy penalties for `strongly_contradicts` votes backed
   by hard evidence, and emits free text of the exact form
   `Country: <name>\nCoordinates: <lat>, <lon>\nReasoning: <...>`. The
   `country_decision_node` then parses that text and writes
   `country_result` (raw text) and `coordinates`
   (`{"lat": float, "lng": float}` or `null`) into the state.

**Path B: no consensus (full path)**

7. **`region_hypotheses`** turns the Judge's `proposed_regions` list (from
   step 3) into up to `max_region_hypotheses` `Hypothesis` objects at the
   region level (e.g. `region_europe`, `region_central_asia`).
8. **`[5x evaluate]` (region)** works identically to step 5 but on
   *region* hypotheses. Each agent scores every region on the same
   five-level scale, using an independently shuffled order.
9. **`region_decision`** is the Judge call that picks the winning region.
   Its rules: sum the weighted confidence scores per region, penalise any
   region that received a `strongly_contradicts` with hard physical
   evidence (e.g. wrong hemisphere), and break ties by the number of
   `strongly_support` votes. On success, `confirmed_region` is set.
10. **`[5x constrained assess]`** re-runs the initial assessment, but this
    time with a hard constraint in the prompt: "This image has been
    confirmed to be from <region>. You MUST only propose countries within
    <region>." This forces each specialist to reason within the pruned
    search space and produces a second, region-consistent set of
    assessments (`<agent>_country_assessment`).
11. **`country_hypotheses` (Path B)** aggregates from those constrained
    assessments. The 3+ agent safety net from step 4 still applies.
    Result: up to `max_country_hypotheses` `Hypothesis` objects.
12. **`[5x evaluate]` (country)** and **`country_decision`** are then
    identical to steps 5 and 6, producing the final answer.

**End-of-run outputs (in every path)**

- `country_result`: raw judge text (preserved verbatim for downstream
  auditing).
- `coordinates`: structured `{"lat", "lng"}` or `null` (see Section 6).
- `final_reasoning`: the judge's `<think>` chain, when the model emitted
  one.
- `error`: optional; set only if the country-decision Judge call failed
  hard (timeout / exception). No fake `(0, 0)` fallback is emitted.

Each numbered step is a separate LangGraph node and can be inspected in
isolation via `graph.py` and the corresponding agent module.

### Why "Parallel Hypotheses"?

Instead of a single best-guess chain, the council evaluates several plausible
hypotheses simultaneously. This reduces anchoring bias and gives the Judge a
better-calibrated confidence signal.

---

## 2. Directory layout

```
.
├── vlm_council/               # Python package (pipeline + agents)
│   ├── agents/                # 5 specialists + judge
│   ├── graph.py               # LangGraph pipeline (Progressive Narrowing)
│   ├── batch.py               # Batch processor with resume + --file-list
│   ├── config.py              # Env-var-based configuration
│   ├── state.py               # LangGraph state (assessments, hypotheses, ...)
│   ├── coordinates.py         # Shared judge-output coordinate parser
│   ├── llm.py                 # vLLM / OpenAI-compatible client
│   ├── image_utils.py         # base64 encoding for the VLM
│   ├── run.py                 # Single-image CLI
│   └── evaluate.py            # Result evaluation (country accuracy + distance to GT)
├── slurm/                     # bwUniCluster 3 launch scripts
│   ├── run_council_uc3.sh     # Single job: starts vLLM + batch
│   ├── launch_short_uc3.sh    # Splits Images/ into parallel jobs
│   ├── run_eval_uc3.sh        # Evaluation job (LLM-as-judge Stage 2)
│   └── launch_eval_short_uc3.sh
├── eval/                      # Evaluation scripts (Stage 1 + Stage 2)
│   └── cluster_results/       # Cluster-run eval outputs (report.md/pdf, plots, judge/)
├── scripts/
│   └── download_dataset.py    # Local downloader for the HuggingFace image set
├── tests/                     # Regression tests (stdlib only, 35 tests)
├── results/                   # Per-image result.json (500 images from the cluster run)
├── slurm/setup_uc3.sh               # One-off setup on uc3
├── pyproject.toml
└── requirements.txt
```

---

## 3. Setup

### Preparing the dataset (once, on your local machine)

The GeoRC benchmark ships as ~3-4 GB of PNGs on HuggingFace. Because
cluster compute nodes do not have outbound internet access, and because
you want to spend GPU time on inference rather than downloads, we
**prepare the dataset locally** and then rsync it to the shared `datasets`
workspace on the cluster.

```bash
# On your laptop / a machine with internet
mkdir -p dataset
python3 scripts/download_dataset.py --output-dir dataset/Images --workers 16

# Drop your ground-truth CSV next to the images
cp /path/to/georc_locations.csv dataset/

# Verify: dataset/ now contains
#   Images/*.png              (500 files, ~3.5 GB)
#   georc_locations.csv       (500 rows: filename,country_code,lat,lng)
```

The download script only uses the Python standard library and is
resume-capable, so if a run gets interrupted you can simply invoke it
again.

Then sync to the cluster:

```bash
# From your local machine
rsync -avh --progress dataset/ <user>@uc3.scc.kit.edu:$(ssh <user>@uc3.scc.kit.edu 'ws_find datasets')/
```

After that, the shared `datasets` workspace on the cluster contains both
`Images/` and `georc_locations.csv` and every approach can symlink them
in (see below).

### Recommended workspace layout on bwUniCluster 3

We recommend keeping three separate workspaces so that expensive artefacts
are downloaded and stored **once** and shared across every approach:

| Workspace | Purpose | Shared across approaches? |
| --- | --- | --- |
| `vlm-council-pn` (or one per approach) | Repo checkout + `results/` + `.venv` | No, one per approach |
| `hf-cache` | HuggingFace model cache (`HF_HOME`), the VLM checkpoints go here | **Yes**, reused by every approach |
| `datasets` | The image benchmark(s), e.g. `Images/`, `georc_locations.csv` | **Yes**, reused by every approach |

Rationale:

- **Approach-specific workspace** keeps every experiment's code, venv, and
  results isolated so runs don't collide and each approach can be evaluated
  independently.
- **Shared HuggingFace workspace** avoids re-downloading 30-60 GB models per
  experiment. Every approach sets `HF_HOME` to this workspace via
  `slurm/setup_uc3.sh`.
- **Shared datasets workspace** holds the input images once; each approach
  workspace **symlinks** the dataset in rather than copying it:

```bash
  # From inside the approach workspace, once after allocating it:
  ln -s "$(ws_find datasets)/Images" Images
  ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv
```

  This keeps the multi-gigabyte image set (roughly 3-4 GB for the 500-image
  benchmark, at ~7 MB per PNG) out of every approach workspace while
  making the paths look local (`Images/` from the batch CLI still works).

### On bwUniCluster 3 (H100 80GB)

```bash
# On the login node, allocate the three workspaces once
ws_allocate datasets 60          # shared, holds Images/ + ground truth
ws_allocate hf-cache 60          # shared, holds HuggingFace model cache
ws_allocate vlm-council-pn 30    # one per approach, this one is for PN + PH

cd $(ws_find vlm-council-pn)

# Upload the project files into the workspace, then symlink the dataset
# (the `datasets` workspace already contains the rsynced Images/ and CSV,
# see "Preparing the dataset" above):
ln -s "$(ws_find datasets)/Images" Images
ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv

# Finally, run the one-off setup:
bash slurm/setup_uc3.sh
```

What `slurm/setup_uc3.sh` does:
- Loads modules `devel/python/3.13.1` + `devel/cuda/12.8`
- Creates `.venv/` and installs `vllm>=0.8`, `langchain-core`,
  `langchain-openai`, `langgraph`
- Points the HuggingFace cache at the `hf-cache` workspace (falls back to
  a local `.cache/` if `hf-cache` doesn't exist, so you can also run without
  a shared cache)

---

## 4. Running the pipeline

### Cluster: single job

```bash
cd $(ws_find vlm-council-pn)
sbatch slurm/run_council_uc3.sh
```

The job starts the vLLM server, waits for `/health`, then invokes
`python -m vlm_council.batch Images/ results/` and cleans up at the end.

Resume: images that already have a valid `result.json` are skipped, so the
same job can be re-submitted after a failure without redoing completed
images.

### Cluster: parallel jobs (recommended)

```bash
# 5 jobs × 10 images on gpu_h100_short
bash slurm/launch_short_uc3.sh 5 10

# Judge thinking + larger context, into a separate output directory
VLM_JUDGE_THINKING=true \
VLM_OUTPUT_DIR=results_gemma4_thinking \
VLM_MAX_MODEL_LEN=16384 \
    bash slurm/launch_short_uc3.sh 5 5
```

The launcher scans `results/*/result.json`, finds unprocessed images and
splits them across `NUM_JOBS` sbatch submissions via `--file-list`.

---

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLM_MODEL` | `google/gemma-4-31b-it` | Vision model |
| `VLM_JUDGE_MODEL` | = `VLM_MODEL` | Judge model (optional separate) |
| `VLM_JUDGE_THINKING` | `false` | Judge in thinking mode |
| `VLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLM_MAX_MODEL_LEN` | `8192` | Max context length |
| `VLM_GPU_MEMORY_UTIL` | `0.9` | vLLM GPU utilisation |
| `VLM_MAX_REGION_HYPOTHESES` | `4` | Parallel region hypotheses |
| `VLM_MAX_COUNTRY_HYPOTHESES` | `5` | Parallel country hypotheses |
| `VLM_CALL_TIMEOUT` | `600` | Per-VLM-call timeout (seconds) |
| `VLM_MM_PROCESSOR_KWARGS` | `{"max_soft_tokens": 1120}` | vLLM multi-modal args |
| `VLM_OUTPUT_DIR` | `results` | Output directory |

Tested model:
- `google/gemma-4-31b-it` (default)

---

## 6. Output format

For every image `<output_dir>/<image-stem>/result.json` is written:

```json
{
  "image_path": "...",
  "model": "google/gemma-4-31b-it",
  "judge_model": "...",
  "timing": { "total_seconds": 82.3 },
  "assessments": {
    "linguistic":  { "candidates": [...], "evidence": [...] },
    "landscape":   { ... },
    "botanics":    { ... },
    "regulatory":  { ... },
    "meta":        { ... }
  },
  "progressive_narrowing": {
    "region_consensus": true,
    "confirmed_region": "Southern Europe",
    "proposed_regions": [...],
    "region_candidates": { ... },
    "region_decision_reasoning": "...",
    "path": "A"
  },
  "country_assessments": { ... },
  "hypothesis_evaluations": [ ... ],
  "country_result": "Country: Portugal\nCoordinates: 39.5, -8.0\nReasoning: ...",
  "coordinates": { "lat": 39.5, "lng": -8.0 },
  "final_reasoning": "..."
}
```

`path: "A"` = region consensus found directly, `path: "B"` = an additional
region-hypothesis round was required.

### `coordinates` field

The judge emits its final answer as free text of the form
`Country: <name>\nCoordinates: <lat>, <lon>\nReasoning: <...>` which is
preserved verbatim in `country_result`. The graph additionally parses the
coordinates into a structured object and writes them to `coordinates`:

- **Success:** `coordinates` is an object `{"lat": <float>, "lng": <float>}`
  where both values are in valid geographic range (lat ∈ [-90, 90],
  lng ∈ [-180, 180]).
- **Judge failure or unparseable output:** `coordinates` is `null`, and, if
  the judge failed hard (timeout / exception), a top-level `"error"` field
  is added describing what went wrong. We deliberately do **not** substitute
  a `(0, 0)` fallback, because that would silently poison downstream
  distance metrics with ~15 000 km outliers.

`error` shape when present:

```json
{
  "error": "TimeoutError: judge exceeded 600s",
  "country_result": "",
  "coordinates": null,
  ...
}
```

Downstream tooling (`vlm_council/evaluate.py`, `eval/loader.py`) skips runs
with a non-empty `error`.

### Legacy result files

Runs recorded before the structured-coordinates change stored
`coordinates` either as an empty string `""` (the graph never populated it)
or as the string `"lat, lng"`. Both forms are still accepted by
`eval/loader.py`, so old and new results can be evaluated together
without migration. See `vlm_council/coordinates.py` for the shared parser.

---

## 7. Results

The cluster run of the 500-image benchmark with random-shuffled hypotheses is
included as follows:

- `results/`: 500 per-image folders, each with `result.json`
  (assessments, hypothesis evaluations, path A/B, final country + coordinates,
  reasoning)
- `eval/cluster_results/`: the evaluation of that run:
  - `report.md`, `report.html`, `report.pdf`: full evaluation report
  - `report_compact.html`, `report_compact.pdf`: condensed version
  - `agent_metrics.json`, `geo_metrics.json`, `funnel_metrics.json`,
    `heatmap_metrics.json`, `judge_summary.json`: machine-readable metrics
  - `plots/`: error distribution, bearing rose, confusion matrix, funnel,
    agent calibration/top-1, judge scores, PN path split, ...
  - `judge/`: per-image LLM-as-judge verdicts

**Headline numbers (500 images, PN + Parallel Hypotheses, Gemma 4 31B IT):**

| Metric | Value |
| --- | --- |
| Country accuracy | 64.0 % (320 / 500) |
| Median haversine error | 439 km |
| Mean haversine error | 1 549 km |
| Path A (region consensus) | 315 / 500 |
| Path B (no consensus) | 185 / 500 |

Re-run the evaluation on new results with `vlm_council/evaluate.py` or the
`eval/` package.

---

## 8. Dependencies

See `pyproject.toml` and `requirements.txt`. Core stack:
- Python 3.11+
- vLLM ≥ 0.8 (CUDA 12.6/12.8 wheel)
- LangGraph + LangChain-OpenAI
- Pillow (image encoding)
