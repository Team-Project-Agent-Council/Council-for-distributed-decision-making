# VLM Tournament Only

> **Results Overview:** [https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Tournament_Only/report.html](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Tournament_Only/report.html)  
> GT-based statistics and the LLM-as-Judge (Qwen 3.6) report for this approach, rendered as a GitHub Page.

A multi-agent council of five specialised Vision-Language Model agents
that geolocates images using a **single-path pipeline** that ends in a
head-to-head **tournament bracket**. Specialists propose candidates
directly from the initial image pass, and the resulting pool is
resolved with a single-elimination bracket.

This is the approach as it ran on bwUniCluster 3 (H100 80GB). The
results from that cluster run are included under `results/` (per-image
outputs) and under `../results-overview/VLM Tournament Only/` (consolidated
results + full LLM-as-Judge evaluation).

**Headline: 67.8 % top-1 country accuracy (339 / 500)**, the strongest
result across the aggregated approaches in this project.

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

### Pipeline (LangGraph)

```
prepare_image
    -> [5 agents: initial assessment]  (parallel)
    -> country_hypotheses     (candidate pool: top-K countries from the pool)
    -> country_evaluate       (5 agents re-score every candidate, see RAG refs)
    -> tournament_node        (bracket: pairwise judge matches with RAG references)
        -> final match emits Country + Coordinates + Reasoning
    -> END
```

This is a single-path pipeline: specialists assess the image, the
top-K countries form the candidate pool, and the tournament bracket
resolves the pool head-to-head.

### End-of-run outputs

- `assessments`: per-agent initial assessments.
- `hypothesis_evaluations`: 5-level confidence per candidate from the
  `country_evaluate` step.
- `candidate_pool`: top-K countries that enter the bracket.
- `rag_refs_seen`: which references were shown to the judge.
- `tournament_log`: full list of bracket matches with winner, both
  countries' reasoning, agreement status.
- `country_result`: raw judge text with `Tournament:` provenance.
- `coordinates`: structured `{"lat", "lng"}` or `null`.
- `final_reasoning`: last-match reasoning.
- `error`: optional; only on hard pipeline failure.

---

## 2. Directory layout

```
.
├── vlm_council/               # Python package (pipeline + agents + tournament)
│   ├── agents/                # 5 specialists + judge + tournament_judge
│   ├── graph.py               # LangGraph pipeline (Tournament Only)
│   ├── tournament.py          # Dynamic bracket + match runner
│   ├── rag/                   # RAG lookup for country reference images
│   ├── rag_toolbox.py         # Shared RAG utilities
│   ├── regions.py             # Country -> region taxonomy
│   ├── batch.py               # Batch processor with resume + --file-list
│   ├── config.py              # Env-var-based configuration
│   ├── state.py               # LangGraph state
│   ├── coordinates.py         # Shared judge-output coordinate parser
│   ├── llm.py                 # vLLM / OpenAI-compatible client
│   ├── image_utils.py         # base64 encoding for the VLM
│   ├── run.py                 # Single-image CLI
│   └── evaluate.py            # Stage-1 evaluation
├── eval_tourn/                # LLM-as-a-Judge evaluation (Stage 2, Qwen 3.6)
│   └── cluster_results/       # Cluster-run eval outputs (report.md/html, plots, judge/)
├── slurm/                     # bwUniCluster 3 launch scripts
├── scripts/
│   └── download_dataset.py    # Local downloader (identical across approaches, run once)
├── results/                   # Per-image result.json (500 images)
├── slurm/setup_uc3.sh
├── pyproject.toml
└── requirements.txt
```

---

## 3. Setup

### Preparing the dataset (once, then reused across all approaches)

**You only ever need to do this once, across the entire multi-approach
project.** The GeoRC benchmark (~3-4 GB of PNGs plus the
`georc_locations.csv` ground truth) is downloaded locally, rsynced to
the shared `datasets` workspace on the cluster, and from then on every
approach reuses the same dataset via symlinks.

```bash
# On your local machine, one-time only (from ANY approach folder)
mkdir -p dataset
python3 scripts/download_dataset.py --output-dir dataset/Images --workers 16
cp /path/to/georc_locations.csv dataset/

rsync -avh --progress dataset/ <user>@uc3.scc.kit.edu:$(ssh <user>@uc3.scc.kit.edu 'ws_find datasets')/
```

### Recommended workspace layout on bwUniCluster 3

| Workspace | Purpose | Shared? |
|---|---|---|
| `vlm-council-tourn` | Repo checkout + `results/` + `.venv` for this approach | No |
| `hf-cache` | HuggingFace model cache | Yes, reused across approaches |
| `datasets` | Images + `georc_locations.csv` | Yes, reused across approaches |

### On bwUniCluster 3 (H100 80GB)

```bash
ws_allocate datasets 60
ws_allocate hf-cache 60
ws_allocate vlm-council-tourn 30

cd $(ws_find vlm-council-tourn)

# Upload the project files into the workspace, then symlink the dataset
ln -s "$(ws_find datasets)/Images" Images
ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv

bash slurm/setup_uc3.sh
```

---

## 4. Running the pipeline

### Cluster: single job

```bash
cd $(ws_find vlm-council-tourn)
sbatch slurm/run_council_uc3.sh
```

### Cluster: chained batches

`slurm/chain_uc3.sh` submits council + eval jobs in
sequence.

---

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VLM_MODEL` | `google/gemma-4-31b-it` | Vision model (all 5 agents + judge + tournament_judge) |
| `VLM_JUDGE_MODEL` | = `VLM_MODEL` | Optional separate judge model |
| `VLM_JUDGE_THINKING` | `false` | Judge in thinking mode |
| `VLM_MAX_COUNTRY_HYPOTHESES` | `6` | Top-K country candidates in the pool |
| `VLM_TOURNAMENT_FINALISTS` | `4` | Bracket size cap (top-N by seed score) |
| `VLM_DATA_DIR` | (unset) | RAG data dir (plonkit references, bollards) |
| `VLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLM_MAX_MODEL_LEN` | `8192` | Max context length |
| `VLM_GPU_MEMORY_UTIL` | `0.9` | vLLM GPU utilisation |
| `VLM_CALL_TIMEOUT` | `600` | Per-VLM-call timeout (seconds) |
| `VLM_OUTPUT_DIR` | `results` | Output directory |

> **RAG data (`VLM_DATA_DIR`):** The RAG reference data extracted from GeoHints
> (references, bollards, etc.) is available here:
> [GeoHints RAG data](https://1drv.ms/f/c/6f8f2f993c06db24/IgBSOczD4WZjRb24uV_ipm64AS6pwNm8tHSPGBx-mnST52U?e=GtpToh).
> Download it and point `VLM_DATA_DIR` at the extracted directory.

Tested model (council): `google/gemma-4-31b-it`
Tested judge (eval Stage 2): `Qwen/Qwen3.6-35B-A3B-FP8`

---

## 6. Output format

For every image `<output_dir>/<image-stem>/result.json` is written. The
bracket is fed directly by the initial assessment + candidate pool:

```json
{
  "candidate_pool": ["Ukraine", "Russia", "Belarus", "Argentina"],
  "tournament_log": [
    {
      "round_label": "semi-1",
      "country_a": "Ukraine",
      "country_b": "Argentina",
      "pool_rank_a": 0,
      "pool_rank_b": 3,
      "winner": "Ukraine",
      "reasoning": "The deep, dark 'Chernozem' soil ...",
      "agreement": "agree"
    }
  ],
  "country_result": "Country: Ukraine\nCoordinates: 48.5, 33.5\nReasoning: ...\n\nTournament:\n  semi-1: Ukraine vs Argentina -> Ukraine\n  semi-2: Russia vs Belarus -> Russia\n  final: Ukraine vs Russia -> Ukraine",
  "coordinates": { "lat": 48.5, "lng": 33.5 }
}
```

### `coordinates` field

- **Success:** `{"lat": <float>, "lng": <float>}`, values in valid
  geographic range.
- **Walkover / no candidates / judge failure:** `null`. No `(0, 0)`
  fake fallback.

---

## 7. Results

The 500-image benchmark ran on Gemma 4 31B IT. Full evaluation
artefacts are stored in two locations for convenience:

- `eval_tourn/cluster_results/` (inside this approach folder)
- `../results-overview/VLM Tournament Only/evaluation/` (consolidated across
  approaches)

Both trees contain the same LLM-as-Judge (Qwen 3.6) evaluation
artefacts:

- `report.md`, `report.html`: evaluation report
- `judge_summary.json`: machine-readable metrics
- `plots/`
- `judge/`: 500 per-image LLM-as-judge verdicts

The council-run outputs (500 `result.json` files) are mirrored under
`../results-overview/VLM Tournament Only/council_run/`.

**Headline numbers (500 images, Tournament Only, Gemma 4 31B IT):**

| Metric | Value |
|---|---|
| Country accuracy | **67.8 % (339 / 500)** |
| Neighbor hits | 14.2 % (71 / 500) |
| Country or Neighbor | 82.0 % (410 / 500) |
| Median haversine error | 395 km |
| Mean haversine error | 1 412 km |

**Tournament bracket activation:**

| Matches per image | Images | Share |
|---|---|---|
| 3 (full bracket: 2 semis + final) | 500 | 100 % |

- **Total tournament matches across all 500 images:** 1 500 (mean 3.00
  per image).
- Every image runs the full top-4 bracket.

**Candidate-pool size** (input to the tournament):

| Candidates | Share |
|---|---|
| 4 | 3 % |
| 5 | 19 % |
| 6 | 77 % |

Almost all images carry a full pool of six candidates into the bracket.

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
- ChromaDB (RAG for country reference images)
