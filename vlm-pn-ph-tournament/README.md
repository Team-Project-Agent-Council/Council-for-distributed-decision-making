# VLM PN + PH + Tournament

> **Results Overview:** [https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_PN_plus_PH_plus_Tournament/report.html](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_PN_plus_PH_plus_Tournament/report.html)  
> GT-based statistics and the LLM-as-Judge (Qwen 3.6) report for this approach, rendered as a GitHub Page.

A multi-agent council of five specialised Vision-Language Model agents
that geolocates images by combining three techniques:

1. **Progressive Narrowing (PN):** first narrow the answer to a world
   region, then to a small country hypothesis set.
2. **Parallel Hypotheses (PH):** every agent evaluates every country
   hypothesis simultaneously and scores it on a 5-level confidence
   scale.
3. **Tournament bracket:** the surviving country candidates enter a
   dynamic single-elimination bracket where a judge compares pairs of
   countries head-to-head with streetview + reference images, driving-
   side / road-line prefilters, and per-country specialist evidence.

The final match additionally produces coordinates, and the whole run is
rendered with a `Tournament:` provenance block that lists every match.

This is the approach as it ran on bwUniCluster 3 (H100 80GB). The
results from that cluster run are included under `results/` (per-image
outputs) and under `../results-overview/VLM PN + PH + Tournament/`
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

### Pipeline (LangGraph)

```
prepare_image
    -> [5 agents Round 1: initial assessment]  (parallel)
    -> region_consensus_check
        -> Path A (consensus): country_hypotheses
        -> Path B (no consensus): region_hypotheses -> region_evaluate
                                  -> region_decision -> country_assess
                                  -> country_hypotheses
    -> country_evaluate  (5 agents in parallel, 5-level confidence scale)
    -> road_evidence     (driving-side, road-line, bollard prefilters)
    -> prefilter         (hard eliminations from driving side + road markings)
    -> candidate_pool    (surviving countries after PN + prefilters)
    -> tournament_node   (bracket: pairwise judge matches with RAG references)
        -> final match emits Country + Coordinates + Reasoning
    -> END
```

### Tournament bracket

- Bracket size is dynamic based on how many candidates survived the
  pre-filters:
  - **>=4 survivors:** trimmed to top-N (default 4), 1-vs-4 and 2-vs-3
    semis, then final. 3 matches.
  - **3 survivors:** A vs B, then winner vs C. 2 matches.
  - **2 survivors:** 1 match.
  - **1 survivor:** walkover, no match.
  - **0 survivors:** fall back to top-N from Path B hypotheses by
    specialist score, then run the bracket on those.
- Each match runs through a **tournament_judge** with:
  - streetview + reference images for both countries (RAG-retrieved
    plonkit references)
  - per-country specialist evidence summary
  - driving-side / road-line observations + recovery warnings
- The **final match** additionally asks the judge for coordinates.
- Seeding is by specialist-confidence score summed across the five
  agents (`+2 strongly_support, +1 support, 0 neutral, -1 contradicts,
  -2 strongly_contradicts`).

### End-of-run outputs

- `assessments`, `country_assessments`: per-agent Round 1 + region-
  constrained assessments.
- `progressive_narrowing`: Path A/B decision, confirmed region,
  proposed regions, region-decision reasoning.
- `hypothesis_evaluations`: every agent's 5-level score for every
  country hypothesis.
- `candidate_pool`: countries surviving the prefilters (input to the
  tournament).
- `rag_findings`, `rag_refs_seen`: deterministic RAG eliminations +
  which references were actually shown to a judge match.
- `road_evidence`, `road_filter_warnings`: driving-side / road-line
  observations + recovery warnings when a prefilter would have
  eliminated everything.
- `tournament_log`: full list of bracket matches with winner, both
  countries' reasoning, and agreement status.
- `country_result`: raw judge text with the `Tournament:` provenance
  block.
- `coordinates`: structured `{"lat", "lng"}` or `null`.
- `final_reasoning`: last-match reasoning.

---

## 2. Directory layout

```
.
├── vlm_council/               # Python package (pipeline + agents + tournament)
│   ├── agents/                # 5 specialists + judge + tournament_judge
│   ├── graph.py               # LangGraph pipeline (PN + PH)
│   ├── tournament.py          # Dynamic bracket + match runner
│   ├── prefilters.py          # Driving-side + road-line eliminations
│   ├── road_evidence.py       # Road-observation collection
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
├── eval_pnt/                  # LLM-as-a-Judge evaluation (Stage 2, Qwen 3.6)
│   └── cluster_results/       # Cluster-run eval outputs (report.md/html, plots, judge/)
├── slurm/                     # bwUniCluster 3 launch scripts
├── scripts/
│   └── download_dataset.py    # Local downloader (identical across approaches, run once)
├── results/                   # Per-image result.json (500 images from the cluster run)
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
| `vlm-council-pnt` | Repo checkout + `results/` + `.venv` for this approach | No |
| `hf-cache` | HuggingFace model cache | Yes, reused across approaches |
| `datasets` | Images + `georc_locations.csv` | Yes, reused across approaches |

### On bwUniCluster 3 (H100 80GB)

```bash
ws_allocate datasets 60
ws_allocate hf-cache 60
ws_allocate vlm-council-pnt 30

cd $(ws_find vlm-council-pnt)

# Upload the project files into the workspace, then symlink the dataset
ln -s "$(ws_find datasets)/Images" Images
ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv

bash slurm/setup_uc3.sh
```

---

## 4. Running the pipeline

### Cluster: single job

```bash
cd $(ws_find vlm-council-pnt)
sbatch slurm/run_council_uc3.sh
```

### Cluster: chained batches

`slurm/chain_uc3.sh` submits a chain of council runs and eval jobs
in sequence, useful for large batches.

---

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VLM_MODEL` | `google/gemma-4-31b-it` | Vision model (all 5 agents + judge + tournament_judge) |
| `VLM_JUDGE_MODEL` | = `VLM_MODEL` | Optional separate judge model |
| `VLM_JUDGE_THINKING` | `false` | Judge in thinking mode |
| `VLM_MAX_REGION_HYPOTHESES` | `4` | Parallel region hypotheses in Path B |
| `VLM_MAX_COUNTRY_HYPOTHESES` | `5` | Parallel country hypotheses |
| `VLM_TOURNAMENT_FINALISTS` | `4` | Bracket size cap (top-N by seed score) |
| `VLM_DATA_DIR` | (unset) | RAG data dir (plonkit references, bollards) |
| `VLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLM_MAX_MODEL_LEN` | `16384` | Max context length |
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

For every image `<output_dir>/<image-stem>/result.json` is written. Key
fields (in addition to the standard PN + PH fields):

```json
{
  "candidate_pool": ["Ukraine", "Russia", "Belarus", "Poland"],
  "rag_findings": [
    { "kind": "elim_driving", "country": "Japan", "why": "..." }
  ],
  "road_evidence": { "driving_side": "right", "line_colour": "white" },
  "road_filter_warnings": ["..."],
  "tournament_log": [
    {
      "round_label": "semi-1",
      "country_a": "Ukraine",
      "country_b": "Poland",
      "pool_rank_a": 0,
      "pool_rank_b": 3,
      "winner": "Ukraine",
      "reasoning": "The pre-cast concrete fence panels ...",
      "agreement": "agree"
    }
  ],
  "country_result": "Country: Ukraine\nCoordinates: 48.5, 33.5\nReasoning: ...\n\nTournament:\n  semi-1: Ukraine vs Poland -> Ukraine\n  semi-2: Russia vs Belarus -> Russia\n  final: Ukraine vs Russia -> Ukraine\nRoad Check: ...",
  "coordinates": { "lat": 48.5, "lng": 33.5 }
}
```

### `coordinates` field

- **Success:** `{"lat": <float>, "lng": <float>}`, values in valid
  geographic range.
- **Walkover / no candidates / judge failure:** `null`. No
  `(0, 0)` fake fallback (the tournament emits `null` explicitly for
  degenerate cases so distance metrics don't see (0, 0) outliers).

---

## 7. Results

The 500-image benchmark ran on Gemma 4 31B IT. Full evaluation
artefacts are stored in two locations for convenience:

- `eval_pnt/cluster_results/` (inside this approach folder)
- `../results-overview/VLM PN + PH + Tournament/evaluation/` (consolidated
  across approaches)

Both trees contain the same LLM-as-Judge (Qwen 3.6) evaluation
artefacts:

- `report.md`, `pn_ph_tourn.html`: evaluation report
- `judge_summary.json`: machine-readable metrics
- `plots/`
- `judge/`: 500 per-image LLM-as-judge verdicts

The council-run outputs (500 `result.json` files) are mirrored under
`../results-overview/VLM PN + PH + Tournament/council_run/`.

**Headline numbers (500 images, PN + PH + Tournament, Gemma 4 31B
IT):**

| Metric | Value |
|---|---|
| Country accuracy | 64.8 % (324 / 500) |
| Neighbor hits | 15.2 % (76 / 500) |
| Country or Neighbor | 80.0 % (400 / 500) |
| Median haversine error | 421 km |
| Mean haversine error | 1 470 km |

**Tournament bracket activation:**

| Matches per image | Images | Share |
|---|---|---|
| 0 (walkover, 1 survivor) | 14 | 3 % |
| 1 (2 survivors) | 20 | 4 % |
| 2 (3 survivors) | 61 | 12 % |
| 3 (>=4 survivors, top-N bracket) | 405 | 81 % |

- **Total tournament matches across all 500 images:** 1 357 (mean 2.71
  per image).
- **97.2 %** of images (486 / 500) triggered at least one head-to-head
  match.
- The top-N bracket (4 seed candidates, 3 matches) dominates.

**Candidate-pool size after prefilters** (input to the tournament):

| Survivors | Share |
|---|---|
| 1 | 3 % |
| 2 | 4 % |
| 3 | 12 % |
| 4 | 23 % |
| 5 | 26 % |
| 6 | 27 % |
| 7 | 3 % |
| 8+ | 0 % |

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
