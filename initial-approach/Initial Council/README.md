# Initial Council (Variant 1)

The baseline council of the *Initial Approach*. Five specialist agents
reason in parallel over the shared vision-pipeline output and a judge
fuses their findings into a single country + coordinate prediction.

This is a standalone, cluster-runnable sub-repo. It consumes the
pre-computed Stage-1 vision outputs from the shared
`../vision_pipeline/` (100 `result.json` under `../results/`).

**Result: 16 % top-1 country accuracy (16 / 100)**, median 6560 km,
mean 6315 km on the 100-image GeoRC subset.

---

## Overview

Two-stage system, this sub-repo is Stage 2 (the council):

1. **Stage 1 (shared vision pipeline)** produces, per image, a dense
   `scene_description` plus per-crop `focused_description`s. Already run;
   outputs live in `../results/<image>/result.json`.
2. **Stage 2 (this council)** reads those descriptions and runs five
   specialists in parallel, then a judge.

In evaluation mode (`evaluate.py`) the vision stage is skipped entirely
and the council reads the pre-computed `result.json`, so the reported
CSV reproduces exactly.

---

## Agents

Five specialists fan out in parallel from the orchestrator; each keeps
its full tool set. A single judge fuses them.

| Agent | Tools |
|---|---|
| `linguistic`  | detect_language, wikidata_search, wikidata_sparql |
| `landscape`   | identify_landscape, wikidata_search, wikidata_sparql |
| `botanics`    | plant_search, gbif_distribution, powo_distribution |
| `regulatory`  | web_search, fetch_page |
| `meta`        | rag_search, wikidata_search, wikidata_sparql |
| `judge`       | identify_country, wikidata_search, wikidata_sparql, geocode |

This is the additional-agents graph with the climate and infrastructure
specialists removed (they are added back in Variant 2).

---

## Graph

```
START -> orchestrator
           |-- linguistic  -> (tools?) -> reasoning --+
           |-- landscape   -> (tools?) -> reasoning --+
           |-- botanics    -> (tools?) -> reasoning --+--> judge -> (tools?) -> reasoning -> END
           |-- regulatory  -> (tools?) -> reasoning --+
           |-- meta        -> (tools?) -> reasoning --+
```

- `council/graph.py`: full pipeline including the live vision node
  (`START -> vision_agent -> orchestrator -> ...`), used by LangGraph
  Studio / `invoke_council.py`.
- `evaluation/graph.py`: eval-only graph, no vision node
  (`START -> orchestrator -> ...`); `general_description` and
  `crop_descriptions` are pre-populated from `result.json`. This is the
  graph that produced the CSV.

---

## Cluster run (bwUniCluster 3, H100, Ollama)

The council ran on **Ollama** (not vLLM), council/judge model
`qwen3:32b`.

```bash
# On the GPU node (via sbatch run_eval_gpu.sh):
export PATH=$(ws_find ollama_models)/bin:$PATH
OLLAMA_HOST=0.0.0.0 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_NUM_PARALLEL=5 ollama serve &

# Python env
uv sync
uv pip install pysqlite3-binary          # cluster sqlite3 too old for chromadb
export CHROMA_SQLITE_PATCH=1

# Batch council-run + evaluation over all 100 pre-computed vision results
python evaluate.py \
    --results-dir "results GeoRC" \
    --mapping georc-location-image-mapping.csv \
    --output initial_council.csv \
    --concurrency 1 --verbose
```

Single-image debugging: `LOCATION_ID=<id> sbatch run_georc_test_gpu.sh`
(drives `georc_test.py`).

---

## Results

**16 % top-1 country accuracy (16 / 100)**, median haversine 6560 km,
mean 6315 km.

The eval CSV is at
`../results/llm_council_evals/initial_council.csv`
(consolidated copy under
`../../results-overview/VLM Initial Approach/llm_council_evals/`).

---

## Notes

- This is the strongest of the two parallel-fan-out configurations that
  use the full tool set: adding climate + infrastructure agents
  (Variant 2) *lowered* accuracy to 13 %.
- All five specialists have tools here. Variants 3 and 4 deliberately
  strip tools from most agents (only botanics + judge keep them), which
  turned out to help.
