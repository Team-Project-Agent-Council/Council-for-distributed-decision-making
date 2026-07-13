# LLM Council - Additional Agents (Variant 2)

Standalone, cluster-runnable sub-repo for the **"Initial Council with additional Agents
(Climate and Infrastructure)"** variant of the Initial Approach GeoGuessr council.

This variant extends the 5-agent baseline with two extra specialists -
**infrastructure** and **climate** - giving **seven** specialist agents plus a judge.
Every specialist keeps its FULL tool set. The graph is imported AS-IS from the source
branch `origin/feat/tool_agents`.

## Overview

A multi-agent LangGraph system that identifies the country (and GPS coordinates) from a
street-level image. Seven specialist agents run in parallel and feed into a judge that
produces the final answer. The council model is `qwen3:32b` served by **Ollama**.

There are two graphs in this repo:

- `council/graph.py` - the full interactive/Studio graph. It begins with a
  `vision_agent` node (Stage-1 vision) and then runs the orchestrator + seven specialists
  + judge.
- `evaluation/graph.py` - the **eval-only** graph. It has **no vision node**: the caller
  pre-populates `general_description` and `crop_descriptions` from each pre-computed
  `result.json`, so START goes straight to the orchestrator. This is the graph that
  produced the submission CSV.

## Agents (7 specialists + judge)

| Agent | Role | Tools | Model |
|---|---|---|---|
| **Vision** (interactive graph only) | Analyzes the image, produces `general_description` + `crop_descriptions` | - | `qwen3-vl:32b` |
| **Orchestrator** | Splits descriptions into per-agent prompts | - | `qwen3:32b` |
| **Linguistic** | Detects languages/scripts on signs, queries Wikidata | `detect_language`, `wikidata_search`, `wikidata_sparql` | `qwen3:32b` |
| **Landscape** | Classifies terrain and vegetation, queries Wikidata | `identify_landscape`, `wikidata_search`, `wikidata_sparql` | `qwen3:32b` |
| **Botanics** | Looks up plant species distributions | `plant_search`, `gbif_distribution`, `powo_distribution` | `qwen3:32b` |
| **Regulatory** | Matches road signs, markings, infrastructure to country standards | `web_search`, `fetch_page` | `qwen3:32b` |
| **Infrastructure** | Analyzes vehicles, street furniture, and architecture | `vehicle_analysis`, `street_analysis`, `architecture_analysis` | `qwen3:32b` |
| **Climate** | Infers climate zone from the scene | `climate_analysis` | `qwen3:32b` |
| **Meta** | Queries ChromaDB RAG knowledge base + Wikidata | `rag_search`, `wikidata_search`, `wikidata_sparql` | `qwen3:32b` |
| **Judge** | Synthesizes all agent outputs, geocodes, produces final answer | `identify_country`, `wikidata_search`, `wikidata_sparql`, `geocode` | `qwen3:32b` |

Each specialist branch is: `agent -> (conditional) tools -> reasoning`. If the agent's
first response contains tool calls, it routes through its `ToolNode`; otherwise it goes
straight to reasoning.

## Graph

```
                        +-- linguistic_agent     --> [tools] --> linguistic_reasoning     --+
                        +-- landscape_agent       --> [tools] --> landscape_reasoning       --+
                        +-- botanics_agent        --> [tools] --> botanics_reasoning        --+
orchestrator_agent --+--+-- regulatory_agent      --> [tools] --> regulatory_reasoning      --+--> judge_agent --> [tools] --> judge_reasoning --> END
                        +-- infrastructure_agent  --> [tools] --> infrastructure_reasoning  --+
                        +-- climate_agent         --> [tools] --> climate_reasoning         --+
                        +-- meta_agent            --> [tools] --> meta_reasoning            --+
```

- **Fan-out:** the orchestrator has **7 outgoing edges**, one per specialist.
- **Fan-in barrier:** the judge fires only after **all 7** reasoning nodes complete
  (a single `add_edge([...7 nodes...], "judge_agent")`).
- Interactive graph (`council/graph.py`): `START -> vision_agent -> orchestrator_agent -> ...`
- Eval graph (`evaluation/graph.py`): `START -> orchestrator_agent -> ...` (no vision node).

## Cluster run (bwUniCluster 3, Ollama)

Inference runs on an H100 GPU node with **Ollama** (not vLLM). The council/judge model is
`qwen3:32b`; Stage-1 vision (already run to produce the `result.json` files) used
`qwen3-vl:32b`.

The self-contained Slurm job `run_eval_gpu.sh` handles everything: it starts Ollama on the
GPU node (`ws_find ollama_models`, `OLLAMA_MAX_LOADED_MODELS=3`, `OLLAMA_NUM_PARALLEL=5`),
verifies `qwen3:32b`, points every agent at the local Ollama, sets up the Python env, runs
the eval, then cleans up.

```bash
# Submit the GPU eval job
sbatch run_eval_gpu.sh

# Monitor
squeue -u $USER
tail -f logs/eval_gpu_<jobid>.out
```

### Python environment

```bash
uv sync
uv pip install pysqlite3-binary   # cluster sqlite3 is too old for chromadb
export CHROMA_SQLITE_PATCH=1       # patches sqlite3 -> pysqlite3 before chromadb loads
```

### Run the evaluation directly

Driven by `evaluate.py` (loader -> `build_eval_graph` -> metrics -> report):

```bash
uv run python evaluate.py \
    --results-dir "results GeoRC" \
    --mapping ../results/llm_council_evals/location-image-mapping.csv \
    --output additional_agents.csv \
    --concurrency 1 --verbose
```

`--results-dir` points at the directory of per-location `result.json` files (the
pre-computed Stage-1 vision outputs). The eval graph reads `general_description` +
`crop_descriptions` from each and runs the council. Ground truth comes from `--mapping`.

## Results

- **Country accuracy: 13% (13/100)**
- **Median distance: 7489 km**
- **Mean distance: 6852 km**

Adding the climate and infrastructure agents **lowered** accuracy versus the 5-agent
baseline ("initial_council", 16%). The extra specialists added noise to the judge's
synthesis rather than sharpening the country prediction.

Verified eval CSV: `../results/llm_council_evals/additional_agents.csv` (100 rows).

## Notes

- The eval-only graph (`evaluation/graph.py`) is intentionally independent of
  `council/graph.py` so changes to the interactive/Studio pipeline can never affect the
  numbers that produced the CSV.
- Stage-1 vision is the **shared** monorepo package `vision_pipeline` (at the repo root),
  not duplicated here. The local `council/vision_agent.py` is a thin standalone wrapper
  exposing `run(image_path) -> VisionOutput` (with `.general_description` and
  `.crop_descriptions`) and is only used by the interactive graph.
- No `(0,0)` fake-coordinate fallback: a location with no valid geocode is left without a
  predicted coordinate rather than defaulting to null island.
