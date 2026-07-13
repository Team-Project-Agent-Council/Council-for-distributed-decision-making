# LLM Council - Optimized Prompt Setup, selected Tools (Variant 3)

Standalone, cluster-runnable sub-repo for the **"Optimized Prompt Setup (selected Tools)"**
variant of the Initial Approach GeoGuessr council.

This is the **best-performing** council variant. It keeps **seven** specialist agents plus
a judge, but does two things differently from the earlier variants:

1. **Revised prompts** - the orchestrator and every specialist use rewritten, more
   disciplined prompts (copy-verbatim road-marking extraction, "only use what is
   described", hemisphere awareness, etc.).
2. **Curated tools** - tools are deliberately reduced. **Only `botanics` and `judge` have
   tool nodes.** The other five specialists (`linguistic`, `landscape`, `infrastructure`,
   `regulatory`, `cultural`) run **prompt-only**: they go `agent -> reasoning` directly with
   no `ToolNode` and no tool routing. This tool reduction is what "selected Tools" means.

The `cultural` agent **replaces** the earlier `climate` agent. The graphs are imported
AS-IS from the source branch `origin/prompt_optimization`.

## Overview

A multi-agent LangGraph system that identifies the country (and GPS coordinates) from a
street-level image. Seven specialist agents run in parallel and feed into a judge that
produces the final answer. The council model is `qwen3:32b` served by **Ollama**.

There are two graphs in this repo:

- `council/graph.py` - the full interactive/Studio graph. It begins with the Stage-1
  **vision pipeline** (scene parser -> detail identifier -> detail extractor -> crop tool
  -> detail focusser -> vision mapping, from the shared `vision_pipeline` root package) and
  then runs the orchestrator + seven specialists + judge.
- `evaluation/graph.py` - the **eval-only** graph. It has **no vision node**: the caller
  pre-populates `general_description` and `crop_descriptions` from each pre-computed
  `result.json`, so START goes straight to the orchestrator. This is the graph that
  produced the submission CSV.

## Agents (7 specialists + judge)

| Agent | Role | Tools | Model |
|---|---|---|---|
| **Vision pipeline** (interactive graph only) | Multi-step Stage-1 image analysis, produces `general_description` + `crop_descriptions` | multi-step (shared `vision_pipeline`) | `qwen3-vl:32b` |
| **Orchestrator** | Splits descriptions into per-agent prompts (verbatim road-marking extraction first) | - (prompt-only) | `qwen3:32b` |
| **Linguistic** | Reads languages/scripts on signs | **none (prompt-only)** | `qwen3:32b` |
| **Landscape** | Classifies terrain and vegetation | **none (prompt-only)** | `qwen3:32b` |
| **Botanics** | Looks up plant species distributions | `plant_search`, `gbif_distribution`, `powo_distribution` | `qwen3:32b` |
| **Regulatory** | Matches road signs, markings, driving conventions to country standards | **none (prompt-only)** | `qwen3:32b` |
| **Infrastructure** | Analyzes vehicles, street furniture, and architecture | **none (prompt-only)** | `qwen3:32b` |
| **Cultural** | Reads clothing, murals, shops, religious buildings, colonial heritage | **none (prompt-only)** | `qwen3:32b` |
| **Meta** | RAG/meta branch (skipped in eval; requires Ollama embeddings) | - | `qwen3:32b` |
| **Judge** | Synthesizes all agent outputs, geocodes, produces final answer | `identify_country`, `wikidata_search`, `wikidata_sparql`, `geocode` | `qwen3:32b` |

**Only `botanics` and `judge` have tool nodes.** This curation is the whole point of the
"selected Tools" variant. The five prompt-only specialists produce their ranked country
lists purely from the orchestrator's prompt, with no external lookups.

- **Botanics branch** (the only specialist with tools): `botanics_agent -> (conditional)
  botanics_tools -> botanics_continue -> (conditional) botanics_tools_2 -> botanics_reasoning`.
  A two-round tool loop: first round identifies species names (`plant_search`), second round
  looks up distributions (`gbif_distribution`, `powo_distribution`).
- **Prompt-only branches** (linguistic, landscape, infrastructure, regulatory, cultural):
  `agent -> reasoning` in a single node - no `ToolNode`, no routing.
- **Judge branch**: `judge_agent -> (conditional) judge_tools -> judge_reasoning`.

## Graph

```
                        +-- linguistic_agent      (prompt-only)                            --+
                        +-- landscape_agent        (prompt-only)                            --+
                        +-- botanics_agent  --> [tools] --> botanics_continue --> [tools] --> botanics_reasoning --+
orchestrator_agent --+--+-- regulatory_agent      (prompt-only)                            --+--> judge_agent --> [judge_tools] --> judge_reasoning --> END
                        +-- infrastructure_agent   (prompt-only)                            --+
                        +-- cultural_agent         (prompt-only)                            --+
                        +-- meta_agent             (RAG, skipped in eval)                   --+
```

- **Fan-out:** the orchestrator has **7 outgoing edges**, one per specialist.
- **Fan-in barrier:** the judge fires only after all seven branches complete
  (`linguistic_agent`, `landscape_agent`, `botanics_reasoning`, `regulatory_agent`,
  `infrastructure_agent`, `cultural_agent`, `meta_agent`).
- Interactive graph (`council/graph.py`):
  `START -> scene_parser -> ... -> vision_mapping -> orchestrator_agent -> ...`
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
    --output optimized_prompts.csv \
    --concurrency 1 --verbose
```

`--results-dir` points at the directory of per-location `result.json` files (the
pre-computed Stage-1 vision outputs). The eval graph reads `general_description` +
`crop_descriptions` from each and runs the council. Ground truth comes from `--mapping`.

## Results

- **Country accuracy: 23% (23/100)** - the **best** council variant
- **Median distance: 6035 km**
- **Mean distance: 6618 km**

Verified eval CSV: `../results/llm_council_evals/optimized_prompts.csv` (100 rows).

## Notes

- **Key finding:** prompt engineering + **tool reduction** beat adding more agents and more
  tools. Variant 2 ("additional_agents", full tool sets on all specialists) scored 13%;
  this variant keeps the same seven-agent structure but strips tools from five specialists
  and rewrites the prompts, reaching 23%. The tools were adding noise, not signal - a
  smaller, sharper prompt-only council out-performs a tool-heavy one.
- The eval-only graph (`evaluation/graph.py`) is intentionally independent of
  `council/graph.py` so changes to the interactive/Studio pipeline can never affect the
  numbers that produced the CSV. Both wire the identical curated tool set (only botanics +
  judge have tool nodes).
- Stage-1 vision is the **shared** monorepo package `vision_pipeline` (at the repo root),
  not duplicated here. `council/graph.py` imports its nodes directly
  (`from vision_pipeline.scene_parser import scene_parser`, etc.). The local
  `council/vision_agent.py` is a thin standalone single-shot wrapper exposing
  `run(image_path) -> object` with `.general_description` and `.crop_descriptions`.
- No `(0,0)` fake-coordinate fallback: a location with no valid geocode is left without a
  predicted coordinate rather than defaulting to null island.
