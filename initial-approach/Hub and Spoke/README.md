# LLM Council - Hub and Spoke (Variant 4)

Standalone, cluster-runnable sub-repo for the **"Initial Council with Hub and Spoke"**
variant of the Initial Approach GeoGuessr council.

This variant uses the **same seven specialists and the same CURATED tool set as Variant 3**
("optimized_prompts" / selected Tools): only **botanics** and **judge** have tool nodes;
the other five specialists run prompt-only. On top of that baseline it adds a
**hub-and-spoke deliberation loop**: after the parallel fan-out, the judge repeatedly
reviews the evidence and can dispatch confrontational follow-up questions to individual
specialists before finalizing. The graph is imported AS-IS from the source branch
`origin/hub_and_spoke`.

## Overview

A multi-agent LangGraph system that identifies the country (and GPS coordinates) from a
street-level image. Seven specialist agents run in parallel and feed into a judge, which
then runs a bounded deliberation loop before producing the final answer. The council model
is `qwen3:32b` served by **Ollama**.

There are two graphs in this repo:

- `council/graph.py` - the full interactive/Studio graph. It begins with the shared
  Stage-1 vision pipeline (`scene_parser -> detail_identifier -> ... -> vision_mapping`),
  then runs the orchestrator + seven specialists + the hub-and-spoke deliberation loop +
  judge.
- `evaluation/graph.py` - the **eval-only** graph. It has **no vision node**: the caller
  pre-populates `general_description` and `crop_descriptions` from each pre-computed
  `result.json`, so START goes straight to the orchestrator. This is the graph that
  produced the submission CSV.

## Agents (7 specialists + judge)

Tools are **CURATED**: only **botanics** and **judge** have tool nodes. The other five
specialists (linguistic, landscape, regulatory, infrastructure, cultural) and the meta
agent run prompt-only in the eval graph.

| Agent | Role | Tools | Model |
|---|---|---|---|
| **Orchestrator** | Splits descriptions into per-agent prompts | prompt-only | `qwen3:32b` |
| **Linguistic** | Reads languages/scripts on signs | prompt-only | `qwen3:32b` |
| **Landscape** | Classifies terrain and vegetation | prompt-only | `qwen3:32b` |
| **Botanics** | Looks up plant species distributions | `plant_search`, `gbif_distribution`, `powo_distribution` | `qwen3:32b` |
| **Regulatory** | Matches road signs, markings, driving side to country standards | prompt-only | `qwen3:32b` |
| **Infrastructure** | Analyzes vehicles, street furniture, and architecture | prompt-only | `qwen3:32b` |
| **Cultural** | Reads clothing, murals, shops, religious buildings, colonial heritage | prompt-only | `qwen3:32b` |
| **Meta** | RAG/meta agent (skipped in the eval graph) | prompt-only | `qwen3:32b` |
| **Judge** | Deliberates, synthesizes evidence, geocodes, produces final answer | `identify_country`, `wikidata_search`, `wikidata_sparql`, `geocode` | `qwen3:32b` |

The botanics branch runs a two-round tool loop
(`botanics_agent -> botanics_tools -> botanics_continue -> botanics_tools_2 -> botanics_reasoning`);
the other five specialists produce their assessment directly from the prompt.

Every specialist also exposes `respond_to_followup(...)`, and the judge exposes
`deliberate(...)` - these power the hub-and-spoke loop described below.

## Graph

```
                        +-- linguistic_agent      --+
                        +-- landscape_agent        --+
                        +-- botanics_agent (tools)  --+
orchestrator_agent --+--+-- regulatory_agent       --+--> judge_deliberation --+
                        +-- infrastructure_agent   --+          ^      |         |
                        +-- cultural_agent          --+          |      | follow-ups
                        +-- meta_agent              --+          |      v
                                                                 +-- followup_dispatch
                                                                        (spokes: respond_to_followup)

judge_deliberation --(satisfied / max rounds)--> judge_agent --> [judge_tools] --> judge_reasoning --> END
```

- **Fan-out:** the orchestrator has **7 outgoing edges**, one per specialist.
- **Fan-in barrier:** the deliberation hub (`judge_deliberation`) fires only after all
  seven branches complete
  (`add_edge([...linguistic, landscape, botanics_reasoning, regulatory, infrastructure, cultural, meta...], "judge_deliberation")`).
- **Hub-and-spoke deliberation loop:**
  - `judge_deliberation` (the hub) reviews every specialist's assessment. If the evidence
    is sufficient (all agents agree on the same #1 country) or the round limit is reached,
    it routes to `judge_agent` to finalize. Otherwise it emits a set of **confrontational
    follow-up questions** targeted at specific specialists.
  - `followup_dispatch` routes each question to the named spoke via that agent's
    `respond_to_followup(original_result, question, original_prompt, prior_exchanges)`,
    runs them in parallel, writes the updated results back into state, records the exchange
    in `agent_followup_history`, and loops back to `judge_deliberation`.
  - The loop is bounded by `MAX_DELIBERATION_ROUNDS = 3`; on the final round the judge is
    forced to finalize.
- Interactive graph (`council/graph.py`): `START -> scene_parser -> ... -> vision_mapping -> orchestrator_agent -> ...`
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
    --output hub_and_spoke.csv \
    --concurrency 1 --verbose
```

`--results-dir` points at the directory of per-location `result.json` files (the
pre-computed Stage-1 vision outputs). The eval graph reads `general_description` +
`crop_descriptions` from each and runs the council. Ground truth comes from `--mapping`.

## Results

- **Country accuracy: 16% (16/100)**
- **Median distance: 7277 km**
- **Mean distance: 7116 km**

The hub-and-spoke deliberation loop gave **no gain over the parallel fan-out baseline**.
Adding the judge-driven follow-up rounds on top of the curated-tool council did not sharpen
the country prediction: the extra rounds mostly re-confirmed positions rather than
resolving disagreements, so accuracy matched the plain parallel council rather than
improving on it.

Verified eval CSV: `../results/llm_council_evals/hub_and_spoke.csv` (100 rows).

## Notes

- The eval-only graph (`evaluation/graph.py`) is intentionally independent of
  `council/graph.py` so changes to the interactive/Studio pipeline can never affect the
  numbers that produced the CSV.
- Stage-1 vision is the **shared** monorepo package `vision_pipeline` (at the repo root),
  not duplicated here. `council/graph.py` imports its nodes directly
  (`scene_parser`, `detail_identifier`, `detail_extractor`, `crop_tool`,
  `detail_focusser`, `state`, `config`). The local `council/vision_agent.py` is a thin
  standalone wrapper exposing `run(image_path) -> object` (with `.general_description` and
  `.crop_descriptions`) and is not used by either graph's default path.
- No `(0,0)` fake-coordinate fallback: a location with no valid geocode is left without a
  predicted coordinate rather than defaulting to null island.
