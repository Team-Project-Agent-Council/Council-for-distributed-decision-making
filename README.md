# Team Project VLM Council - GeoRC Geolocation

Multi-agent Vision-Language-Model councils for image country-geolocation
on the GeoRC benchmark. University of Mannheim team project on
**"Multi-LLM Council for Distributed Decision Making"**.

This is a **monorepo**: each approach, the aggregated evaluation, and the
interactive demo live in their own top-level folder.

> - **Evaluation navigation page** (per-approach GT statistics + LLM-as-Judge reports): [Evaluation Overview](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/)  
> - **Interactive GeoBench demo**: [Demo](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/demo/)

## Approaches

| Folder | Approach | Council / Coordination |
|---|---|---|
| [`vlm-progressive-narrowing-with-parallel-hypotheses/`](vlm-progressive-narrowing-with-parallel-hypotheses/) | Progressive Narrowing + Parallel Hypotheses | Region-then-country narrowing, parallel hypothesis scoring |
| [`vlm-pn-ph-tournament/`](vlm-pn-ph-tournament/) | PN + PH + Tournament | Adds a head-to-head elimination bracket |
| [`vlm-tournament-only/`](vlm-tournament-only/) | Tournament Only | Bracket straight from the initial pass (best VLM result) |
| [`vlm-hub-and-spoke/`](vlm-hub-and-spoke/) | Hub and Spoke | Judge coordinator dispatches follow-ups to agents |
| [`vlm-debate/`](vlm-debate/) | Debate | Adversarial moderator pairs disagreeing agents |
| [`vlm-global-context-reguess/`](vlm-global-context-reguess/) | Global Context Reguess | Two-round re-guess with global context |
| [`initial-approach/`](initial-approach/) | Initial Approach | Grounded vision pipeline + 4 council variants (Ollama, 100-image subset) |
| [`vlm-baseline/`](vlm-baseline/) | Baseline | Single-pass VLM guess, no council (reference point) |

## Supporting folders

| Folder | Contents |
|---|---|
| [`results-overview/`](results-overview/) | Aggregated per-approach results (GT statistics + LLM-as-Judge) and the static-site generator behind the GitHub Pages navigation page |
| [`demo/`](demo/) | Interactive GeoBench demo (Next.js frontend + FastAPI backend) that replays a recorded Progressive Narrowing run |

## Two evaluation tracks

Every VLM council approach is evaluated on:

1. **Ground-truth statistics** (`<approach>/results/` + the
   `analyze_rounds` scripts): country accuracy, haversine distance, and
   per-gate GT survival.
2. **LLM-as-Judge** (Qwen 3.6): role adherence, hallucination, synthesis
   quality, and approach-specific dynamics.

The `results-overview/` folder consolidates both tracks per approach.

## GitHub Pages

A single Pages site is built from this repo on every push to `main`:

- `/` - the evaluation **navigation page** (one card per approach,
  linking each approach's consolidated evaluation report).
- `/demo/` - the interactive demo.

Enable once under **Settings -> Pages -> Source: GitHub Actions**. The
workflow is [`.github/workflows/pages.yml`](.github/workflows/pages.yml).

## Running an approach on the cluster

Each VLM council approach ships bwUniCluster 3 (uc3) launch scripts under
its own `slurm/` folder. The core scripts are shared across approaches:

- `slurm/setup_uc3.sh` - one-time environment setup
- `slurm/run_council_uc3.sh` - the council run (vLLM + Gemma 4)

Beyond that, the exact set of scripts varies by approach depending on how
its jobs are batched and evaluated:

- **Batch launcher.** Most approaches split the images into parallel
  `gpu_h100_short` jobs with `slurm/launch_short_uc3.sh`. The two
  tournament approaches (`vlm-pn-ph-tournament`, `vlm-tournament-only`)
  instead use `slurm/chain_uc3.sh`, a dependency-chain launcher that
  serializes 30-min jobs so each resumes where the previous left off.
- **LLM-as-Judge.** Approaches that run the judge on the cluster ship a
  `slurm/run_eval_uc3.sh` (and, for the tournament approaches, a
  `slurm/run_eval_uc3_qwen.sh` Qwen variant). `vlm-debate` additionally
  splits its judge stage with `slurm/launch_judge_short_uc3.sh` /
  `slurm/run_judge_uc3.sh`. Hub-and-Spoke and Global Context Reguess have
  no cluster eval script; their judge track was run separately.

Each approach's README documents its own script set. The Initial Approach
runs on Ollama with a single `slurm/run_vision_uc3.sh`, and the Baseline
uses its own `run_baseline_*.sh` scripts (both documented in their READMEs).

The GeoRC dataset is downloaded once and reused across all approaches -
see any approach's `scripts/download_dataset.py`.
