# VLM Council - GeoRC Evaluation Results

> **Results Overview:** [https://team-project-agent-council.github.io/Council-for-distributed-decision-making/](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/)  
> The **navigation page**: one card per approach, each linking to that
> approach's consolidated evaluation report. Rebuilt automatically
> on every push to `main`.

Evaluation results for all VLM Council approaches to GeoRC image
country-geolocation. Each approach has one **consolidated evaluation
report** (`<approach>/evaluation/`) that combines:

1. **Ground-truth statistics** - country accuracy, haversine distance,
   geographic bias, and per-agent accuracy against the ground truth.
2. **Approach dynamics** - approach-specific behaviour (debate/round/
   tournament/narrowing), including where in the pipeline the correct
   answer is lost.
3. **LLM-as-Judge verdicts** - a Qwen 3.6 judge scores each run for role
   adherence, hallucination, and synthesis quality. Rendered as
   `report.html` with plots and a machine-readable `judge_summary.json`.

## Approaches

| Approach | Report | Live page |
|---|---|---|
| VLM Progressive Narrowing with Parallel Hypotheses | Evaluation report | [open](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Progressive_Narrowing_with_Parallel_Hypotheses/report.html) |
| VLM PN + PH + Tournament | Evaluation report | [open](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_PN_plus_PH_plus_Tournament/report.html) |
| VLM Tournament Only | Evaluation report | [open](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Tournament_Only/report.html) |
| VLM Hub and Spoke | Evaluation report | [open](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Hub_and_Spoke/report.html) |
| VLM Global Context Reguess | Evaluation report | [open](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Global_Context_Reguess/report.html) |
| VLM Debate | Evaluation report | [open](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Debate/report.html) |

The **Initial Approach** ships its results here too (`VLM Initial
Approach/`), but as per-variant CSVs (`llm_council_evals/`) rather than a
consolidated evaluation report, so `build_site.py` excludes it from the
navigation page. The **Baseline** (single-pass VLM, no council) is not
consolidated here; its results live in `../vlm-baseline/results/`.

## Published site

A GitHub Pages site is generated from these results on every push to
`main` and published automatically. The **navigation page**
([https://team-project-agent-council.github.io/Council-for-distributed-decision-making/](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/))
links each approach to its consolidated evaluation report.

- Generator: [`scripts/build_site.py`](scripts/build_site.py) (standard
  library only, no dependencies).
- Workflow: [`.github/workflows/pages.yml`](.github/workflows/pages.yml).

### Enabling Pages (one-time)

In the GitHub repo: **Settings -> Pages -> Build and deployment ->
Source: GitHub Actions**. The next push to `main` (or a manual
**Actions -> Deploy evaluation site -> Run workflow**) publishes the
site.

### Building locally

```bash
python3 scripts/build_site.py
# open site/index.html
```

The `site/` directory is a build artefact and is git-ignored; it is
rebuilt fresh by the workflow.

## Layout

```
.
├── <approach>/
│   ├── analysis/              # GT-based track
│   │   ├── gt_pipeline_analysis.txt
│   │   └── plots/*.png        # (Debate only)
│   ├── evaluation/            # LLM-as-Judge track
│   │   ├── report.html
│   │   ├── judge_summary.json
│   │   ├── plots/*.png
│   │   └── judge/*.json       # per-image verdicts
│   └── council_run/           # per-image raw council outputs (result.json)
├── Initial Approach/
│   ├── llm_council_evals/*.csv
│   └── vision_pipeline_run/
├── scripts/build_site.py
└── .github/workflows/pages.yml
```
