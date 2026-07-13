# VLM Council v12 - Evaluation Suite

Two-stage evaluation over `results_v12_pn/*/result.json`.

## Stages

**Stage 1 (CPU, deterministic)** - runs in seconds, no LLM:
- `eval.geo` - geo-spatial bias, north/east t-tests, error histograms, country-pair confusion (incl. asymmetric pairs).
- `eval.agents` - per-agent top-1/top-3/coverage, confidence calibration, per-region accuracy.
- `eval.influence` - Path-B revision buckets (productive vs destructive), 5×5 persuasion matrix, hypothesis-consensus and tournament alignment.

**Stage 2 (GPU, LLM-as-judge)** - needs vLLM:
- `eval.judge` - per image, sends `(ground truth, image, full discussion trace)` to the judge VLM. Pydantic-enforced verdict covers role adherence (T/F per agent), argumentative quality (5-scale), assertiveness (5-scale), constructive synthesis (bool), cross-agent decision influence (correct/incorrect change counts), and free-text notes.
- `eval.judge_aggregate` - folds per-image verdicts into `judge_summary.json`. Pass `--results <dir>` so that agent abstentions (empty candidate lists, e.g. the linguistic agent on an image without legible text) are excluded from role-adherence scoring rather than counted as violations; without it, role adherence is computed over every verdict.

**Report** - `eval.report` composes a single `report.md` with linked PNGs.

## Layout

```
eval_outputs/
├── geo_metrics.json
├── agent_metrics.json
├── influence.json
├── judge/<image_id>.json     # one per image (resume-capable)
├── judge_summary.json
├── plots/
│   ├── error_distribution.png
│   ├── bearing_rose.png
│   ├── confusion_matrix.png
│   ├── agent_top1.png
│   ├── agent_calibration.png
│   ├── persuasion_matrix.png
│   ├── judge_role_adherence.png
│   ├── judge_quality.png
│   └── judge_influence.png
└── report.md
```

## Running

### Local - Stage 1 only (no GPU)

```bash
pip install -e '.[eval]'
vlm_council_v12/eval/run_local_stage1.sh \
    vlm_council_v12/results_v12_pn \
    georc_locations.csv \
    eval_outputs
```

Or directly:

```bash
python -m eval all --skip-judge \
    --results vlm_council_v12/results_v12_pn \
    --gt georc_locations.csv \
    --out eval_outputs
```

### Cluster - full pipeline (Stage 1 + 2 + report)

```bash
sbatch vlm_council_v12/slurm/run_eval_uc3_v12.sh
```

The SLURM script boots vLLM on the same model the council uses (`google/gemma-4-31b-it`), runs all three Stage-1 commands, then `eval.judge` against each per-image `result.json`, then aggregate + report. Resume: re-submit if the slot expires; per-image `judge/<image_id>.json` files are skipped.

Smoke test (3 images judged):

```bash
sbatch vlm_council_v12/slurm/run_eval_uc3_v12.sh --limit 3
```

## Subcommands

```bash
python -m eval geo       --results <dir> --gt <csv> --out <out>
python -m eval agents    --results <dir> --gt <csv> --out <out>
python -m eval influence --results <dir> --gt <csv> --out <out>
python -m eval judge     --results <dir> --gt <csv> --out <out> \
                         [--image-root <dir>] [--model <m>] [--api-base <url>] \
                         [--concurrency 4] [--limit N]
python -m eval aggregate --out <out> --results <dir>
python -m eval report    --out <out>
python -m eval all       --results <dir> --gt <csv> --out <out> \
                         [--skip-judge] [--concurrency 4]
```

## Environment variables

| Var | Purpose |
|---|---|
| `VLM_JUDGE_LLM_MODEL` | Judge model id (defaults to council `VLM_MODEL`) |
| `VLM_JUDGE_LLM_API_BASE` | Judge endpoint (`http://host:port/v1`) |
| `VLM_RESULTS_DIR` | SLURM input - defaults to `results_v12_pn/` |
| `VLM_GT_CSV` | SLURM input - defaults to `georc_locations.csv` |
| `VLM_IMAGE_ROOT` | SLURM input - defaults to `Images/` |
| `VLM_EVAL_OUT` | SLURM output - defaults to `eval_outputs/` |
| `VLM_JUDGE_CONCURRENCY` | Async semaphore size (default 4) |

## Not yet implemented (future work)

- **Anchoring / ordering bias.** Replay `judge.decide_country()` against saved hypothesis evaluations with shuffled candidate order, then count how often the final winner changes. Needs another full GPU run.
