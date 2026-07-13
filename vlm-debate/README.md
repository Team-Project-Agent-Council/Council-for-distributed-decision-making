# VLM Debate

> **Results Overview:** [https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Debate/report.html](https://team-project-agent-council.github.io/Council-for-distributed-decision-making/VLM_Debate/report.html)  
> GT-based statistics and the LLM-as-Judge (Qwen 3.6) report for this approach, rendered as a GitHub Page.

A multi-agent council of five specialised Vision-Language Model agents
that collaboratively geolocate images. The council uses a
**Debate topology** with an adversarial moderator: after each agent
produces an independent assessment in Round 1, a Moderator inspects
the outputs for **contradictions**, pairs the disagreeing agents, and
runs them through structured back-and-forth exchanges. Only when the
moderator declares the debate resolved (or the round cap is reached)
does a Judge synthesise the final country + coordinates.

This is the approach as it ran on bwUniCluster 3 (H100 80GB). The
results from that cluster run are included under `results/` (per-image
outputs) and under `../results-overview/VLM Debate/` (consolidated results +
full LLM-as-Judge evaluation).

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

Each agent returns a list of **country candidates** with confidence and
supporting evidence.

### Debate pipeline (LangGraph)

```
prepare_image
    -> [5 agents Round 1: independent assessment]  (parallel)
    -> round1_complete  (barrier)
    -> moderator  <-------------+
         |                      |
         | terminate=false      | (next debate round)
         v                      |
       debate_round  ---------->+
         (all pairings in parallel)
         |
         | terminate=true
         v
    judge_final -> END
```

- **Round 1** is a clean-slate assessment: every agent sees only the
  image and its own domain prompt.
- **Moderator** is a Judge-LLM call that inspects all Round-1 outputs,
  identifies concrete contradictions (e.g. Linguistic says Cyrillic,
  Regulatory says Turkish plates), and opens **pairings** between the
  disagreeing agents. It can also decide to terminate immediately if
  the picture is clear.
- **Debate round** runs all pairings in parallel. Each pairing consists
  of paired agents exchanging arguments; either agent may **revise**
  its position after seeing the other's rebuttal.
- The Moderator sees the updated positions and either opens another
  round or terminates. Up to `DEBATE_MAX_ROUNDS` rounds allowed.
- **Judge** finally synthesises the country from Round-1 outputs and
  the full debate transcript.

### Phase-by-phase walkthrough

1. **`prepare_image`** encodes the image once as base64.
2. **`round1_<name>`** (5 nodes, parallel) applies each agent's
   expertise independently. Output: `AgentAssessment`.
3. **`round1_complete`** barrier synchronises all Round-1 outputs.
4. **`moderator`** examines the five Round-1 assessments and produces a
   `ModeratorDecision` (contradictions found, pairings opened,
   terminate yes/no).
5. **`debate_round`** runs every opened pairing in parallel. Each
   pairing generates a list of `DebateMessage` entries (position,
   argument, revised-flag, confidence). Output: `DebatePairing`.
6. Back to `moderator` for another round, or forward to `judge_final`
   if terminate.
7. **`judge_final`** receives Round-1 assessments + every pairing's
   full exchanges + moderator decisions and emits the final country +
   coordinates + reasoning.

### End-of-run outputs

- `round_1_assessments`: per-agent `AgentAssessment` from Round 1.
- `debate.total_rounds`: number of debate rounds triggered.
- `debate.moderator_decisions`: full trace of moderator judgements
  (contradictions, pairings, reasoning).
- `debate.pairings`: every pairing with its full back-and-forth
  exchange history.
- `country_result`: raw judge text.
- `coordinates`: structured `{"lat", "lng"}` or `null`.
- `final_reasoning`: judge thinking chain.
- `error`: optional; set only on hard judge failure.

---

## 2. Directory layout

```
.
├── vlm_council/               # Python package (pipeline + agents)
│   ├── agents/                # 5 specialists + judge (with moderator)
│   ├── graph.py               # LangGraph pipeline (Debate)
│   ├── batch.py               # Batch processor with resume + --file-list
│   ├── config.py              # Env-var-based configuration
│   ├── state.py               # LangGraph state (round 1, debate, final)
│   ├── coordinates.py         # Shared judge-output coordinate parser
│   ├── llm.py                 # vLLM / OpenAI-compatible client
│   ├── image_utils.py         # base64 encoding for the VLM
│   ├── run.py                 # Single-image CLI
│   ├── evaluate.py            # Stage-1 evaluation (accuracy + debate stats)
│   ├── analyze_rounds.py      # Round-1 vs Debate analytics
│   └── plot_analysis.py       # Debate-specific visualisations
├── eval_debate/               # LLM-as-a-Judge evaluation (Stage 2, Qwen 3.6)
│   └── cluster_results/       # Cluster-run eval outputs (report.md/html, plots, judge/)
├── slurm/                     # bwUniCluster 3 launch scripts
│   ├── run_council_uc3.sh
│   ├── launch_short_uc3.sh
│   ├── run_eval_uc3.sh
│   ├── run_judge_uc3.sh
│   └── launch_judge_short_uc3.sh
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
the shared `datasets` workspace on the cluster, and from then on
**every approach reuses the same dataset via symlinks**. There is no
need to re-download or re-upload the images for the Debate,
Hub-and-Spoke, Progressive Narrowing, or Global Context Re-guess
approaches: each approach workspace just points at the same `datasets`
workspace with `ln -s`.

The download script `scripts/download_dataset.py` is bundled with
every approach for self-containedness, but the approaches ship
**identical copies** of it. Running it once from any one approach
folder is enough. **If the dataset is already present in the shared
`datasets` workspace (because another approach has already prepared
it), skip this section entirely** and go straight to *Recommended
workspace layout* below.

```bash
# On your local machine, one-time only (from ANY approach folder)
mkdir -p dataset
python3 scripts/download_dataset.py --output-dir dataset/Images --workers 16
cp /path/to/georc_locations.csv dataset/

# Rsync into the shared datasets workspace (one-time only)
rsync -avh --progress dataset/ <user>@uc3.scc.kit.edu:$(ssh <user>@uc3.scc.kit.edu 'ws_find datasets')/
```

### Recommended workspace layout on bwUniCluster 3

| Workspace | Purpose | Shared? |
|---|---|---|
| `vlm-council-debate` | Repo checkout + `results/` + `.venv` for this approach | No |
| `hf-cache` | HuggingFace model cache | Yes, reused across approaches |
| `datasets` | Images + `georc_locations.csv` | Yes, reused across approaches |

### On bwUniCluster 3 (H100 80GB)

```bash
ws_allocate datasets 60
ws_allocate hf-cache 60
ws_allocate vlm-council-debate 30

cd $(ws_find vlm-council-debate)

# Upload the project files into the workspace, then symlink the dataset
ln -s "$(ws_find datasets)/Images" Images
ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv

bash slurm/setup_uc3.sh
```

---

## 4. Running the pipeline

### Cluster: single job

```bash
cd $(ws_find vlm-council-debate)
sbatch slurm/run_council_uc3.sh
```

Resume-capable.

### Cluster: parallel jobs

```bash
bash slurm/launch_short_uc3.sh 5 10
```

---

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VLM_MODEL` | `google/gemma-4-31b-it` | Vision model (all agents + moderator + judge) |
| `VLM_JUDGE_THINKING` | `true` | Judge in thinking mode |
| `DEBATE_MAX_ROUNDS` | `3` | Max debate rounds triggered by moderator |
| `DEBATE_MAX_EXCHANGES` | `6` | Max exchanges per single pairing |
| `DEBATE_MIN_CONFIDENCE` | `medium` | Moderator opens pairings only for agents at least this confident |
| `VLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLM_MAX_MODEL_LEN` | `16384` | Max context length |
| `VLM_GPU_MEMORY_UTIL` | `0.9` | vLLM GPU utilisation |
| `VLM_IMAGE_TOKEN_BUDGET` | `1120` | Per-image token budget for vLLM multi-modal |
| `VLM_CALL_TIMEOUT` | `600` | Per-VLM-call timeout (seconds) |
| `VLM_OUTPUT_DIR` | `results` | Output directory |

Tested model (council): `google/gemma-4-31b-it`
Tested judge (eval Stage 2): `Qwen/Qwen3.6-35B-A3B-FP8`

---

## 6. Output format

For every image `<output_dir>/<image-stem>/result.json` is written:

```json
{
  "image_path": "...",
  "model": "google/gemma-4-31b-it",
  "judge_thinking": true,
  "image_token_budget": 1120,
  "debate_max_rounds": 3,
  "debate_max_exchanges": 6,
  "debate_min_confidence": "medium",
  "timing": { "total_seconds": 82.3 },
  "round_1_assessments": {
    "linguistic": { "agent_name": "linguistic", "candidates": [...], "evidence": [...] },
    ...
  },
  "debate": {
    "total_rounds": 2,
    "moderator_decisions": [
      { "debate_round": 1, "contradictions_found": [...], "pairings_opened": [...], "reasoning": "...", "terminate": false }
    ],
    "pairings": [
      {
        "debate_round": 1, "agent_a": "linguistic", "agent_b": "regulatory",
        "agent_a_initial_position": "Russia", "agent_b_initial_position": "Azerbaijan",
        "exchanges": [
          { "agent_name": "linguistic", "position": "Russia", "revised": false, "confidence": "high", "argument": "...", "key_evidence": [...] },
          { "agent_name": "regulatory", "position": "Azerbaijan", "revised": true, "confidence": "medium", "argument": "...", "key_evidence": [...] }
        ]
      }
    ]
  },
  "country_result": "Country: Azerbaijan\nCoordinates: 40.4, 49.9\nReasoning: ...",
  "coordinates": { "lat": 40.4, "lng": 49.9 },
  "final_reasoning": "..."
}
```

### `coordinates` field

- **Success:** `{"lat": <float>, "lng": <float>}`, values in valid
  geographic range.
- **Judge failure or unparseable output:** `null`, plus optional
  top-level `"error"` field. No `(0, 0)` fake fallback.

---

## 7. Results

The 500-image benchmark ran on Gemma 4 31B IT with judge thinking
enabled. Full evaluation artefacts are stored in two locations:

- `eval_debate/cluster_results/` (inside this approach folder)
- `../results-overview/VLM Debate/evaluation/` (consolidated across approaches)

Both trees contain the same LLM-as-Judge (Qwen 3.6) evaluation
artefacts:

- `report.md`, `report.html`: evaluation report
- `judge_summary.json`: machine-readable metrics
- `plots/`
- `judge/`: 500 per-image LLM-as-judge verdicts

The council-run outputs (500 `result.json` files) are mirrored under
`../results-overview/VLM Debate/council_run/`.

**Headline numbers (500 images, Debate, Gemma 4 31B IT):**

| Metric | Value |
|---|---|
| Country accuracy | 63.0 % (315 / 500) |
| Neighbor hits | 17.2 % (86 / 500) |
| Country or Neighbor | 80.2 % (401 / 500) |
| Median haversine error | 435 km |
| Mean haversine error | 1 568 km |

**Debate activation:**

| Debate rounds triggered | Images | Share |
|---|---|---|
| 0 (moderator resolved immediately) | 413 | 83 % |
| 1 | 66 | 13 % |
| 2 | 16 | 3 % |
| 3 (max reached) | 5 | 1 % |

**Debate impact:**

- Total exchanges across all debates: 500
- Revised positions: 127 (25 % of exchanges resulted in an agent
  changing its stance)

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
