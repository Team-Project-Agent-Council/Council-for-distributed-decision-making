# VLM Baseline

A single-model baseline for the GeoRC geolocation benchmark: the image and a
single prompt go straight to one Vision-Language Model, which returns one
answer. No council, no specialist agents, no judge. It exists purely as the
reference point the multi-agent VLM Council approaches are compared against.

**Headline: 65.0 % top-1 country accuracy (325 / 500)** on the 500-image GeoRC
benchmark with Gemma 4, matching the strongest council variants and confirming
that multi-agent orchestration does not, on this benchmark, beat a well-prompted
single model on raw localization accuracy.

---

## 1. Approach

The baseline sends the street-view image plus one geolocation prompt to a single
VLM and parses `Country`, `Coordinates` and `Reasoning` from the response. It is
deliberately minimal so that any accuracy difference against the council
approaches can be attributed to the coordination mechanism rather than to a
different base model or prompt budget.

---

## 2. Directory layout

```
.
├── baseline_eval.py           # Single-image + batch runner (one VLM call per image)
├── council/                   # Shared VLM client + prompt/parse helpers
├── evaluation/                # Stage-1 evaluation (country accuracy + distance to GT)
├── Images/                    # georc_locations.csv ground truth (+ downloaded PNGs)
├── results/                   # Per-image result.json (500 images)
├── scripts/
│   └── download_dataset.py    # Local dataset downloader (identical across approaches)
├── slurm/                     # bwUniCluster 3 launch + eval scripts
├── pyproject.toml
└── requirements.txt
```

---

## 3. Setup

```bash
# Preparing the dataset (once, reused across all approaches)
python3 scripts/download_dataset.py --output-dir Images

# Dependencies
pip install -r requirements.txt          # runtime
# or, as an editable install with the optional dev extra:
pip install -e ".[dev]"
```

The ground-truth CSV (`Images/georc_locations.csv`) is the same 500-image GeoRC
set used by every approach.

---

## 4. Running

```bash
# Single image (debug)
python baseline_eval.py 74bPHM081cMUaNKT_4 --verbose

# Full 500-image batch
python baseline_eval.py --all --concurrency 1

# On bwUniCluster 3 (H100)
sbatch slurm/run_baseline_gemma4.sh
```

Evaluation of the produced `results/` against the ground truth:

```bash
python -m evaluation ... --mapping Images/georc_locations.csv
# or on the cluster:
sbatch slurm/run_eval_gpu.sh
```

---

## 5. Results

**Headline numbers (500 images, single-model Gemma 4):**

| Metric | Value |
|---|---|
| Country accuracy | 65.0 % (325 / 500) |
| Correct or Neighbor | 80.0 % |
| Median haversine error | 384 km |
| Mean haversine error | 1 601 km |

These figures are the baseline row in the cross-approach results table of the
paper and the `results-overview` comparison.

---

## 6. Dependencies

See `pyproject.toml` and `requirements.txt`. Core stack: vLLM / OpenAI-compatible
client, LangChain (OpenAI / Ollama / Anthropic message wrappers), Click,
python-dotenv, pycountry (for country-name matching in the evaluation).
