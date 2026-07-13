# VLM Initial Approach

The *Initial Approach* to GeoRC country prediction is a **two-stage
system** that combines a grounded vision pipeline with a multi-agent
LLM council:

1. **Vision Pipeline** - a LangGraph pipeline that takes a raw Street
   View image, produces a rich structured scene description, identifies
   notable visual details, extracts them with grounded bounding-boxes,
   crops them, and generates a focused description of each crop.
2. **LLM Council** - a downstream multi-agent council that consumes the
   vision output (scene description + focused crop descriptions) to
   reason about the country. **Four variants** of the council were
   evaluated head-to-head.

Both stages run on **Ollama** with vision-language and text models. The
system was evaluated on the **first 100 images** of the GeoRC
benchmark (not the 500 used by the later approaches).

**Headline (best variant): Optimized Prompt Setup - 23 % top-1 country
accuracy (23 / 100)**, ~6035 km median distance error.

---

## 1. Approach

### Stage 1 - Vision Pipeline

```
prepare_image
    -> scene_parser        (Qwen3-VL 32B - dense, structured scene description)
    -> detail_identifier   (Gemma 4 26B - which details deserve a closer look?)
    -> [conditional: has_details?]
       yes -> detail_extractor  (Qwen 2.5 VL 32B - grounded bounding-boxes)
           -> crop_tool          (PIL crops each detail bbox)
           -> detail_focusser    (Qwen3-VL 32B - detailed description per crop)
           -> END
       no  -> END
```

The vision pipeline does **not** emit a country prediction or
coordinates. It only *sees* and *describes*. Its JSON output (scene
description + per-crop focused description) feeds every council
variant.

| Node | Model | Task |
|---|---|---|
| `scene_parser` | Qwen3-VL 32B | 400-800-word structured scene description (text/signs, infrastructure, vehicles, road surface, buildings, vegetation). |
| `detail_identifier` | Gemma 4 26B | Pick the top-K notable details worth zooming into (name + reason). |
| `detail_extractor` | Qwen 2.5 VL 32B | Per-detail normalized bounding box `[x1, y1, x2, y2]`. Deduplication + validation drops degenerate boxes. |
| `crop_tool` | Pillow | Crop each valid bbox and save `<crop_dir>/<name>.png`. |
| `detail_focusser` | Qwen3-VL 32B | Fine-grained per-crop description (material, colour, text, condition, mounting). |

### Stage 2 - LLM Council (4 variants)

All variants use the same orchestrator + judge scaffold. They differ
in **which agents are wired**, **how they are coordinated**, and **how
their prompts are set up**.

```
                               orchestrator
                                     |
                    +----------------+----------------+
                    |   (parallel specialist agents)  |
                    +----------------+----------------+
                                     |
                                   judge
                                     |
                              (country, coords)
```

| Variant | Vision input | Agents (specialists) | Coordination |
|---|---|---|---|
| **Initial Council** | granular `vision_pipeline` (via thin `vision_agent` wrapper -> emits `general_description` + `crop_descriptions[]` into council state) | linguistic, landscape, botanics, regulatory, meta (5) | Parallel fan-out -> judge |
| **+ Additional Agents (Climate & Infrastructure)** | granular `vision_pipeline` (same thin wrapper) | + climate, + infrastructure (7) | Parallel fan-out -> judge |
| **+ Optimized Prompt Setup (selected Tools)** | granular `vision_pipeline` integrated into the council graph as individual nodes | + cultural, + infrastructure (7); revised prompts + curated tool set (geocode, wikidata, biodiversity, websearch) | Parallel fan-out -> judge |
| **+ Hub and Spoke** | granular `vision_pipeline` integrated | + cultural, + infrastructure (7); adds `followup_dispatch` + `judge_deliberation` | Sequential Hub coordinator dispatches each Spoke, iterative follow-ups |

**All four variants consume the exact same granular vision pipeline**
(`vision_pipeline/`, the as-tested `f1dacac`-era version that produced
the 100 `result.json` shipped under `results/`). Variants 1 and 2 wrap
the pipeline in a small `vision_agent.py` shim so their council state
schema (`general_description` + `crop_descriptions[]`) is preserved;
variants 3 and 4 wire the pipeline nodes into the council graph
directly. In evaluation mode (via `evaluate.py`) all variants skip the
vision stage and read the pre-computed `result.json`, so the reported
CSV numbers reproduce exactly.

All variants call the judge (Qwen 3 32B, thinking mode) once with all
specialist outputs and expect: `Country: <X>\nCoordinates: <lat>, <lng>\nReasoning: ...`.

### End-of-run outputs

Per image, per variant:

- **Vision Pipeline**: `results/<image-stem>/result.json` -
  `scene_description`, `details[]` (with `bbox`, `crop_path`,
  `focused_description`), `errors[]`, `bboxes.png`, `crops/*.png`.
- **LLM Council**: `results/llm_council_evals/<variant>.csv` -
  `location_id, gt_country, predicted_country, country_match, dist_km,
  geoguessr_score, pred_lat, pred_lon, error` per image.

---

## 2. Directory layout

```
.
├── vision_pipeline/                # Stage 1: LangGraph vision pipeline
│   ├── graph.py                    #   pipeline wiring
│   ├── state.py                    #   PipelineState + Detail TypedDicts
│   ├── config.py                   #   env-var-based configuration
│   ├── scene_parser.py             #   node: dense scene description
│   ├── detail_identifier.py        #   node: pick notable details
│   ├── detail_extractor.py         #   node: bbox extraction (grounding.py)
│   ├── grounding.py                #   Qwen 2.5 VL bbox extractor
│   ├── florence.py                 #   alternate grounder (Florence-2)
│   ├── crop_tool.py                #   node: pixel crops
│   ├── detail_focusser.py          #   node: per-crop description
│   ├── ollama_client.py
│   ├── visualize.py                #   bbox overlay renderer
│   ├── image_utils.py
│   ├── batch.py                    #   batch runner (resume, --limit)
│   └── __main__.py                 #   single-image CLI
# Stage 2: four self-contained council variants (each its own mini-repo:
# council/ package + evaluate.py + evaluation/ + geoguessr_rag + run scripts).
# All four consume the shared vision_pipeline/ output above.
├── Initial Council/                # Variant 1: 5 agents baseline (16 % / 100)
│   ├── council/                    #   linguistic, landscape, botanics, regulatory, meta + judge
│   ├── evaluation/                 #   variant-local eval runner (loader, runner, metrics, report, tracer)
│   ├── evaluate.py                 #   variant-local Stage-2 CLI
│   └── README.md
├── Additional Agents/              # Variant 2: + climate + infrastructure (13 % / 100)
│   ├── council/                    #   7 specialists + judge
│   ├── evaluation/
│   ├── evaluate.py
│   └── README.md
├── Optimized Prompts/              # Variant 3: + cultural, revised prompts (best, 23 % / 100)
│   ├── council/                    #   7 specialists + judge, curated tool set
│   ├── evaluation/
│   ├── evaluate.py
│   └── README.md
├── Hub and Spoke/                  # Variant 4: sequential Hub coordinator (16 % / 100)
│   ├── council/                    #   7 specialists + judge + followup_dispatch/judge_deliberation loop
│   ├── evaluation/
│   ├── evaluate.py
│   └── README.md
├── scripts/
│   └── download_dataset.py         # local GeoRC downloader (run once)
├── slurm/
│   └── run_vision_uc3.sh           # Stage-1 SLURM launcher
├── results/
│   ├── 1NJsXTxIF9GGMDxC_1/         # per-image vision output (100 folders)
│   │   ├── result.json
│   │   ├── bboxes.png
│   │   └── crops/*.png
│   ├── bbox_overview/*.png         # mirror of all bbox overlays (81 images)
│   └── llm_council_evals/          # per-variant CSV + overview PNG
│       ├── initial_council.csv
│       ├── additional_agents.csv
│       ├── optimized_prompts.csv
│       ├── hub_and_spoke.csv
│       ├── initial_council.png
│       ├── additional_agents.png
│       ├── optimized_prompts.png
│       ├── hub_and_spoke.png
│       └── georc_locations.csv     # ground truth
├── 01_imgs/                        # sample streetview images
├── plonkit_meta.json               # RAG country metadata used by meta/cultural
├── example.env
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

The Initial Approach was evaluated on the **first 100 images** of the
GeoRC benchmark (the same benchmark the later approaches use with 500
images).

### On bwUniCluster 3 (H100 80GB)

```bash
ws_allocate datasets 60
ws_allocate hf-cache 60
ws_allocate vlm-initial 30

cd $(ws_find vlm-initial)

ln -s "$(ws_find datasets)/Images" Images
ln -s "$(ws_find datasets)/georc_locations.csv" georc_locations.csv

uv venv && source .venv/bin/activate
uv pip install -e .
```

### Ollama setup on the compute node

```bash
export PATH=$(ws_find ollama_models)/bin:$PATH

OLLAMA_HOST=0.0.0.0 \
OLLAMA_MAX_LOADED_MODELS=3 \
OLLAMA_NUM_PARALLEL=4 \
ollama serve &
sleep 10

ollama pull qwen3-vl:32b
ollama pull gemma4:26b
ollama pull qwen2.5vl:32b
ollama pull qwen3:32b        # judge + specialists
```

---

## 4. Running the pipeline

### Stage 1 - vision pipeline (produces the per-image JSON that every variant consumes)

```bash
cd $(ws_find vlm-initial)
sbatch slurm/run_vision_uc3.sh
```

### Stage 2 - one of the four councils

Every variant is a **self-contained sub-repo** in its own folder and
consumes the same `results/<image>/result.json` files that Stage 1
produced. There is no variant switch: you `cd` into the variant you want
and run its own `evaluate.py`. Each writes its CSV into the shared
`../results/llm_council_evals/`.

```bash
# Variant 1 - baseline (5 agents)
cd "Initial Council"
python evaluate.py \
  --results-dir ../results/ \
  --mapping ../results/llm_council_evals/location-image-mapping.csv \
  --output ../results/llm_council_evals/initial_council.csv

# Variant 2 - + climate + infrastructure
cd "../Additional Agents"
python evaluate.py \
  --results-dir ../results/ \
  --mapping ../results/llm_council_evals/location-image-mapping.csv \
  --output ../results/llm_council_evals/additional_agents.csv

# Variant 3 - optimized prompts + selected tools (best)
cd "../Optimized Prompts"
python evaluate.py \
  --results-dir ../results/ \
  --mapping ../results/llm_council_evals/location-image-mapping.csv \
  --output ../results/llm_council_evals/optimized_prompts.csv

# Variant 4 - hub-and-spoke coordinator
cd "../Hub and Spoke"
python evaluate.py \
  --results-dir ../results/ \
  --mapping ../results/llm_council_evals/location-image-mapping.csv \
  --output ../results/llm_council_evals/hub_and_spoke.csv
```

Each variant folder also ships its own SLURM launchers
(`run_eval_gpu.sh`, `run_georc_test_gpu.sh`) for the cluster.

`--skip-council` on any variant's `evaluate.py` skips the LLM calls and
just regenerates the CSV from an existing per-image council result.

---

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint (via SSH tunnel or same node) |
| `VISION_MODEL` | `qwen3-vl:32b` | Scene parser + focusser |
| `TEXT_MODEL` | `gemma4:26b` | Detail identifier |
| `GROUNDING_MODEL` | `qwen2.5vl:32b` | Bbox extractor |
| `COUNCIL_MODEL` | `qwen3:32b` | All council specialist agents + judge |
| `MAX_DETAILS` | `5` | Cap on details per image |
| `META_THINK_REASON` | `true` | `/think` on meta / cultural reasoning step |
| `JUDGE_THINK_TOOL` | `true` | `/think` on judge decision |
| `META_THINK_RUN` | `false` | `/no_think` on meta tool-dispatch (fast) |
| `JUDGE_THINK_RUN` | `false` | `/no_think` on judge tool-dispatch (fast) |

---

## 6. Output format

### Vision pipeline (per image)

```json
{
  "image_path": "/pfs/.../Images/1NJsXTxIF9GGMDxC_1.png",
  "scene_description": "### Foreground to Background Description ...",
  "detected_objects": [],
  "details": [
    {
      "name": "green sign",
      "reason": "The Cyrillic text 'АЗС' ...",
      "bbox": [0.343, 0.401, 0.386, 0.650],
      "crop_path": "results/1NJsXTxIF9GGMDxC_1/crops/green_sign.png",
      "focused_description": "The green sign is a tall, rectangular ..."
    }
  ],
  "has_details": true,
  "errors": []
}
```

### Council eval (per variant CSV)

```
location_id,gt_country,predicted_country,country_match,dist_km,geoguessr_score,pred_lat,pred_lon,error
1NJsXTxIF9GGMDxC_1,Kyrgyzstan,Russia,False,2935.8,1152,57.5,41.0,
```

---

## 7. Results

The Initial Approach was evaluated on **100 images** from GeoRC. Full
per-image vision outputs are stored in two locations:

- `results/` (inside this approach folder)
- `../results-overview/VLM Initial Approach/vision_pipeline_run/`

Council-eval CSVs (one per variant):

- `results/llm_council_evals/` (inside this approach folder)
- `../results-overview/VLM Initial Approach/llm_council_evals/` (consolidated)

**Vision-pipeline coverage (100 images):**

| Bucket | Count | Share |
|---|---|---|
| Images with >=1 identified detail | 82 | 82 % |
| Images with 0 identified details (short-circuit) | 18 | 18 % |
| Total identified details | 273 | mean 2.73 / image |

**Council head-to-head (numbers computed from the CSVs shipped
alongside the code):**

| # | Council Variant | Country Accuracy | Median km | Mean km |
|---|---|---|---|---|
| 1 | Initial Council | **16 / 100 = 16.0 %** | 6560 | 6315 |
| 2 | + Additional Agents (Climate & Infra) | **13 / 100 = 13.0 %** | 7489 | 6852 |
| 3 | + Optimized Prompt Setup (selected Tools) | **23 / 100 = 23.0 %** | 6035 | 6618 |
| 4 | + Hub and Spoke | **16 / 100 = 16.0 %** | 7277 | 7116 |

**Key finding: optimized prompts + a curated tool set beat all
architectural variants** on this benchmark. Adding climate +
infrastructure agents *reduced* accuracy (13 % vs 16 %), and switching
to a Hub-and-Spoke coordinator gave no gain over the parallel-fan-out
baseline (16 % vs 16 %). The single largest improvement came from
prompt engineering, not agent-graph topology.

The country accuracy numbers here are **substantially lower than the
later approaches** because the Initial Approach:

- Runs on a smaller model family (Qwen 3 32B judge, Gemma 4 26B for
  identifier) vs Gemma 4 31B IT + Qwen 3.6 35B in the later runs.
- Uses only per-crop text descriptions rather than the raw image
  through a VLM council.
- Has no tournament / progressive-narrowing structure.

These insights motivated the redesign into the pure-VLM council
architectures documented in the sister-approach folders.

---

## 8. Dependencies

See `pyproject.toml` and `requirements.txt`. Core stack:

- Python 3.11+
- LangGraph + LangChain-Ollama
- Ollama (with `qwen3-vl:32b`, `gemma4:26b`, `qwen2.5vl:32b`,
  `qwen3:32b` pulled)
- Pillow (image cropping + bbox overlays)
- FastAPI + ChromaDB (RAG server for the meta / cultural agent)
