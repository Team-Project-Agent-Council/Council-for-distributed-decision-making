# GeoBench Backend: FastAPI

The FastAPI service driving the Progressive Narrowing visualisation.
Wraps the vendored `vlm_council` LangGraph pipeline and streams progress
to the frontend via Server-Sent Events.

For the wider project context (Mannheim team project on multi-LLM
councils, cluster setup, SSH tunnelling), see the
[top-level README](../README.md).

---

## Stack

- Python 3.13+
- FastAPI 0.135.1, Pydantic v2
- Uvicorn with hot-reload
- LangGraph pipeline through the vendored `vlm_council` package
- Talks to a remote vLLM OpenAI-compatible endpoint (typically an
  SSH-tunnelled port from a GPU cluster)
- No database. The demo is fully in-memory per run.

## What it exposes

| Prefix | Purpose |
|---|---|
| `/api/council/*` | Static metadata (5 agent profiles + 6 collaboration steps) served from `data/council_config.json`. Backs the `/council` page. |
| `/api/demo/*` | Progressive Narrowing runs. Start a run, stream SSE events, look up dataset images. Run state lives in `services/demo_service.py` as an in-memory dict keyed by `runId`. |

## Running

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

Sanity checks:

```bash
curl -s http://localhost:8000/health                     # -> {"status":"ok"}
curl -s http://localhost:8000/api/council/agents | jq
```

## Environment variables

See [`.env.example`](./.env.example) for the full documented set. The
essentials:

```env
# CORS origins the API accepts (include every port the frontend uses).
CORS_ORIGINS=["http://localhost:3000"]

# vLLM endpoint. Typically the SSH-tunnelled local port from the cluster.
VLM_API_BASE=http://localhost:8001/v1
VLM_MODEL=google/gemma-4-31b-it
VLM_JUDGE_MODEL=google/gemma-4-31b-it
VLM_JUDGE_THINKING=true
VLM_MAX_MODEL_LEN=16384
VLM_GPU_MEMORY_UTIL=0.9
VLM_CALL_TIMEOUT=600

# Optional: absolute path to a directory containing georc_locations.csv
# and the referenced image files. Enables the "Random from dataset"
# button on /test-council. The repo does not ship a bundled dataset -
# download it from the project OneDrive (see the top-level README) and
# point this variable at the extracted folder.
# DEMO_DATASET_DIR=/absolute/path/to/VLM-Council/Images
```

## SSE event stream

Clients subscribe to `GET /api/demo/runs/{runId}/events` and receive a
well-defined sequence:

```
run_started
  -> phase1_started
  -> agent_assessment × 5                        # specialists in parallel
  -> region_consensus_result
    -> (Path A) skip to country_hypotheses_generated
    -> (Path B) region_hypotheses_generated
                -> region_evaluation × N×5
                -> region_evaluation_complete
                -> region_decision
                -> country_assessment × 5
  -> country_hypotheses_generated
  -> country_evaluation × N×5
  -> country_evaluation_complete
  -> final_started
  -> final_result
  -> done
```

An `error` event may replace any of the above mid-stream. The exact
payload shape for each event type is defined in `services/api/types.ts`
(frontend) and emitted by
`council_adapters/progressive_narrowing_adapter.py`.

## File map

| Path | Purpose |
|------|---------|
| `main.py` | FastAPI app: load `.env`, CORS, register `council` + `demo` routers |
| `config.py` | Pydantic `BaseSettings` for the CORS + service-level config |
| `models/api.py` | `AgentProfile`, `CollaborationStep`, `CouncilInfo` |
| `routers/council.py` | `GET /api/council/agents`, serves the JSON config |
| `routers/demo.py` | `POST /api/demo/run`, SSE stream, dataset endpoints |
| `services/demo_service.py` | Run-lifecycle manager + per-run `asyncio.Queue` + TTL cleanup + dataset CSV loader |
| `council_adapters/progressive_narrowing_adapter.py` | Wraps the vendored LangGraph pipeline; per-agent SSE event emission via `evaluate_hypotheses` patch + rescue parser for malformed Gemma JSON + country alias canonicalisation |
| `vendor/vlm_council/` | Vendored from upstream. Do not edit. |
| `data/council_config.json` | The 5 PN specialist agents + 6 collaboration steps |
| `data/country_centroids.json` | 200 countries to (lat, lng) for SSE event payloads |

## Adapter architecture

The demo adapter in
`council_adapters/progressive_narrowing_adapter.py` wraps the vendored
`vlm_council` LangGraph pipeline; it never modifies the vendored code
directly.

Three monkey-patches are applied per run and restored on teardown:

1. **`evaluate_hypotheses`** on each of the five agent modules, wraps
   the call so we can emit a `region_evaluation` / `country_evaluation`
   SSE event as soon as each agent returns, without waiting for the
   whole node's `asyncio.gather` to finish.
2. **`country_hypotheses_node`** on `vlm_graph`, canonicalises country
   name aliases (USA / United States / U.S. -> one entry) before the
   judge builds hypotheses, so duplicates do not clutter the matrix.
3. **Rescue parser** for `evaluate_hypotheses` output. The vendored
   strict parser expects a JSON array with `hypothesis_id` +
   `confidence` on every item. Gemma occasionally returns
   `{"evaluations": [...]}` or even reverts to the Phase-1
   `{"candidates": [...]}` format. The rescue parser tolerates both and
   synthesises `strongly_support` / `contradicts` verdicts from the
   Phase-1 candidate list when needed.

## Manual smoke test

Requires the SSH tunnel to the cluster to be up (see the
[top-level README](../README.md#3-vllm-on-the-cluster)).

```bash
# Verify the tunnel:
curl -s http://localhost:8001/v1/models | jq

# Start a run against a Street View sample. Point at any image you have
# on disk - either from your DEMO_DATASET_DIR or an ad-hoc file.
RUN_ID=$(curl -s -F image=@"$DEMO_DATASET_DIR/6fGwHxCTvCbaK77Q_5.png" \
  http://localhost:8000/api/demo/run | jq -r .runId)

# Stream events:
curl -N http://localhost:8000/api/demo/runs/$RUN_ID/events
```

## Cluster setup

The vLLM endpoint runs on a bwUniCluster 3 H100 GPU node. Two references:

- The [top-level README](../README.md#3-vllm-on-the-cluster), condensed
  quick-start with the full `sbatch` command
- [`CLUSTER_SETUP.md`](./CLUSTER_SETUP.md), step-by-step walkthrough
