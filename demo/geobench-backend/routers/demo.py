"""Endpoints for the Progressive Narrowing demo.

  POST /api/demo/run
        multipart: either `image` (UploadFile) or `datasetId` (str form field)
        → { runId, imageUrl, groundTruth?: { lat, lng, label } }

  GET  /api/demo/runs/{runId}/events
        Server-Sent Events stream until `event: done`.

  GET  /api/demo/dataset/random
        → { datasetId, imageUrl, lat, lng, label }

  GET  /api/demo/runs/{runId}
        → { result: { country, lat, lng, reasoning } | null, error?, finished }
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile
from fastapi.responses import StreamingResponse

from services import demo_service


router = APIRouter(prefix="/api/demo", tags=["demo"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_image_url(dataset_id: str) -> str:
    """URL that the frontend can fetch to display the dataset image preview.

    We serve images straight from this router via the `/image/{id}` endpoint
    below, since the dataset lives on disk in `DEMO_DATASET_DIR`.
    """
    return f"/api/demo/dataset/image/{dataset_id}"


def _country_label(country_code: str) -> str:
    return country_code.upper() if country_code else "Unknown"


# ---------------------------------------------------------------------------
# POST /api/demo/run
# ---------------------------------------------------------------------------


@router.post("/run")
async def start_demo_run(
    image: Optional[UploadFile] = File(None),
    datasetId: Optional[str] = Form(None),
):
    """Start a new Progressive Narrowing run.

    Accepts either an uploaded image file (`image` multipart part) OR a
    reference to a bundled dataset entry (`datasetId` form field), but not
    both. Uploaded images are inlined into a `data:` URL for the frontend
    preview; dataset entries reference the stable
    `/api/demo/dataset/image/{id}` route instead.

    When a `datasetId` is used the response also includes `groundTruth`
    with the true lat/lng and country code, so the frontend can render the
    correct-answer pin alongside the council's guess.

    Response: `{ runId, imageUrl, groundTruth?: {lat, lng, label} }`.

    The run itself is dispatched to the vlm_council pipeline in the
    background — the client subscribes to progress via
    `GET /api/demo/runs/{runId}/events` and polls `GET /api/demo/runs/{runId}`
    for the final snapshot.

    Raises 400 if neither / both inputs are supplied or the upload isn't an
    image; 404 if `datasetId` doesn't resolve to a known dataset entry.
    """
    if image is None and not datasetId:
        raise HTTPException(
            status_code=400,
            detail="Provide either an `image` upload or a `datasetId` form field.",
        )

    if datasetId:
        entry = demo_service.find_dataset_entry(datasetId)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Unknown datasetId: {datasetId}")
        image_bytes, mime = demo_service.read_dataset_image(entry)
        ground_truth = {
            "lat": entry.lat,
            "lng": entry.lng,
            "label": _country_label(entry.country_code),
        }
        image_url = _dataset_image_url(entry.dataset_id)
    else:
        assert image is not None
        if not (image.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        image_bytes = await image.read()
        mime = image.content_type or "image/jpeg"
        ground_truth = None
        # Inline data URL — small enough for the typical Street View image
        # and avoids needing to keep the temp file alive for an extra
        # GET round-trip.
        import base64
        b64 = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{mime};base64,{b64}"

    state = await demo_service.start_run(
        image_bytes=image_bytes,
        mime=mime,
        image_url=image_url,
        ground_truth=ground_truth,
    )

    return {
        "runId": state.run_id,
        "imageUrl": state.image_url,
        **({"groundTruth": state.ground_truth} if state.ground_truth else {}),
    }


# ---------------------------------------------------------------------------
# GET /api/demo/runs/{runId}/events  (SSE)
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str = Path(...)):
    """Stream pipeline progress as Server-Sent Events.

    Opens a long-lived HTTP response and emits one SSE frame per event
    produced by the vlm_council pipeline. Event order follows the
    documented schema (`phase1_started → agent_assessment × 5 →
    region_consensus_result → …`, see the backend README) and terminates
    with an `event: done` frame. The client is expected to reconnect on
    error and can re-subscribe safely — new subscribers replay the buffered
    events from the run start.

    Raises 404 if the run doesn't exist. Response has `Cache-Control:
    no-cache` and `X-Accel-Buffering: no` so events flush through nginx /
    Cloudfront without buffering.
    """
    state = demo_service.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")

    async def event_stream():
        async for evt in demo_service.subscribe(run_id):
            event_type = evt["type"]
            data_json = json.dumps(evt)
            # SSE frame: `event:` + `data:` + blank line.
            yield f"event: {event_type}\ndata: {data_json}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Disable proxy buffering so events arrive incrementally even
            # behind nginx/Cloudfront.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/demo/runs/{runId}  (JSON snapshot)
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}")
async def get_run_snapshot(run_id: str = Path(...)):
    """Return a JSON snapshot of the current run state.

    Useful as a poll target for clients that can't consume the SSE stream,
    or as a final "did it succeed" check after `event: done` arrived over
    SSE. The `result` field is `null` until `final_result` is emitted; the
    `error` field is set if the pipeline failed mid-run.

    Response: `{ runId, imageUrl, groundTruth?, result?, error?, finished }`.
    Raises 404 if the run doesn't exist.
    """
    state = demo_service.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")
    return {
        "runId": state.run_id,
        "imageUrl": state.image_url,
        "groundTruth": state.ground_truth,
        "result": state.result,
        "error": state.error,
        "finished": state.finished_at is not None,
    }


# ---------------------------------------------------------------------------
# GET /api/demo/dataset/random
# ---------------------------------------------------------------------------


@router.get("/dataset/random")
async def random_dataset_entry():
    """Return a random image from the Street View dataset.

    Reads `georc_locations.csv` from `DEMO_DATASET_DIR` and picks a
    uniformly random entry. The `label` field contains the ground-
    truth country code so the frontend can later render the correct-answer
    pin.

    Response: `{ datasetId, imageUrl, lat, lng, label }`.

    Raises 404 with a helpful message if the dataset directory is empty
    or misconfigured.
    """
    entry = demo_service.get_random_dataset_entry()
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No dataset available. Set DEMO_DATASET_DIR to a folder "
                "containing georc_locations.csv + the referenced image files."
            ),
        )
    return {
        "datasetId": entry.dataset_id,
        "imageUrl": _dataset_image_url(entry.dataset_id),
        "lat": entry.lat,
        "lng": entry.lng,
        "label": _country_label(entry.country_code),
    }


# ---------------------------------------------------------------------------
# GET /api/demo/dataset/image/{id}
# ---------------------------------------------------------------------------


@router.get("/dataset/image/{dataset_id}")
async def get_dataset_image(dataset_id: str = Path(...)):
    """Serve the raw image bytes for a dataset entry.

    Streams the file from disk with `Cache-Control: public, max-age=3600`
    so browsers cache the Street View still and repeated calls to
    `/dataset/random` (which reuse dataset ids) don't re-transfer the same
    image. Media type is derived from the file extension.

    Raises 404 if `dataset_id` doesn't resolve to a known entry.
    """
    entry = demo_service.find_dataset_entry(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown datasetId: {dataset_id}")
    image_bytes, mime = demo_service.read_dataset_image(entry)
    return StreamingResponse(
        iter([image_bytes]),
        media_type=mime,
        headers={"Cache-Control": "public, max-age=3600"},
    )
