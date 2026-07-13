import json
from pathlib import Path
from fastapi import APIRouter
from models.api import CouncilInfo

router = APIRouter(prefix="/api/council", tags=["council"])

_CONFIG_PATH = Path(__file__).parent.parent / "data" / "council_config.json"


@router.get("/agents", response_model=CouncilInfo)
async def get_agents():
    """Return the static council metadata.

    Loads `data/council_config.json` from disk and returns it as a
    `CouncilInfo` object (5 agent profiles + 6 collaboration steps). This
    endpoint powers the `/council` page on the frontend and does not
    touch the LLM cluster — safe to call at any time.
    """
    with open(_CONFIG_PATH) as f:
        return CouncilInfo.model_validate(json.load(f))
