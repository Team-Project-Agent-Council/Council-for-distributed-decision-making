from dotenv import load_dotenv

# Load .env into os.environ BEFORE any vendored modules import — the vendored
# vlm_council package reads VLM_* config directly via os.environ.get(...) from
# `vlm_council/config.py` and `vlm_council/llm.py`, not via Pydantic Settings,
# so the variables must be present in the process environment by the time the
# adapter is first imported.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import council, demo


app = FastAPI(title="GeoBench API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(council.router)
app.include_router(demo.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
