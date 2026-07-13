"""Pipeline configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration.

    Defaults are tuned for local development via SSH tunnel.
    On the cluster, the SLURM script overrides via environment variables.
    """

    ollama_host: str

    vision_model: str
    text_model: str
    grounding_model: str

    # Pipeline behavior
    max_details: int
    crop_output_dir: str


def load_config() -> PipelineConfig:
    """Load configuration from environment variables with sensible defaults."""
    return PipelineConfig(
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        vision_model=os.environ.get("VISION_MODEL", "qwen3-vl:32b"),
        text_model=os.environ.get("TEXT_MODEL", "gemma4:26b"),
        grounding_model=os.environ.get("GROUNDING_MODEL", "qwen2.5vl:32b"),
        max_details=int(os.environ.get("MAX_DETAILS", "5")),
        crop_output_dir=os.environ.get("CROP_OUTPUT_DIR", "./crops"),
    )
