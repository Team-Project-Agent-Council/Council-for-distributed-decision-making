from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    council_adapter: Literal["mock", "live"] = "mock"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    total_rounds: int = 5
    mongodb_url: str = ""
    google_maps_api_key: str = ""

    # `extra="ignore"` so unrelated env vars (e.g. VLM_API_BASE for the
    # vendored vlm_council package, which reads os.environ directly) don't
    # cause Settings validation to fail. Those keys are still loaded into
    # os.environ by load_dotenv() in main.py and consumed by the vendored
    # modules; Pydantic just doesn't complain about them.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
