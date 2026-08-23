"""Central configuration. Paths are resolved relative to the repo root."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    # Both env files are read; backend/.env wins on conflicts, so a developer can
    # override the shared root .env locally.
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths -----------------------------------------------------------
    datasets_dir: Path = REPO_ROOT / "datasets"
    local_db_dir: Path = REPO_ROOT / "localDB"
    stage_dir: Path = REPO_ROOT / "stage"
    artifacts_dir: Path = BACKEND_DIR / "artifacts"
    duckdb_path: Path = BACKEND_DIR / "artifacts" / "transit_media.duckdb"

    # --- models ----------------------------------------------------------
    # Two providers, and the choice belongs to the rep rather than to a deploy-time env
    # var: Gemini's free tier allows ~20 requests/day/model and one orchestration costs
    # 15-20 of them, so a demo runs out of Gemini before it runs out of questions. The UI
    # picks per request; this is only the default when a request does not name one.
    # See app/agents/providers.py for the registry these settings feed.
    model_provider: str = "gemini"

    # Google Gemini via langchain-google-genai.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    #: Everything the UI may offer on this provider. `gemini_model` is always included.
    gemini_models: list[str] = ["gemini-3.5-flash-lite"]

    # Azure OpenAI via langchain-openai (AzureChatOpenAI). The key/endpoint/version come
    # from the hackathon provider.
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    #: model name the human picks -> the DEPLOYMENT name the API wants. They are not the
    #: same string on Azure, and sending the model name gets a 404 rather than a useful
    #: error, so the mapping lives here and is applied in exactly one place.
    azure_openai_deployments: dict[str, str] = {
        "gpt-5.4-nano": "Team7-GPT-5.4-nano-39a7f0abb4d54f9c265d",
        "gpt-5.4-mini": "Pod3-GPT-5.4-mini-1028149d34b55e617df4",
    }
    azure_openai_model: str = "gpt-5.4-mini"

    # Optional per-tier overrides. Honoured only when the id names a model the selected
    # provider actually offers, so a MASTER_MODEL left pointing at Gemini does not follow
    # the rep across to Azure and 404 there. An explicit choice from the UI wins over both.
    master_model: str | None = None
    specialist_model: str | None = None

    # Fail fast instead of backing off forever when the model is saturated. Gemini
    # returns 503 UNAVAILABLE / 504 DEADLINE_EXCEEDED under load, and the client's
    # default retry policy makes that look like a hang.
    request_timeout_seconds: float = 90.0
    model_max_retries: int = 3

    # Free-tier Gemini enforces a per-MINUTE request cap (gemini-3.5-flash-lite: 15/min).
    # One orchestration makes ~20 model calls in ~20s, which blows straight through it, so
    # a client-side limiter is required rather than optional. 12/min leaves headroom.
    model_requests_per_minute: float = 12.0
    # Azure's deployments are provisioned per-minute in tokens rather than requests and
    # are far less tight than the Gemini free tier, so the limiter is kept (a burst of
    # ~20 tool-calling turns can still trip a small TPM quota) but set much higher.
    azure_requests_per_minute: float = 60.0

    # --- pipeline tuning -------------------------------------------------
    candidate_pool_size: int = 250
    max_screens_in_package: int = 120
    solver_time_limit_seconds: int = 30

    # --- api -------------------------------------------------------------
    # Both loopback spellings. The frontend now calls the API on 127.0.0.1 (see
    # frontend/src/lib/api.ts for why), and a developer who opens the page on
    # 127.0.0.1:3000 rather than localhost:3000 sends the matching Origin header.
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    def ensure_dirs(self) -> None:
        for d in (self.local_db_dir, self.stage_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
