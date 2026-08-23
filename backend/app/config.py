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
    # Provider: Google Gemini via langchain-google-genai.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    # Optional per-tier overrides; both fall back to gemini_model.
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

    @property
    def master_model_id(self) -> str:
        return self.master_model or self.gemini_model

    @property
    def specialist_model_id(self) -> str:
        return self.specialist_model or self.gemini_model

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
