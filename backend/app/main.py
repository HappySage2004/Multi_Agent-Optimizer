"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents import providers
from app.api import campaign, messages, sessions, uploads
from app.api import providers as providers_api
from app.config import get_settings
from app.data.db import available_tables, has_ridership_actuals


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on a bad dataset path rather than at the first agent call.
    settings = get_settings()
    settings.ensure_dirs()
    available_tables()
    yield


app = FastAPI(
    title="Transit Media Campaign Recommendation System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(providers_api.router)
app.include_router(sessions.router)
app.include_router(messages.router)
app.include_router(uploads.router)
app.include_router(campaign.router)


@app.get("/health")
def health() -> dict:
    """Liveness plus the two things that silently degrade a run: a missing dataset and a
    provider with no credentials.

    `model_providers_configured` is a map rather than a single flag because either
    provider can be usable on its own — the sidebar warns only when *nothing* is wired up,
    and the settings dialog reads GET /models for the per-provider detail.
    """
    default_provider = providers.default_provider()
    try:
        selection = providers.resolve(default_provider)
        master, specialist = selection.master_model, selection.specialist_model
    except providers.UnknownModel:
        master = specialist = ""
    configured = {pid: providers.is_configured(pid) for pid in providers.PROVIDER_IDS}
    return {
        "status": "ok",
        "tables": len(available_tables()),
        "ridership_actuals_provisioned": has_ridership_actuals(),
        # Retained under its old name so an older frontend build keeps rendering.
        "gemini_api_key_configured": configured[providers.GEMINI],
        "model_providers_configured": configured,
        "any_model_provider_configured": any(configured.values()),
        "default_provider": default_provider,
        "master_model": master,
        "specialist_model": specialist,
    }
