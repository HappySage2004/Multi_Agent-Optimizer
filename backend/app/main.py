"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import campaign, messages, sessions, uploads
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

app.include_router(sessions.router)
app.include_router(messages.router)
app.include_router(uploads.router)
app.include_router(campaign.router)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "tables": len(available_tables()),
        "ridership_actuals_provisioned": has_ridership_actuals(),
        "gemini_api_key_configured": bool(settings.gemini_api_key),
        "master_model": settings.master_model_id,
        "specialist_model": settings.specialist_model_id,
    }
