"""Request/response models for the HTTP layer. Mirror these in frontend/src/lib/types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str = "New Campaign"


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None


class CampaignQuery(BaseModel):
    """A natural-language brief, optionally continuing an existing session."""

    query: str = Field(min_length=1)
    session_id: str | None = None
    upload_ids: list[str] = Field(
        default_factory=list, description="Ids of documents staged via POST /uploads"
    )


class CampaignRunOut(BaseModel):
    session_id: str | None = None
    run_id: str | None = None
    answer: str
    stub_stages: list[str] = []
    provenance: str = "computed"
    run_state: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = Field(
        default=None, description="Token and call totals for this run, from AgentRunLogger"
    )


class ArtifactRowsOut(BaseModel):
    """Top-N rows of one run artifact. Backs the inspector panel; never agent context."""

    run_id: str
    kind: str
    artifact_id: str
    provenance: str = "computed"
    total_rows: int
    returned_rows: int
    columns: list[str] = []
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []


class UploadOut(BaseModel):
    id: str
    session_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int
    stored_path: str
