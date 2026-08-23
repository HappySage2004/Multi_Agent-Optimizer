"""Request/response models for the HTTP layer. Mirror these in frontend/src/lib/types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.clarification import ClarificationRequest


class SessionCreate(BaseModel):
    title: str = "New Campaign"


class SessionUpdate(BaseModel):
    """Rename a session. The UI titles a session from the brief once one is submitted."""

    title: str = Field(min_length=1, max_length=120)


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None


class MessageCreate(BaseModel):
    """Append a message to a session's transcript.

    The campaign endpoints already persist both halves of a turn, so this exists for the
    client to record anything the server did not see -- and to keep the transcript a
    first-class CRUD resource rather than a side effect of running the pipeline.
    """

    role: Literal["user", "assistant"]
    text: str = ""
    run_id: str | None = None
    attachments: list[str] = Field(default_factory=list)
    pipeline_ran: bool | None = None
    tool_trail: list[str] = Field(default_factory=list)
    token_usage: dict[str, Any] | None = None


class MessageUpdate(BaseModel):
    """Amend a stored message. Every field is optional; unset fields are left alone.

    `session_id` is deliberately absent -- moving a message between sessions would reorder
    two transcripts at once.
    """

    text: str | None = None
    run_id: str | None = None
    attachments: list[str] | None = None
    pipeline_ran: bool | None = None
    tool_trail: list[str] | None = None
    token_usage: dict[str, Any] | None = None


class MessageOut(BaseModel):
    """One persisted chat message.

    An assistant message carries the turn's metadata as well as its prose: `run_id` is the
    package it reported on, and `pipeline_ran` distinguishes a rebuild from a follow-up
    answered off the existing package. The UI needs both to decide whether to render the
    metrics deck under this message.
    """

    id: str
    session_id: str
    created_at: str | None = None
    updated_at: str | None = None
    role: Literal["user", "assistant"]
    text: str = ""
    run_id: str | None = None
    attachments: list[str] = Field(default_factory=list)
    pipeline_ran: bool | None = None
    tool_trail: list[str] = Field(default_factory=list)
    token_usage: dict[str, Any] | None = None


class CampaignQuery(BaseModel):
    """A natural-language brief, optionally continuing an existing session."""

    query: str = Field(min_length=1)
    session_id: str | None = None
    upload_ids: list[str] = Field(
        default_factory=list, description="Ids of documents staged via POST /uploads"
    )


class CampaignRunOut(BaseModel):
    session_id: str | None = None
    session_title: str | None = Field(
        default=None, description="The session's title after this run named it from the brief"
    )
    message_id: str | None = Field(
        default=None, description="Id of the persisted assistant message for this turn"
    )
    run_id: str | None = None
    pipeline_ran: bool = Field(
        default=True,
        description="False when this turn answered from an existing package without rebuilding",
    )
    answer: str
    stub_stages: list[str] = []
    provenance: str = "computed"
    run_state: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = Field(
        default=None, description="Token and call totals for this run, from AgentRunLogger"
    )
    pending_questions: ClarificationRequest | None = Field(
        default=None,
        description=(
            "Set when the agent stopped at the pre-flight gate to ask instead of building. "
            "`run_id` is null and `pipeline_ran` is false in that case — there is no "
            "package yet. The UI renders these as selectable options."
        ),
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
    """A staged document plus what came out of it.

    The extraction fields are on the upload response on purpose: a scanned PDF that yields
    no text has to be visible to the rep *at upload*, not discovered three minutes later
    when the package comes back without their constraints in it.
    """

    id: str
    session_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int
    stored_path: str
    extraction_status: str = "failed"
    char_count: int = 0
    page_count: int | None = None
    truncated: bool = False
    extraction_detail: str = ""
    #: A first line of the text, for the attachment chip. Never agent context.
    preview: str = ""
