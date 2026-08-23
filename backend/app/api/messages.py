"""Chat transcript CRUD.

The conversation is now a persisted resource in its own right, not a side effect of
running the pipeline. The campaign endpoints write both halves of every turn (see
`services/transcripts`), so these routes are the read/amend/clear surface over what they
recorded.

Nested under `/sessions/{session_id}` for the collection, flat under `/messages/{id}` for
a single record -- a message id is globally unique, so requiring the session in the path
would only invite the two to disagree.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import MessageCreate, MessageOut, MessageUpdate
from app.services import local_db, transcripts

router = APIRouter(tags=["messages"])


def _out(record: dict) -> MessageOut:
    return MessageOut(
        id=record["id"],
        session_id=record.get("session_id") or "",
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
        role=record.get("role") or "assistant",
        text=record.get("text") or "",
        run_id=record.get("run_id"),
        attachments=record.get("attachments") or [],
        pipeline_ran=record.get("pipeline_ran"),
        tool_trail=record.get("tool_trail") or [],
        token_usage=record.get("token_usage"),
    )


def _require_session(session_id: str) -> None:
    if local_db.get_record(local_db.SESSIONS, session_id) is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def list_messages(session_id: str) -> list[MessageOut]:
    """The session's transcript, in the order it was sent.

    A session with no messages returns `[]`; only an unknown session is a 404, so the UI
    can tell "nothing said yet" from "wrong id".
    """
    _require_session(session_id)
    return [_out(r) for r in transcripts.list_for_session(session_id)]


@router.post("/sessions/{session_id}/messages", response_model=MessageOut, status_code=201)
def create_message(session_id: str, payload: MessageCreate) -> MessageOut:
    _require_session(session_id)
    return _out(
        transcripts.append(
            session_id,
            payload.role,
            payload.text,
            run_id=payload.run_id,
            attachments=payload.attachments,
            pipeline_ran=payload.pipeline_ran,
            tool_trail=payload.tool_trail,
            token_usage=payload.token_usage,
        )
    )


@router.delete("/sessions/{session_id}/messages")
def clear_messages(session_id: str) -> dict:
    """Clear the transcript without touching the session or its runs.

    This is the "new conversation, same session" action. Deleting the runs too would throw
    away the packages the user built, which is a different and much more destructive
    intent -- that is `DELETE /sessions/{id}`.
    """
    _require_session(session_id)
    return {"session_id": session_id, "deleted": transcripts.clear_session(session_id)}


@router.get("/messages/{message_id}", response_model=MessageOut)
def get_message(message_id: str) -> MessageOut:
    record = transcripts.get(message_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No message '{message_id}'")
    return _out(record)


@router.patch("/messages/{message_id}", response_model=MessageOut)
def update_message(message_id: str, payload: MessageUpdate) -> MessageOut:
    if transcripts.get(message_id) is None:
        raise HTTPException(status_code=404, detail=f"No message '{message_id}'")
    # `exclude_unset` so PATCHing one field cannot blank the others -- an omitted field
    # and an explicit null would otherwise be indistinguishable.
    record = transcripts.update(message_id, payload.model_dump(exclude_unset=True))
    if record is None:
        raise HTTPException(status_code=404, detail=f"No message '{message_id}'")
    return _out(record)


@router.delete("/messages/{message_id}")
def delete_message(message_id: str) -> dict:
    if not transcripts.delete(message_id):
        raise HTTPException(status_code=404, detail=f"No message '{message_id}'")
    return {"deleted": message_id}
