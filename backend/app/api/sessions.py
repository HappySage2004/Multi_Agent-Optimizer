"""Chat sessions. Persisted in localDB/sessions.json.

A session owns three other collections -- its transcript, its runs and its uploads -- so
deleting one cascades. Leaving them behind would keep a session's whole history on disk
with nothing able to reach it, and `latest_run_for_session` would still resolve runs for a
session the user believes is gone.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import SessionCreate, SessionOut, SessionUpdate
from app.services import local_db, session_titles

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _out(record: dict) -> SessionOut:
    return SessionOut(
        id=record["id"],
        title=record.get("title") or session_titles.DEFAULT_TITLE,
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    )


@router.post("", response_model=SessionOut)
def create_session(payload: SessionCreate) -> SessionOut:
    return _out(local_db.insert(local_db.SESSIONS, {"title": payload.title}))


@router.get("", response_model=list[SessionOut])
def list_sessions() -> list[SessionOut]:
    # Sessions that ran before automatic naming existed are named here, from their own run
    # history, so the sidebar is not a column of identical "New Campaign" rows.
    records = session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    return [_out(r) for r in sorted(records, key=lambda r: r.get("created_at") or "", reverse=True)]


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str) -> SessionOut:
    record = local_db.get_record(local_db.SESSIONS, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")
    return _out(record)


@router.patch("/{session_id}", response_model=SessionOut)
def update_session(session_id: str, payload: SessionUpdate) -> SessionOut:
    """Rename a session — the only mutable field.

    Marks the title as user-typed, which is what stops a later run replacing it with the
    campaign objective it resolved.
    """
    record = local_db.update(
        local_db.SESSIONS,
        session_id,
        {"title": payload.title.strip(), "title_source": session_titles.SOURCE_USER},
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")
    return _out(record)


@router.delete("/{session_id}")
def delete_session(session_id: str) -> dict:
    """Delete a session and everything that belongs to it.

    The counts come back so a caller can see what went. Files on disk -- staged uploads
    under `stage/{session_id}/` and artifact parquet under `backend/artifacts/` -- are
    left alone: they are unreachable once these records are gone, and deleting bytes is a
    heavier, irreversible step than removing an index entry.
    """
    if local_db.get_record(local_db.SESSIONS, session_id) is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")

    cascaded = {
        collection: local_db.delete_where(collection, session_id=session_id)
        for collection in (local_db.MESSAGES, local_db.RUNS, local_db.UPLOADS)
    }
    local_db.delete(local_db.SESSIONS, session_id)
    return {"deleted": session_id, "cascaded": cascaded}
