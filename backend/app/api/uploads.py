"""Document staging and parsing.

Files land in stage/{session_id}/, their text is extracted once here, and only metadata
plus a summary goes into localDB. Contents are untrusted: type and size are checked,
nothing is executed, and no document is ever passed whole into agent context — the agent
reads a bounded excerpt through `master_tools.read_campaign_document`.

Only the three formats `app/services/documents.py` can actually read are accepted. The
endpoint used to allow .csv/.md/.xlsx/.pptx as well, which was worse than rejecting them:
the file uploaded cleanly, the agent never saw a word of it, and nothing said so.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import UploadOut
from app.config import get_settings
from app.logging_utils import info
from app.services import documents, local_db

router = APIRouter(prefix="/uploads", tags=["uploads"])

MAX_BYTES = 20 * 1024 * 1024
#: Exactly what the parser supports — see the module docstring for why this is not wider.
ALLOWED_SUFFIXES = set(documents.SUPPORTED_SUFFIXES)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _out(record: dict) -> UploadOut:
    """Project a localDB record onto the response model, tolerating older records.

    Uploads staged before extraction existed have none of the extraction keys, so every
    one of them falls back to its model default rather than raising a KeyError.
    """
    return UploadOut(
        **{k: record[k] for k in UploadOut.model_fields if k in record},
    )


def _safe_filename(name: str) -> str:
    # Strip any directory component before sanitizing, so "../../etc/passwd" cannot escape.
    stem = Path(name).name
    cleaned = _SAFE_NAME.sub("_", stem).strip("._") or "upload"
    return cleaned[:120]


@router.post("", response_model=UploadOut)
async def stage_document(session_id: str = Form(...), file: UploadFile = File(...)) -> UploadOut:
    settings = get_settings()
    filename = _safe_filename(file.filename or "upload")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    payload = await file.read()
    if len(payload) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_BYTES // 1024 // 1024} MB")
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")

    session_dir = settings.stage_dir / _safe_filename(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    stored = session_dir / f"{uuid.uuid4().hex[:8]}-{filename}"
    stored.write_bytes(payload)

    # Parsed here, once, rather than per agent turn: re-reading a 50-page PDF on every
    # tool call is wasted work against a per-minute rate limit.
    summary = documents.extract_and_store(stored)
    info(
        f"staged {filename} ({len(payload)} bytes) -> extraction={summary.extraction_status} "
        f"chars={summary.char_count} pages={summary.page_count}"
    )

    record = local_db.insert(
        local_db.UPLOADS,
        {
            "session_id": session_id,
            "filename": filename,
            "content_type": file.content_type,
            "size_bytes": len(payload),
            "stored_path": str(stored.relative_to(settings.stage_dir.parent)),
            **summary.model_dump(mode="json"),
        },
    )
    return _out(record)


@router.get("", response_model=list[UploadOut])
def list_uploads(session_id: str | None = None) -> list[UploadOut]:
    records = local_db.list_records(local_db.UPLOADS)
    if session_id:
        records = [r for r in records if r.get("session_id") == session_id]
    return [_out(r) for r in records]


@router.get("/{upload_id}", response_model=UploadOut)
def get_upload(upload_id: str) -> UploadOut:
    record = local_db.get_record(local_db.UPLOADS, upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No upload '{upload_id}'")
    return _out(record)


@router.delete("/{upload_id}")
def delete_upload(upload_id: str) -> dict:
    """Forget a staged document. Its bytes and extracted text go too.

    Unlike a run artifact, an upload is the user's own file and they asked for it gone —
    so here the files really are deleted, not just dereferenced.
    """
    record = local_db.get_record(local_db.UPLOADS, upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No upload '{upload_id}'")

    root = get_settings().stage_dir.parent
    for key in ("stored_path", "text_path"):
        if relative := record.get(key):
            (root / str(relative)).unlink(missing_ok=True)

    local_db.delete(local_db.UPLOADS, upload_id)
    return {"deleted": upload_id}
