"""Document staging.

Files land in stage/{session_id}/ and only their metadata goes into localDB. Contents are
untrusted: type and size are checked, nothing is executed, and no document is ever passed
whole into agent context.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import UploadOut
from app.config import get_settings
from app.services import local_db

router = APIRouter(prefix="/uploads", tags=["uploads"])

MAX_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".csv", ".docx", ".xlsx", ".pptx"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


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

    record = local_db.insert(
        local_db.UPLOADS,
        {
            "session_id": session_id,
            "filename": filename,
            "content_type": file.content_type,
            "size_bytes": len(payload),
            "stored_path": str(stored.relative_to(settings.stage_dir.parent)),
        },
    )
    return UploadOut(**{k: record[k] for k in UploadOut.model_fields})


@router.get("", response_model=list[UploadOut])
def list_uploads(session_id: str | None = None) -> list[UploadOut]:
    records = local_db.list_records(local_db.UPLOADS)
    if session_id:
        records = [r for r in records if r.get("session_id") == session_id]
    return [UploadOut(**{k: r[k] for k in UploadOut.model_fields}) for r in records]
