"""localDB/*.json — application persistence only.

Sessions, chat transcripts, campaign runs, and upload metadata. No analytical data, no
file bytes, no row-level artifact content. One JSON file per collection, each holding a
list of records keyed by `id`. Writes are atomic (temp file + replace) and serialized by
a per-file lock.

**Record order is insertion order and callers may rely on it.** `_load` preserves the
stored sequence, and `update`/`delete` rewrite in place, so a collection never reorders.
This is what makes a chat transcript replayable without a sort key: `created_at` has only
second granularity, so two messages in one turn tie and sorting on it would be unstable.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

SESSIONS = "sessions"
MESSAGES = "messages"
RUNS = "runs"
UPLOADS = "uploads"

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(collection: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(collection, threading.Lock())


def _path(collection: str) -> Path:
    return get_settings().local_db_dir / f"{collection}.json"


def _load(collection: str) -> list[dict[str, Any]]:
    path = _path(collection)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text() or "[]")
    except json.JSONDecodeError:
        # A truncated file must not take the API down; surface it as empty and let the
        # caller re-seed. Corrupt content is preserved for inspection.
        path.replace(path.with_suffix(".json.corrupt"))
        return []
    return data if isinstance(data, list) else []


def _atomic_write(collection: str, records: list[dict[str, Any]]) -> None:
    path = _path(collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(records, fh, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def list_records(collection: str) -> list[dict[str, Any]]:
    with _lock_for(collection):
        return _load(collection)


def get_record(collection: str, record_id: str) -> dict[str, Any] | None:
    return next((r for r in list_records(collection) if r.get("id") == record_id), None)


def insert(collection: str, record: dict[str, Any]) -> dict[str, Any]:
    record = {
        "id": record.get("id") or f"{collection[:3]}-{uuid.uuid4().hex[:12]}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    with _lock_for(collection):
        records = _load(collection)
        records.append(record)
        _atomic_write(collection, records)
    return record


def update(collection: str, record_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    with _lock_for(collection):
        records = _load(collection)
        for i, r in enumerate(records):
            if r.get("id") == record_id:
                records[i] = {
                    **r,
                    **patch,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                _atomic_write(collection, records)
                return records[i]
    return None


def delete_where(collection: str, **match: Any) -> int:
    """Drop every record whose fields equal `match`, in one atomic write.

    Cascade cleanup (a deleted session's messages, runs and uploads) would otherwise be
    one read+write per record, and a partial failure would leave the collection half
    orphaned.
    """
    with _lock_for(collection):
        records = _load(collection)
        remaining = [r for r in records if any(r.get(k) != v for k, v in match.items())]
        removed = len(records) - len(remaining)
        if removed:
            _atomic_write(collection, remaining)
        return removed


def delete(collection: str, record_id: str) -> bool:
    with _lock_for(collection):
        records = _load(collection)
        remaining = [r for r in records if r.get("id") != record_id]
        if len(remaining) == len(records):
            return False
        _atomic_write(collection, remaining)
        return True
