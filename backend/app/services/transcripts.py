"""Chat transcripts — the conversation itself, persisted.

Sessions, runs and uploads were already durable, but the messages were not: the agent's
answer lived only in the SSE `done` event and the browser's React state, so a reload or a
session switch lost every word of it. A restored session could show its package and not
the reasoning that justified it, which is the opposite of the explainability the whole
system is for.

One record per message. The **assistant** message also carries that turn's metadata --
which run it reported on, whether the pipeline actually ran, the tool trail and the token
totals -- because those describe the turn, not the session, and a follow-up turn's answer
is a different claim from a rebuild's.

Ordering is `local_db` insertion order, deliberately. `created_at` has second granularity,
so a fast turn's user and assistant messages tie on it and sorting would be unstable.

Written by the campaign endpoints, so a transcript is durable whether or not the client
survives the request. The CRUD endpoints in `api/messages.py` exist for the client to
read, amend and clear it.
"""

from __future__ import annotations

from typing import Any, Literal

from app.services import local_db

Role = Literal["user", "assistant"]

#: Fields a caller may set. Everything else on a record is assigned by `local_db.insert`.
_WRITABLE = (
    "session_id",
    "role",
    "text",
    "run_id",
    "attachments",
    "pipeline_ran",
    "tool_trail",
    "token_usage",
)


def append(
    session_id: str,
    role: Role,
    text: str,
    *,
    run_id: str | None = None,
    attachments: list[str] | None = None,
    pipeline_ran: bool | None = None,
    tool_trail: list[str] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one message to a session's transcript and return the stored record."""
    return local_db.insert(
        local_db.MESSAGES,
        {
            "session_id": session_id,
            "role": role,
            "text": text,
            "run_id": run_id,
            "attachments": attachments or [],
            "pipeline_ran": pipeline_ran,
            "tool_trail": tool_trail or [],
            "token_usage": token_usage,
        },
    )


def list_for_session(session_id: str) -> list[dict[str, Any]]:
    """The session's messages in the order they were sent."""
    return [
        r for r in local_db.list_records(local_db.MESSAGES) if r.get("session_id") == session_id
    ]


def get(message_id: str) -> dict[str, Any] | None:
    return local_db.get_record(local_db.MESSAGES, message_id)


def update(message_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    """Amend a message. Only the writable fields are honoured.

    `id`, `created_at` and `session_id` are excluded: moving a message between sessions
    would reorder two transcripts at once, and there is no use for it.
    """
    allowed = {k: v for k, v in patch.items() if k in _WRITABLE and k != "session_id"}
    if not allowed:
        return get(message_id)
    return local_db.update(local_db.MESSAGES, message_id, allowed)


def delete(message_id: str) -> bool:
    return local_db.delete(local_db.MESSAGES, message_id)


def clear_session(session_id: str) -> int:
    """Drop a session's whole transcript, leaving the session and its runs intact.

    Backs the UI's "clear conversation" action, which is not a session delete.
    """
    return local_db.delete_where(local_db.MESSAGES, session_id=session_id)


def first_user_text(session_id: str) -> str | None:
    """The opening brief, used to name a session that never got titled."""
    for record in list_for_session(session_id):
        if record.get("role") == "user" and (record.get("text") or "").strip():
            return str(record["text"]).strip()
    return None
