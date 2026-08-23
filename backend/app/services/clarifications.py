"""Storage for the one round of clarifying questions a session may have open.

Session-scoped, not run-scoped, and that is the whole reason this is its own collection:
the questions are asked BEFORE `create_campaign_spec`, so there is no `run_id` to hang them
on yet. That is also why they cannot live in `run_state`.

At most one open record per session. Asking again overwrites, and creating a spec closes it
— the pipeline starting is the definition of "we are past asking".
"""

from __future__ import annotations

from app.logging_utils import debug, info
from app.models.clarification import ClarificationRequest
from app.services import local_db

CLARIFICATIONS = "clarifications"


def _record_for(session_id: str) -> dict | None:
    # Insertion order is preserved by local_db, so the last match is the newest round.
    matches = [
        r for r in local_db.list_records(CLARIFICATIONS) if r.get("session_id") == session_id
    ]
    return matches[-1] if matches else None


def put(request: ClarificationRequest) -> ClarificationRequest:
    """Replace this session's open round with `request`."""
    close(request.session_id)
    local_db.insert(CLARIFICATIONS, request.model_dump(mode="json"))
    info(
        f"clarification asked session={request.session_id}: "
        f"{len(request.questions)} question(s) on "
        f"{[q.field for q in request.questions]}"
    )
    return request


def get_open(session_id: str) -> ClarificationRequest | None:
    """The session's unanswered round, or None. What the API returns to the UI."""
    record = _record_for(session_id)
    if record is None:
        return None
    request = ClarificationRequest.model_validate(record)
    return request if request.open else None


def close(session_id: str) -> bool:
    """Mark every round on this session answered. Idempotent.

    Called when a spec is created — the rep either answered or the agent proceeded, and
    either way the questions must stop being re-presented. Marked rather than deleted so
    the transcript keeps what was asked.
    """
    closed = False
    for record in local_db.list_records(CLARIFICATIONS):
        if record.get("session_id") == session_id and not record.get("answered"):
            local_db.update(CLARIFICATIONS, record["id"], {"answered": True})
            closed = True
    if closed:
        debug(f"clarification closed for session={session_id}")
    return closed
