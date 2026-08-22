"""Naming a chat session after the brief that started it.

A session the UI creates has no title until a brief arrives, so it reads as
"New Campaign" in the sidebar and a history of runs is indistinguishable. The first brief
submitted against a session is what names it; a session that was never run, or that the
user renamed, keeps the name it has.

Lives here rather than in `api/campaign.py` so `api/sessions.py` can reuse it without
pulling in the agent graph.
"""

from __future__ import annotations

import re

from app.services import local_db

DEFAULT_TITLE = "New Campaign"
MAX_CHARS = 60
# Escaped rather than a literal character so this stays readable in any editor encoding.
ELLIPSIS = "\u2026"

# Conversational lead-ins, which carry no information about the campaign.
_LEAD_IN = re.compile(
    r"^(?:hi|hello|hey)[,\s]+"
    r"|^(?:i|we)\s+(?:have|need|want|am\s+planning|would\s+like)\s+"
    r"|^(?:i|we)'?d\s+like\s+(?:to\s+)?"
    r"|^(?:please|can\s+you|could\s+you|build|create|plan)\s+(?:me\s+)?",
    re.IGNORECASE,
)
# Lead-ins stack ("Hi, I'd like ...", "Please can you ..."), and one `re.sub` pass only
# ever matches the outermost one at the anchor. Bounded so this cannot spin.
_LEAD_IN_PASSES = 3


def title_from_text(text: str) -> str:
    """A short, human session title from a brief or a resolved campaign objective.

    Truncates on a word boundary so the sidebar shows a readable phrase rather than a cut
    mid-word. Returns `DEFAULT_TITLE` when there is nothing usable to name a session after.
    """
    condensed = " ".join(text.split())
    if not condensed:
        return DEFAULT_TITLE

    # The first sentence carries the subject; the rest is usually constraints. The split is
    # after the punctuation, so drop the terminator — a trailing "." reads as noise in a
    # sidebar label.
    first = re.split(r"(?<=[.!?])\s", condensed, maxsplit=1)[0].rstrip(".!?").rstrip()
    for _ in range(_LEAD_IN_PASSES):
        stripped = _LEAD_IN.sub("", first).strip()
        if stripped == first:
            break
        first = stripped
    if not first:
        first = condensed

    if len(first) <= MAX_CHARS:
        return first
    clipped = first[:MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:-")
    return f"{clipped or first[:MAX_CHARS]}{ELLIPSIS}"


def title_of(session_id: str) -> str:
    session = local_db.get_record(local_db.SESSIONS, session_id) or {}
    return session.get("title") or DEFAULT_TITLE


def name_if_unnamed(session_id: str, text: str) -> str:
    """Name a session from `text`, but only while it still carries the placeholder title.

    Returns the session's title afterwards, named or not. A session the user renamed, or
    that an earlier brief already named, is left alone.
    """
    session = local_db.get_record(local_db.SESSIONS, session_id)
    if session is None:
        return DEFAULT_TITLE
    if session.get("title") != DEFAULT_TITLE:
        return session["title"]

    title = title_from_text(text)
    if title == DEFAULT_TITLE:
        return DEFAULT_TITLE
    local_db.update(local_db.SESSIONS, session_id, {"title": title})
    return title


def backfill_from_runs(sessions: list[dict]) -> list[dict]:
    """Name any still-unnamed session after its latest run, in place on disk.

    Sessions predating automatic naming, and any whose run started before the title was
    recorded, would otherwise read "New Campaign" forever. The resolved
    `campaign_objective` is preferred over the raw brief — it is what intake actually
    extracted, and it is what the centre-panel header shows. A session with no run stays
    at the default, because nothing has named it yet.
    """
    unnamed = [s for s in sessions if s.get("title") == DEFAULT_TITLE]
    if not unnamed:
        return sessions

    latest: dict[str, dict] = {}
    for run in local_db.list_records(local_db.RUNS):
        session_id = run.get("session_id")
        if session_id is None:
            continue
        current = latest.get(session_id)
        if current is None or (run.get("created_at") or "") >= (current.get("created_at") or ""):
            latest[session_id] = run

    for session in unnamed:
        spec = (latest.get(session["id"], {}).get("campaign_spec")) or {}
        source = spec.get("campaign_objective") or spec.get("original_query") or ""
        title = title_from_text(source)
        if title == DEFAULT_TITLE:
            continue
        session["title"] = title
        local_db.update(local_db.SESSIONS, session["id"], {"title": title})

    return sessions
