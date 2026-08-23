"""Naming a chat session after the campaign it is about.

A session the UI creates has no title until a brief arrives, so it reads as
"New Campaign" in the sidebar and a history of runs is indistinguishable.

Naming happens in **two steps**, because the two useful names arrive at different times:

1. The first brief provisionally names the session, so the sidebar is not a column of
   placeholders while a 45-90s run streams.
2. When intake resolves a `campaign_objective`, that **replaces** the provisional name.
   The brief is the rep's raw sentence; the objective is what the system decided the
   campaign is, and it is what the centre-panel header shows. A sidebar showing one while
   the header shows the other is the same campaign under two names.

`title_source` is what makes step 2 safe: a title the user typed is never overwritten. It
is recorded on the session record, and only `api/sessions.py`'s PATCH sets it to "user".

Titles are capped at **five words**, not just at a character count. The sidebar rail is
~240px, so a longer label truncates with CSS into something unreadable anyway — better to
cut on a word boundary and say so with an ellipsis.

Lives here rather than in `api/campaign.py` so `api/sessions.py` can reuse it without
pulling in the agent graph.
"""

from __future__ import annotations

import re

from app.services import local_db, transcripts

DEFAULT_TITLE = "New Campaign"
#: Five words is the ask and the rail's width agrees; MAX_CHARS is the secondary guard for
#: five very long words (or one, which has no word boundary to cut on at all).
MAX_WORDS = 5
MAX_CHARS = 60

#: How a session got its current name. Only "user" is protected from being replaced.
SOURCE_AUTO = "auto"
SOURCE_USER = "user"
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

    words = first.split(" ")
    truncated = len(words) > MAX_WORDS
    if truncated:
        first = " ".join(words[:MAX_WORDS])

    if len(first) > MAX_CHARS:
        # Still too long on characters: five words can be five long ones, and a brief with
        # no spaces at all is one word that the word cap never touches.
        first = first[:MAX_CHARS].rsplit(" ", 1)[0] or first[:MAX_CHARS]
        truncated = True

    first = first.rstrip(",;:-")
    return f"{first}{ELLIPSIS}" if truncated else first


def _is_user_named(session: dict) -> bool:
    """Whether the current title was typed by a human.

    Sessions predating `title_source` have no such field. They are treated as auto-named,
    which is the conservative reading: the only way one of them got a title was
    `name_if_unnamed` or `backfill_from_runs`, both automatic.
    """
    return session.get("title_source") == SOURCE_USER


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
    local_db.update(local_db.SESSIONS, session_id, {"title": title, "title_source": SOURCE_AUTO})
    return title


def rename_from_objective(session_id: str, objective: str) -> str:
    """Upgrade an auto-named session to the objective intake resolved.

    The provisional name comes from the rep's raw sentence, which is why the sidebar used
    to read like a chat log while the header read like a campaign. Once a run produces a
    `campaign_objective`, that is the better name and it replaces the provisional one --
    unless the user typed the current title, in which case nothing here touches it.

    Returns the session's title afterwards, renamed or not.
    """
    session = local_db.get_record(local_db.SESSIONS, session_id)
    if session is None:
        return DEFAULT_TITLE
    if _is_user_named(session):
        return session["title"]

    title = title_from_text(objective)
    if title in (DEFAULT_TITLE, session.get("title")):
        return session.get("title") or DEFAULT_TITLE
    local_db.update(local_db.SESSIONS, session_id, {"title": title, "title_source": SOURCE_AUTO})
    return title


def exceeds_cap(title: str) -> bool:
    """Whether a stored title is longer than the rail is willing to show.

    Its own function because it is the test for "this title predates the cap", not just a
    length check — titles written before MAX_WORDS existed are still on disk and the
    sidebar is where they show.
    """
    return len(title.split()) > MAX_WORDS or len(title) > MAX_CHARS + len(ELLIPSIS)


def backfill_from_runs(sessions: list[dict]) -> list[dict]:
    """Bring every auto-named session up to date, in place on disk. Two repairs:

    1. **Still unnamed.** Sessions predating automatic naming, and any whose run started
       before the title was recorded, would otherwise read "New Campaign" forever.
    2. **Named before the five-word cap existed.** Those titles are already on disk, so
       capping `title_from_text` alone would leave the sidebar showing sentence-long rows
       under a rule that supposedly forbids them.

    Both prefer the resolved `campaign_objective` over the raw brief — it is what intake
    actually extracted, and it is what the centre-panel header shows, which is the same
    preference `rename_from_objective` applies to live runs.

    A session with a transcript but no run falls back to its opening message: a
    conversation that failed before intake, or that never asked for a package, is still a
    conversation the user recognises. An over-long title with neither falls back to
    shortening itself. Only a session with nothing at all stays at the default, because
    then genuinely nothing has named it.

    A title the user typed is never touched, by either repair.
    """
    repairable = [
        s
        for s in sessions
        if not _is_user_named(s)
        and (s.get("title", DEFAULT_TITLE) == DEFAULT_TITLE or exceeds_cap(s.get("title") or ""))
    ]
    if not repairable:
        return sessions

    latest: dict[str, dict] = {}
    for run in local_db.list_records(local_db.RUNS):
        session_id = run.get("session_id")
        if session_id is None:
            continue
        current = latest.get(session_id)
        if current is None or (run.get("created_at") or "") >= (current.get("created_at") or ""):
            latest[session_id] = run

    for session in repairable:
        current_title = session.get("title") or DEFAULT_TITLE
        spec = (latest.get(session["id"], {}).get("campaign_spec")) or {}
        source = (
            spec.get("campaign_objective")
            or spec.get("original_query")
            or transcripts.first_user_text(session["id"])
            # Nothing better survives, so shorten what is already there rather than
            # reverting a named session to the placeholder.
            or (current_title if current_title != DEFAULT_TITLE else "")
        )
        title = title_from_text(source)
        if title in (DEFAULT_TITLE, current_title):
            continue
        session["title"] = title
        local_db.update(
            local_db.SESSIONS, session["id"], {"title": title, "title_source": SOURCE_AUTO}
        )

    return sessions
