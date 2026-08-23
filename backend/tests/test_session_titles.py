"""Session naming — the sidebar's only distinguishing label.

Storage is redirected into a temp localDB by the autouse fixture in conftest.py, so these
touch a real (throwaway) JSON database rather than mocks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import local_db, session_titles


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _new_session() -> str:
    return local_db.insert(local_db.SESSIONS, {"title": session_titles.DEFAULT_TITLE})["id"]


# ------------------------------------------------------------------- title_from_text


@pytest.mark.parametrize(
    ("brief", "expected"),
    [
        # The conversational lead-in carries nothing about the campaign.
        ("I have $50,000 for a 30-day flight", "$50,000 for a 30-day flight"),
        ("We need a $10k awareness push", "a $10k awareness push"),
        # Lead-ins stack, so stripping has to repeat. Six words survive the strip, so the
        # five-word cap then bites.
        ("Hi, I'd like a campaign for the North Line", "a campaign for the North\u2026"),
        ("Please can you build me a frequency campaign", "a frequency campaign"),
        # Only the first sentence; the rest is constraints.
        ("Retail launch in Downtown Core. Optimize for reach.", "Retail launch in Downtown Core"),
        # Nothing to name it after.
        ("", session_titles.DEFAULT_TITLE),
        ("   \n\t ", session_titles.DEFAULT_TITLE),
    ],
)
def test_title_from_text(brief: str, expected: str) -> None:
    assert session_titles.title_from_text(brief) == expected


@pytest.mark.parametrize(
    ("brief", "expected"),
    [
        # Exactly five words is not truncated -- the cap is inclusive.
        ("Retail launch in Downtown Core", "Retail launch in Downtown Core"),
        ("Metro platform awareness push for Q4", "Metro platform awareness push for\u2026"),
        # Trailing punctuation left at the cut reads as noise on a rail label.
        (
            "Reach commuters, students, workers, tourists everywhere",
            "Reach commuters, students, workers, tourists\u2026",
        ),
    ],
)
def test_titles_are_capped_at_five_words(brief: str, expected: str) -> None:
    """The sidebar rail is ~240px; a longer label truncates into CSS mush anyway."""
    assert session_titles.title_from_text(brief) == expected


def test_long_brief_truncates_on_a_word_boundary() -> None:
    brief = (
        "Consumer tech product launch targeting commuters aged 18-34 in the Downtown Core "
        "zone of Las Hackland"
    )
    title = session_titles.title_from_text(brief)
    stem = title.removesuffix(session_titles.ELLIPSIS)

    assert title.endswith(session_titles.ELLIPSIS)
    assert len(title) <= session_titles.MAX_CHARS + len(session_titles.ELLIPSIS)
    # A word boundary, not a cut mid-word.
    assert brief.startswith(stem)
    assert not stem.endswith(" ")


def test_unbroken_string_still_truncates() -> None:
    """A brief with no spaces has no word boundary to cut on; it must not come back whole.

    One word never trips the word cap, which is why MAX_CHARS has to stay as a second
    guard rather than being replaced by it.
    """
    title = session_titles.title_from_text("x" * 400)
    assert len(title) == session_titles.MAX_CHARS + len(session_titles.ELLIPSIS)


def test_five_very_long_words_still_hit_the_character_guard() -> None:
    title = session_titles.title_from_text(" ".join(["averyverylongword"] * 5))
    assert title.endswith(session_titles.ELLIPSIS)
    assert len(title) <= session_titles.MAX_CHARS + len(session_titles.ELLIPSIS)


# ------------------------------------------------------------------ name_if_unnamed


def test_first_brief_names_an_unnamed_session() -> None:
    session_id = _new_session()
    returned = session_titles.name_if_unnamed(session_id, "Retail launch in Downtown Core")

    assert returned == "Retail launch in Downtown Core"
    assert session_titles.title_of(session_id) == "Retail launch in Downtown Core"


def test_a_named_session_is_never_renamed() -> None:
    """A second brief in the same session must not relabel the user's history."""
    session_id = _new_session()
    session_titles.name_if_unnamed(session_id, "Retail launch in Downtown Core")
    returned = session_titles.name_if_unnamed(session_id, "Something else entirely")

    assert returned == "Retail launch in Downtown Core"
    assert session_titles.title_of(session_id) == "Retail launch in Downtown Core"


def test_a_user_rename_survives_a_later_brief(client: TestClient) -> None:
    session_id = _new_session()
    assert client.patch(f"/sessions/{session_id}", json={"title": "Q4 retail"}).status_code == 200

    session_titles.name_if_unnamed(session_id, "A brief that would otherwise rename this")
    assert session_titles.title_of(session_id) == "Q4 retail"


def test_an_empty_brief_leaves_the_placeholder() -> None:
    session_id = _new_session()
    assert session_titles.name_if_unnamed(session_id, "   ") == session_titles.DEFAULT_TITLE
    assert session_titles.title_of(session_id) == session_titles.DEFAULT_TITLE


def test_missing_session_does_not_raise() -> None:
    assert session_titles.name_if_unnamed("ses-does-not-exist", "brief") == (
        session_titles.DEFAULT_TITLE
    )


# ------------------------------------------------------------- rename_from_objective


def test_the_objective_replaces_the_provisional_brief_title() -> None:
    """The whole point: the sidebar and the centre header must show the same name.

    `_ensure_session` names a session from the rep's raw sentence so the rail is not a
    placeholder during a 90s run. The header shows `campaign_objective`. Without this
    upgrade the two disagree, and the rail reads like a chat log.
    """
    session_id = _new_session()
    session_titles.name_if_unnamed(session_id, "I have $50,000 for a 30-day flight")
    assert session_titles.title_of(session_id) == "$50,000 for a 30-day flight"

    returned = session_titles.rename_from_objective(session_id, "Retail launch in Downtown Core")

    assert returned == "Retail launch in Downtown Core"
    assert session_titles.title_of(session_id) == "Retail launch in Downtown Core"


def test_the_objective_is_capped_at_five_words_too() -> None:
    session_id = _new_session()
    session_titles.name_if_unnamed(session_id, "A brief")
    session_titles.rename_from_objective(
        session_id, "Consumer tech product launch targeting young commuters"
    )
    assert session_titles.title_of(session_id) == "Consumer tech product launch targeting\u2026"


def test_a_user_rename_survives_the_objective_upgrade(client: TestClient) -> None:
    """A title someone typed is the one thing here that is never overwritten."""
    session_id = _new_session()
    session_titles.name_if_unnamed(session_id, "I have $50,000 for a flight")
    assert client.patch(f"/sessions/{session_id}", json={"title": "Q4 retail"}).status_code == 200

    returned = session_titles.rename_from_objective(session_id, "Retail launch in Downtown Core")

    assert returned == "Q4 retail"
    assert session_titles.title_of(session_id) == "Q4 retail"


def test_a_legacy_session_with_no_title_source_is_treated_as_auto_named() -> None:
    """Records predating `title_source` only ever got a title automatically, so upgrading
    them is safe -- and the conservative reading would leave them stale forever."""
    session_id = local_db.insert(local_db.SESSIONS, {"title": "an old brief-derived name"})["id"]
    assert "title_source" not in local_db.get_record(local_db.SESSIONS, session_id)

    session_titles.rename_from_objective(session_id, "Airport corridor push")
    assert session_titles.title_of(session_id) == "Airport corridor push"


def test_an_empty_objective_leaves_the_provisional_name() -> None:
    """A run that died before intake recorded anything must not blank the rail."""
    session_id = _new_session()
    session_titles.name_if_unnamed(session_id, "Retail launch in Downtown Core")

    assert session_titles.rename_from_objective(session_id, "") == "Retail launch in Downtown Core"
    assert session_titles.title_of(session_id) == "Retail launch in Downtown Core"


def test_rename_of_a_missing_session_does_not_raise() -> None:
    assert session_titles.rename_from_objective("ses-does-not-exist", "x") == (
        session_titles.DEFAULT_TITLE
    )


# ----------------------------------------------------------------- backfill_from_runs


def test_backfill_names_a_legacy_session_from_its_run() -> None:
    """Sessions predating automatic naming get named from their own run history."""
    session_id = _new_session()
    local_db.insert(
        local_db.RUNS,
        {
            "session_id": session_id,
            "campaign_spec": {
                "campaign_objective": "Consumer tech product launch in Downtown Core",
                "original_query": "I have $50,000 for a 30-day campaign",
            },
        },
    )

    records = session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    backfilled = next(r for r in records if r["id"] == session_id)

    # The resolved objective is preferred over the raw brief, then capped at five words.
    assert backfilled["title"] == "Consumer tech product launch in\u2026"
    # And it was persisted, not just returned.
    assert session_titles.title_of(session_id) == "Consumer tech product launch in\u2026"


def test_backfill_falls_back_to_the_original_query() -> None:
    session_id = _new_session()
    local_db.insert(
        local_db.RUNS,
        {
            "session_id": session_id,
            "campaign_spec": {"original_query": "I have $25,000 for a bus shelter test"},
        },
    )

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "$25,000 for a bus shelter\u2026"


def test_backfill_prefers_the_latest_run() -> None:
    session_id = _new_session()
    for created_at, objective in (
        ("2026-01-01T00:00:00", "The earlier objective"),
        ("2026-06-01T00:00:00", "The latest objective"),
    ):
        local_db.insert(
            local_db.RUNS,
            {
                "session_id": session_id,
                "created_at": created_at,
                "campaign_spec": {"campaign_objective": objective},
            },
        )

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "The latest objective"


def test_backfill_leaves_a_session_with_no_runs_alone() -> None:
    """Nothing has named it yet, so the placeholder is the honest label."""
    session_id = _new_session()
    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == session_titles.DEFAULT_TITLE


def test_backfill_shortens_a_title_written_before_the_five_word_cap() -> None:
    """Capping `title_from_text` alone leaves every existing row breaking the rule."""
    session_id = local_db.insert(
        local_db.SESSIONS,
        {"title": "$50,000 for a 30-day campaign starting 2026-10-01 targeting commuters"},
    )["id"]
    local_db.insert(
        local_db.RUNS,
        {
            "session_id": session_id,
            "campaign_spec": {"campaign_objective": "Commuter reach in Downtown Core"},
        },
    )

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "Commuter reach in Downtown Core"


def test_backfill_shortens_in_place_when_there_is_no_run_to_name_it_from() -> None:
    """No better source survives, so shorten what is there rather than reverting to the
    placeholder -- a named session must not become "New Campaign" again."""
    session_id = local_db.insert(
        local_db.SESSIONS, {"title": "Retail launch across the whole eastern metro corridor"}
    )["id"]

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "Retail launch across the whole…"


def test_backfill_leaves_a_user_typed_long_title_alone(client: TestClient) -> None:
    """The cap is a default, not a policy the rep's own words are subject to."""
    session_id = _new_session()
    long_title = "Q4 retail launch across the eastern metro corridor"
    client.patch(f"/sessions/{session_id}", json={"title": long_title})

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == long_title


def test_backfill_is_idempotent_on_a_capped_title() -> None:
    """It runs on every GET /sessions, so a second pass must not keep re-truncating."""
    session_id = local_db.insert(local_db.SESSIONS, {"title": "Retail launch across the whole…"})[
        "id"
    ]

    for _ in range(3):
        session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "Retail launch across the whole…"


def test_backfill_does_not_touch_an_already_named_session() -> None:
    session_id = _new_session()
    session_titles.name_if_unnamed(session_id, "Retail launch")
    local_db.insert(
        local_db.RUNS,
        {"session_id": session_id, "campaign_spec": {"campaign_objective": "Something else"}},
    )

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "Retail launch"


# ----------------------------------------------------------------------- endpoints


def test_list_sessions_backfills(client: TestClient) -> None:
    session_id = _new_session()
    local_db.insert(
        local_db.RUNS,
        {
            "session_id": session_id,
            "campaign_spec": {"campaign_objective": "Airport corridor push"},
        },
    )

    listed = client.get("/sessions").json()
    assert next(s for s in listed if s["id"] == session_id)["title"] == "Airport corridor push"


def test_patch_renames(client: TestClient) -> None:
    session_id = _new_session()
    response = client.patch(f"/sessions/{session_id}", json={"title": "  Q4 retail  "})

    assert response.status_code == 200
    # Whitespace is trimmed on the way in.
    assert response.json()["title"] == "Q4 retail"
    assert response.json()["updated_at"] is not None
    assert client.get(f"/sessions/{session_id}").json()["title"] == "Q4 retail"


def test_patch_unknown_session_is_404(client: TestClient) -> None:
    assert client.patch("/sessions/ses-nope", json={"title": "x"}).status_code == 404


def test_patch_rejects_an_empty_title(client: TestClient) -> None:
    session_id = _new_session()
    assert client.patch(f"/sessions/{session_id}", json={"title": ""}).status_code == 422


def test_patch_marks_the_title_as_user_typed(client: TestClient) -> None:
    """The flag is not cosmetic -- it is what stops the next run overwriting the rename."""
    session_id = _new_session()
    client.patch(f"/sessions/{session_id}", json={"title": "Q4 retail"})

    record = local_db.get_record(local_db.SESSIONS, session_id)
    assert record["title_source"] == session_titles.SOURCE_USER


def test_automatic_naming_marks_the_title_as_auto(client: TestClient) -> None:
    session_id = _new_session()
    session_titles.name_if_unnamed(session_id, "Retail launch in Downtown Core")

    record = local_db.get_record(local_db.SESSIONS, session_id)
    assert record["title_source"] == session_titles.SOURCE_AUTO
