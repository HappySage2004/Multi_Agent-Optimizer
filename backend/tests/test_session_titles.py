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
        # Lead-ins stack, so stripping has to repeat.
        ("Hi, I'd like a campaign for the North Line", "a campaign for the North Line"),
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
    """A brief with no spaces has no word boundary to cut on; it must not come back whole."""
    title = session_titles.title_from_text("x" * 400)
    assert len(title) == session_titles.MAX_CHARS + len(session_titles.ELLIPSIS)


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

    # The resolved objective is preferred over the raw brief.
    assert backfilled["title"] == "Consumer tech product launch in Downtown Core"
    # And it was persisted, not just returned.
    assert session_titles.title_of(session_id) == "Consumer tech product launch in Downtown Core"


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
    assert session_titles.title_of(session_id) == "$25,000 for a bus shelter test"


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
