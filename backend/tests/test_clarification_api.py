"""The clarification round over HTTP — the payload the UI actually renders against.

`/campaign/run` and `/campaign/stream` need a real model call, so they are not exercised
here. What is testable without one, and what breaks the UI if it regresses, is the shape:
the hydration route, and that `pending_questions` serializes with every field the card
reads.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import CampaignRunOut
from app.main import app
from app.services import clarifications, local_db
from app.tools import master_tools

SESSION = "ses-api-clarify"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean():
    # The session has to exist: `ask_clarifying_questions` refuses an unknown one, because
    # questions stored against a session nobody reads leave the rep with a request and no
    # options to click.
    if local_db.get_record(local_db.SESSIONS, SESSION) is None:
        local_db.insert(local_db.SESSIONS, {"id": SESSION, "title": "clarification test"})
    clarifications.close(SESSION)
    yield
    clarifications.close(SESSION)


def _ask(session_id: str = SESSION) -> dict:
    return master_tools.ask_clarifying_questions.invoke(
        {
            "session_id": session_id,
            "understood": "$50k, 30 days, Downtown Core.",
            "questions": [
                {
                    "field": "audience_terms",
                    "question": "Who is this aimed at?",
                    "option_a": "students",
                    "option_b": "young_professionals",
                    "recommended": "B",
                    "recommendation_reason": "downtown launch skews professional",
                },
                {
                    "field": "industry_vertical",
                    "question": "Which vertical?",
                    "option_a": "retail",
                    "option_b": "technology",
                    "recommended": "A",
                    "recommendation_reason": "the brief mentions a storefront",
                },
            ],
        }
    )


def test_hydration_route_returns_null_when_nothing_is_open(client):
    body = client.get("/sessions/ses-nothing-open/clarification").json()
    assert body == {"session_id": "ses-nothing-open", "pending_questions": None}


def test_hydration_route_returns_the_open_round(client):
    _ask()
    body = client.get(f"/sessions/{SESSION}/clarification").json()

    pending = body["pending_questions"]
    assert pending is not None
    assert pending["session_id"] == SESSION
    assert pending["understood"] == "$50k, 30 days, Downtown Core."
    assert pending["answered"] is False
    assert [q["field"] for q in pending["questions"]] == [
        "audience_terms",
        "industry_vertical",
    ]


def test_serialized_options_carry_every_field_the_card_reads(client):
    """The card renders `key`, `label`, `detail`, `kind` and `value`. All must survive JSON."""
    _ask()
    pending = client.get(f"/sessions/{SESSION}/clarification").json()["pending_questions"]
    question = pending["questions"][0]

    assert question["recommended_key"] == "B"
    assert [o["key"] for o in question["options"]] == ["A", "B", "C", "D"]
    assert [o["kind"] for o in question["options"]] == [
        "answer",
        "answer",
        "defer",
        "custom",
    ]
    for option in question["options"]:
        assert set(option) == {"key", "label", "detail", "kind", "value"}

    # The defer option must arrive with its recommendation text, or the UI shows a bare
    # "Decide for yourself" and the rep is guessing.
    defer = next(o for o in question["options"] if o["kind"] == "defer")
    assert "young_professionals" in defer["detail"]


def test_a_closed_round_is_not_served_again(client):
    _ask()
    clarifications.close(SESSION)
    body = client.get(f"/sessions/{SESSION}/clarification").json()
    assert body["pending_questions"] is None


def test_campaign_run_out_defaults_to_no_questions():
    """The field is additive: an ordinary turn must not start reporting a phantom round."""
    out = CampaignRunOut(answer="Here is the package.")
    assert out.pending_questions is None
    assert "pending_questions" in out.model_dump()


def test_campaign_run_out_round_trips_a_request():
    _ask()
    pending = clarifications.get_open(SESSION)
    out = CampaignRunOut(answer="I need two things first.", pending_questions=pending)

    dumped = out.model_dump(mode="json")
    assert dumped["pending_questions"]["questions"][0]["options"][3]["kind"] == "custom"
    assert CampaignRunOut.model_validate(dumped).pending_questions == pending


def test_rounds_are_scoped_per_session(client):
    """Two reps in two sessions must not see each other's questions."""
    _ask(SESSION)
    other = "ses-api-clarify-other"
    try:
        assert client.get(f"/sessions/{other}/clarification").json()["pending_questions"] is None
        assert client.get(f"/sessions/{SESSION}/clarification").json()["pending_questions"]
    finally:
        clarifications.close(other)
