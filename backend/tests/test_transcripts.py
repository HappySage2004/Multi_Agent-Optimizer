"""Chat transcripts — the conversation, persisted.

The agent's answer used to live only in the SSE `done` event and the browser's React
state, so a reload lost every word of it and a restored session showed a package with no
reasoning attached. These cover the storage, the CRUD surface over it, the cascade when a
session is deleted, and the ordering guarantee the UI replays a transcript from.

Storage is redirected into a temp localDB by the autouse fixture in conftest.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import local_db, session_titles, transcripts


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _new_session(title: str = session_titles.DEFAULT_TITLE) -> str:
    return local_db.insert(local_db.SESSIONS, {"title": title})["id"]


# ------------------------------------------------------------------------- storage


def test_append_and_read_back() -> None:
    session_id = _new_session()
    transcripts.append(session_id, "user", "I have $50,000 for a 30-day flight")
    transcripts.append(session_id, "assistant", "## Recommended package\n\n- 6 screens")

    out = transcripts.list_for_session(session_id)
    assert [m["role"] for m in out] == ["user", "assistant"]
    # Markdown survives the round trip verbatim — the UI renders it.
    assert out[1]["text"] == "## Recommended package\n\n- 6 screens"


def test_order_is_insertion_order_not_timestamp() -> None:
    """Six messages in one second must still come back in the order they were sent.

    `created_at` has second granularity, so a fast turn's messages tie on it. Ordering
    therefore comes from localDB's insertion order, and this pins that.
    """
    session_id = _new_session()
    for i in range(6):
        transcripts.append(session_id, "user" if i % 2 == 0 else "assistant", f"m{i}")

    assert [m["text"] for m in transcripts.list_for_session(session_id)] == [
        f"m{i}" for i in range(6)
    ]
    # All within one second, so a sort on created_at could not have produced that.
    stamps = {m["created_at"] for m in transcripts.list_for_session(session_id)}
    assert len(stamps) <= 2


def test_order_survives_an_update() -> None:
    """`local_db.update` rewrites in place, so amending a message must not reorder."""
    session_id = _new_session()
    ids = [transcripts.append(session_id, "user", f"m{i}")["id"] for i in range(4)]
    transcripts.update(ids[0], {"text": "edited"})

    texts = [m["text"] for m in transcripts.list_for_session(session_id)]
    assert texts == ["edited", "m1", "m2", "m3"]


def test_transcripts_are_scoped_to_their_session() -> None:
    a, b = _new_session(), _new_session()
    transcripts.append(a, "user", "brief A")
    transcripts.append(b, "user", "brief B")

    assert [m["text"] for m in transcripts.list_for_session(a)] == ["brief A"]
    assert [m["text"] for m in transcripts.list_for_session(b)] == ["brief B"]


def test_assistant_message_carries_the_turn_metadata() -> None:
    """A follow-up's answer is a different claim from a rebuild's, so both are recorded."""
    session_id = _new_session()
    record = transcripts.append(
        session_id,
        "assistant",
        "Block 5 was chosen because it carries the peak ridership.",
        run_id="run-abc123",
        pipeline_ran=False,
        tool_trail=["get_active_run", "inspect_package"],
        token_usage={"total_tokens": 4211},
    )

    assert record["run_id"] == "run-abc123"
    assert record["pipeline_ran"] is False
    assert record["tool_trail"] == ["get_active_run", "inspect_package"]
    assert record["token_usage"] == {"total_tokens": 4211}


def test_update_refuses_to_move_a_message_between_sessions() -> None:
    a, b = _new_session(), _new_session()
    message_id = transcripts.append(a, "user", "brief")["id"]

    transcripts.update(message_id, {"session_id": b, "text": "amended"})

    assert transcripts.get(message_id)["session_id"] == a
    assert transcripts.get(message_id)["text"] == "amended"
    assert transcripts.list_for_session(b) == []


def test_clear_session_leaves_the_runs_alone() -> None:
    """ "Clear conversation" is not "delete campaign" — the packages must survive."""
    session_id = _new_session()
    transcripts.append(session_id, "user", "brief")
    transcripts.append(session_id, "assistant", "answer")
    local_db.insert(local_db.RUNS, {"session_id": session_id, "status": "validated"})

    assert transcripts.clear_session(session_id) == 2
    assert transcripts.list_for_session(session_id) == []
    assert [r for r in local_db.list_records(local_db.RUNS) if r["session_id"] == session_id]


# ----------------------------------------------------------------------- endpoints


def test_crud_round_trip(client: TestClient) -> None:
    session_id = _new_session()

    created = client.post(
        f"/sessions/{session_id}/messages",
        json={"role": "user", "text": "brief", "attachments": ["rfp.pdf"]},
    )
    assert created.status_code == 201
    message_id = created.json()["id"]
    assert created.json()["attachments"] == ["rfp.pdf"]

    assert client.get(f"/messages/{message_id}").json()["text"] == "brief"

    listed = client.get(f"/sessions/{session_id}/messages").json()
    assert [m["id"] for m in listed] == [message_id]

    patched = client.patch(f"/messages/{message_id}", json={"text": "revised brief"})
    assert patched.status_code == 200
    assert patched.json()["text"] == "revised brief"
    # An omitted field is left alone, not blanked.
    assert patched.json()["attachments"] == ["rfp.pdf"]

    assert client.delete(f"/messages/{message_id}").status_code == 200
    assert client.get(f"/sessions/{session_id}/messages").json() == []


def test_empty_transcript_is_not_a_404(client: TestClient) -> None:
    """The UI must be able to tell "nothing said yet" from "wrong session id"."""
    session_id = _new_session()
    assert client.get(f"/sessions/{session_id}/messages").status_code == 200
    assert client.get(f"/sessions/{session_id}/messages").json() == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/sessions/ses-nope/messages"),
        ("delete", "/sessions/ses-nope/messages"),
        ("get", "/messages/msg-nope"),
        ("patch", "/messages/msg-nope"),
        ("delete", "/messages/msg-nope"),
    ],
)
def test_unknown_ids_are_404(client: TestClient, method: str, path: str) -> None:
    response = getattr(client, method)(path, **({"json": {}} if method == "patch" else {}))
    assert response.status_code == 404


def test_post_to_unknown_session_is_404(client: TestClient) -> None:
    response = client.post("/sessions/ses-nope/messages", json={"role": "user", "text": "x"})
    assert response.status_code == 404


def test_role_is_validated(client: TestClient) -> None:
    session_id = _new_session()
    response = client.post(f"/sessions/{session_id}/messages", json={"role": "system", "text": "x"})
    assert response.status_code == 422


def test_clear_endpoint(client: TestClient) -> None:
    session_id = _new_session()
    for role in ("user", "assistant"):
        client.post(f"/sessions/{session_id}/messages", json={"role": role, "text": "x"})

    response = client.delete(f"/sessions/{session_id}/messages")
    assert response.status_code == 200
    assert response.json()["deleted"] == 2
    assert client.get(f"/sessions/{session_id}/messages").json() == []


# ------------------------------------------------------------------------- cascade


def test_deleting_a_session_cascades(client: TestClient) -> None:
    """A session's transcript, runs and uploads go with it.

    Left behind, they would be unreachable but still live: `latest_run_for_session` would
    keep resolving runs for a session the user believes is gone.
    """
    session_id = _new_session()
    other = _new_session()

    transcripts.append(session_id, "user", "brief")
    transcripts.append(other, "user", "keep me")
    local_db.insert(local_db.RUNS, {"session_id": session_id, "status": "validated"})
    local_db.insert(local_db.RUNS, {"session_id": other, "status": "validated"})
    local_db.insert(local_db.UPLOADS, {"session_id": session_id, "filename": "rfp.pdf"})

    response = client.delete(f"/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["cascaded"] == {"messages": 1, "runs": 1, "uploads": 1}

    assert local_db.get_record(local_db.SESSIONS, session_id) is None
    assert transcripts.list_for_session(session_id) == []
    assert [r for r in local_db.list_records(local_db.RUNS) if r["session_id"] == session_id] == []

    # The neighbouring session is untouched.
    assert len(transcripts.list_for_session(other)) == 1
    assert [r for r in local_db.list_records(local_db.RUNS) if r["session_id"] == other]


def test_delete_where_matches_on_every_field() -> None:
    session_id = _new_session()
    transcripts.append(session_id, "user", "a")
    transcripts.append(session_id, "assistant", "b")

    assert local_db.delete_where(local_db.MESSAGES, session_id=session_id, role="user") == 1
    assert [m["role"] for m in transcripts.list_for_session(session_id)] == ["assistant"]


# ----------------------------------------------------------------- session naming


def test_a_session_with_only_a_transcript_still_gets_named() -> None:
    """A conversation that never reached intake is still one the user recognises."""
    session_id = _new_session()
    transcripts.append(session_id, "user", "I have $25,000 for a bus shelter test")

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "$25,000 for a bus shelter test"


def test_a_run_objective_still_wins_over_the_transcript() -> None:
    session_id = _new_session()
    transcripts.append(session_id, "user", "some rambling opening message")
    local_db.insert(
        local_db.RUNS,
        {
            "session_id": session_id,
            "campaign_spec": {"campaign_objective": "Airport corridor push"},
        },
    )

    session_titles.backfill_from_runs(local_db.list_records(local_db.SESSIONS))
    assert session_titles.title_of(session_id) == "Airport corridor push"


def test_first_user_text_ignores_assistant_and_blank_messages() -> None:
    session_id = _new_session()
    transcripts.append(session_id, "assistant", "How can I help?")
    transcripts.append(session_id, "user", "   ")
    transcripts.append(session_id, "user", "the real brief")

    assert transcripts.first_user_text(session_id) == "the real brief"


def test_first_user_text_of_an_empty_session_is_none() -> None:
    assert transcripts.first_user_text(_new_session()) is None
