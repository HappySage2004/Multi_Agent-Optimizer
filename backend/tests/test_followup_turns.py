"""A follow-up question must not re-run the pipeline.

The full orchestration costs ~90s and ~17 model calls, so answering "why that time block?"
by rebuilding the package is both slow and wrong — it can return a *different* package than
the one the user is asking about. These cover the machinery the Master Agent's triage rule
depends on: finding the session's existing run, reporting the inputs that would justify a
rebuild, and telling the UI whether one happened.

No LLM in the loop; storage is redirected into a temp localDB by conftest.py.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.api.campaign import _pipeline_ran
from app.services import local_db, run_state
from app.tools import master_tools

START = (date.today() + timedelta(days=14)).isoformat()


def _new_session() -> str:
    return local_db.insert(local_db.SESSIONS, {"title": "New Campaign"})["id"]


def _spec(session_id: str, **overrides) -> dict:
    payload = {
        "campaign_objective": "reach young commuters for a product launch",
        "optimization_goal": "reach",
        "start_date": START,
        "duration_days": 30,
        "budget": 50_000.0,
        "city_ids": ["LH"],
        "zone_ids": ["LH-ZONE-001"],
        "audience_age_min": 18,
        "audience_age_max": 34,
        "audience_commuter": True,
        "session_id": session_id,
    }
    payload.update(overrides)
    return master_tools.create_campaign_spec.invoke(payload)


def _get_active_run(session_id: str) -> dict:
    return master_tools.get_active_run.invoke({"session_id": session_id})


# ------------------------------------------------------- latest_run_for_session


def test_no_runs_yet() -> None:
    assert run_state.latest_run_for_session(_new_session()) is None


def test_returns_the_most_recent_run() -> None:
    session_id = _new_session()
    first = _spec(session_id)["run_id"]
    second = _spec(session_id, budget=30_000.0)["run_id"]

    assert first != second
    assert run_state.latest_run_for_session(session_id) == second


def test_runs_are_scoped_to_their_own_session() -> None:
    """One session's package must never answer another session's follow-up."""
    a, b = _new_session(), _new_session()
    run_a = _spec(a)["run_id"]
    run_b = _spec(b, budget=12_345.0)["run_id"]

    assert run_state.latest_run_for_session(a) == run_a
    assert run_state.latest_run_for_session(b) == run_b


# -------------------------------------------------------------- get_active_run


def test_opening_brief_reports_none() -> None:
    """With no run, the agent is told to start the pipeline rather than guess."""
    out = _get_active_run(_new_session())
    assert out["status"] == "none"
    assert "create_campaign_spec" in out["detail"]


def test_reports_the_existing_run_and_its_inputs() -> None:
    session_id = _new_session()
    run_id = _spec(session_id)["run_id"]

    out = _get_active_run(session_id)
    assert out["status"] == "ok"
    assert out["run_id"] == run_id
    # No package has been optimized yet, so this must not claim one.
    assert out["has_package"] is False
    assert out["state"]["run_id"] == run_id


def test_campaign_inputs_cover_every_optimizer_input() -> None:
    """The agent decides "rebuild or answer" off this dict, so it must be complete.

    A field the optimizer consumes but which is missing here would be a change the agent
    cannot see — it would answer from a stale package instead of rebuilding.
    """
    session_id = _new_session()
    _spec(session_id)
    inputs = _get_active_run(session_id)["campaign_inputs"]

    expected = {
        "campaign_objective",
        "industry_vertical",
        "ad_type",
        "budget",
        "start_date",
        "duration_days",
        "city_ids",
        "zone_ids",
        "corridor_ids",
        "target_audience",
        "audience_terms",
        "optimization_goal",
        "requested_num_screens",
        "preferred_dayparts",
        "preferred_time_blocks",
        "day_type_focus",
        "hard_constraints",
        "soft_preferences",
    }
    assert set(inputs) == expected


def test_campaign_inputs_are_json_safe() -> None:
    """These go into a prompt, so nothing may be a date, tuple or Pydantic model."""
    import json

    session_id = _new_session()
    _spec(session_id)
    out = _get_active_run(session_id)

    # Raises TypeError on a non-serializable value.
    json.dumps(out)
    assert out["campaign_inputs"]["start_date"] == START


def test_campaign_inputs_reflect_a_changed_brief() -> None:
    """After a rebuild, a follow-up must be compared against the *new* inputs."""
    session_id = _new_session()
    _spec(session_id, budget=50_000.0)
    _spec(session_id, budget=30_000.0, optimization_goal="frequency")

    inputs = _get_active_run(session_id)["campaign_inputs"]
    assert inputs["budget"] == 30_000.0
    assert inputs["optimization_goal"] == "frequency"


def test_unknown_session_reports_none_rather_than_raising() -> None:
    out = _get_active_run("ses-does-not-exist")
    assert out["status"] == "none"


# ----------------------------------------------------------------- _pipeline_ran


@pytest.mark.parametrize(
    ("before", "after", "expected", "why"),
    [
        ("run-1", "run-2", True, "a rebuild creates a new run"),
        ("run-1", "run-1", False, "a follow-up leaves the run id unchanged"),
        (None, "run-1", True, "the opening brief creates the session's first run"),
        (None, None, False, "the agent never got as far as creating a run"),
        ("run-1", None, False, "no current run means nothing was built"),
    ],
)
def test_pipeline_ran(before: str | None, after: str | None, expected: bool, why: str) -> None:
    assert _pipeline_ran(after, before) is expected, why


def test_pipeline_ran_end_to_end() -> None:
    """The signal the UI keys off, measured the way the endpoint measures it."""
    session_id = _new_session()

    before = run_state.latest_run_for_session(session_id)
    first = _spec(session_id)["run_id"]
    assert _pipeline_ran(run_state.latest_run_for_session(session_id), before) is True

    # A follow-up turn calls no run-creating tool, so the id is unchanged.
    before = run_state.latest_run_for_session(session_id)
    assert before == first
    assert _pipeline_ran(run_state.latest_run_for_session(session_id), before) is False

    # A revised brief does create one.
    _spec(session_id, budget=25_000.0)
    assert _pipeline_ran(run_state.latest_run_for_session(session_id), before) is True
