"""The pre-flight clarification gate.

The gate exists because a brief with no audience and no industry leaves 0.80 of the
relevance weight defaulted to a flat 0.5, and the pipeline reports that only in a [DEBUG]
line the rep never sees. These tests pin the contract the UI renders against, and the
invariants that stop the gate becoming friction: a closed field list, a hard question cap,
and a defer option that always says what it defers to.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.models.campaign import AUDIENCE_TERMS
from app.models.clarification import ASKABLE_FIELDS, build_question
from app.services import clarifications
from app.tools import master_tools

SESSION = "ses-clarify-test"

AUDIENCE_Q = {
    "field": "audience_terms",
    "question": "Who is this aimed at?",
    "option_a": "students",
    "option_a_detail": "Scores university POIs and midday traffic.",
    "option_b": "young_professionals",
    "option_b_detail": "Scores income index and the commuter peaks.",
    "recommended": "B",
    "recommendation_reason": "a product launch in a downtown zone skews professional",
}


def _ask(**overrides):
    payload = {
        "session_id": SESSION,
        "understood": "$50k, 30 days, Downtown Core, optimizing for reach.",
        "questions": [dict(AUDIENCE_Q)],
    }
    payload.update(overrides)
    return master_tools.ask_clarifying_questions.invoke(payload)


@pytest.fixture(autouse=True)
def _clean():
    clarifications.close(SESSION)
    yield
    clarifications.close(SESSION)


# --------------------------------------------------------------- option shape


def test_the_tool_builds_exactly_four_options_the_agent_did_not_write():
    out = _ask()
    assert out["status"] == "asked"
    assert out["rendered_options"] == {"q1": ["A", "B", "C", "D"]}

    question = clarifications.get_open(SESSION).questions[0]
    kinds = [o.kind for o in question.options]
    assert kinds == ["answer", "answer", "defer", "custom"]


def test_defer_option_states_what_it_defers_to():
    """A `Decide for yourself` that does not say what it decides is a silent default."""
    _ask()
    defer = next(
        o for o in clarifications.get_open(SESSION).questions[0].options if o.kind == "defer"
    )
    assert defer.label == "Decide for yourself"
    # It committed to B and quoted the reason.
    assert "young_professionals" in defer.detail
    assert "skews professional" in defer.detail


def test_custom_option_is_the_only_one_marked_for_a_text_field():
    _ask()
    options = clarifications.get_open(SESSION).questions[0].options
    assert [o.key for o in options if o.kind == "custom"] == ["D"]


def test_answer_options_carry_a_machine_readable_value():
    _ask()
    options = clarifications.get_open(SESSION).questions[0].options
    values = {o.key: o.value for o in options}
    assert values["A"] == "students"
    assert values["B"] == "young_professionals"
    # Defer and custom have no value — there is nothing to resolve them to yet.
    assert values["C"] is None and values["D"] is None


def test_recommended_must_be_a_or_b():
    with pytest.raises(ValueError, match="must be 'A' or 'B'"):
        build_question(
            index=1,
            field="industry_vertical",
            question="Which vertical?",
            option_a="retail",
            option_b="finance",
            recommended="C",
            recommendation_reason="n/a",
        )


# ------------------------------------------------------------------ the gate


def test_off_vocabulary_audience_terms_are_rejected():
    """An unknown term does not fail loudly downstream — it collapses the score to 0.5."""
    out = _ask(questions=[{**AUDIENCE_Q, "option_a": "hipsters"}])
    assert out["status"] == "invalid"
    assert "hipsters" in out["detail"]
    assert clarifications.get_open(SESSION) is None


def test_every_askable_field_is_accepted_and_nothing_else_is():
    for field in ASKABLE_FIELDS:
        question = {
            "field": field,
            "question": f"What about {field}?",
            "option_a": "students" if field == "audience_terms" else "first",
            "option_b": "families" if field == "audience_terms" else "second",
            "recommended": "A",
            "recommendation_reason": "because",
        }
        assert _ask(questions=[question])["status"] == "asked", field

    assert _ask(questions=[{**AUDIENCE_Q, "field": "optimization_goal"}])["status"] == "invalid"
    assert _ask(questions=[{**AUDIENCE_Q, "field": "slots_per_day"}])["status"] == "invalid"


def test_question_cap_is_enforced():
    assert _ask(questions=[dict(AUDIENCE_Q)] * 3)["status"] == "asked"
    capped = _ask(questions=[dict(AUDIENCE_Q)] * 4)
    assert capped["status"] == "invalid"
    assert "limit is 3" in capped["detail"]


def test_empty_questions_is_rejected_with_a_pointer_to_building():
    out = _ask(questions=[])
    assert out["status"] == "invalid"
    assert "create_campaign_spec" in out["detail"]


def test_missing_required_keys_are_named():
    out = _ask(questions=[{"field": "industry_vertical", "question": "Which?"}])
    assert out["status"] == "invalid"
    assert "option_a" in out["detail"] and "option_b" in out["detail"]


def test_the_result_tells_the_agent_to_stop():
    """Asking and then building anyway is the failure mode that makes the gate pointless."""
    out = _ask()
    assert "END YOUR TURN NOW" in out["detail"]
    assert "Do not call create_campaign_spec" in out["detail"]


# --------------------------------------------------------------- round lifecycle


def test_asking_twice_replaces_rather_than_stacks():
    _ask()
    _ask(questions=[{**AUDIENCE_Q, "question": "Second ask?"}])
    open_round = clarifications.get_open(SESSION)
    assert len(open_round.questions) == 1
    assert open_round.questions[0].question == "Second ask?"


def test_creating_a_spec_closes_the_open_round():
    """The pipeline starting is the definition of 'past asking'."""
    _ask()
    assert clarifications.get_open(SESSION) is not None

    created = master_tools.create_campaign_spec.invoke(
        {
            "campaign_objective": "reach commuters",
            "optimization_goal": "reach",
            "start_date": (date.today() + timedelta(days=14)).isoformat(),
            "duration_days": 30,
            "budget": 50_000.0,
            "city_ids": ["LH"],
            "zone_ids": ["LH-ZONE-001"],
            "audience_terms": ["commuters"],
            "session_id": SESSION,
        }
    )
    assert created["status"] == "ok"
    assert clarifications.get_open(SESSION) is None


def test_a_session_with_no_round_reports_none():
    assert clarifications.get_open("ses-never-asked") is None


def test_audience_vocabulary_matches_the_spec_contract():
    """The tool's allowed values must not drift from what the engine actually scores."""
    assert set(AUDIENCE_TERMS) == {
        "young_professionals",
        "professionals",
        "students",
        "families",
        "high_income",
        "commuters",
    }
