"""The shapes an LLM actually passes, and what the tool boundary makes of them.

Every tool here is called by a language model, so its argument types are a contract with
something that emits *plausible* JSON rather than typed JSON. These tests pin the three
failures that were live and silent -- a bare string where a list was declared, a JSON
string where an object was declared, and a comma-separated string where a closed
vocabulary was declared -- plus the rule that a bad SHAPE is always a recoverable result
and never an exception.

Why recoverable matters more than correct: raising out of a tool aborts the SSE stream and
the rep loses the turn. A `status: "invalid"` result costs one model call.
"""

from __future__ import annotations

import pytest

from app.models.campaign import (
    AUDIENCE_TERMS,
    DISPLAY_TYPE_NON_CONSTRAINTS,
    ENFORCED_HARD_CONSTRAINTS,
    HARD_CONSTRAINT_SHAPES,
    SCREEN_TYPES,
    TIME_BLOCK_IDS,
)
from app.services import clarifications
from app.tools import coerce
from app.tools.coerce import ArgumentError

# --------------------------------------------------------------- vocabularies


def test_enforced_constraints_are_exactly_the_typed_ones():
    """A key that is enforced but untyped is the state that shipped a broken filter."""
    assert frozenset(HARD_CONSTRAINT_SHAPES) == ENFORCED_HARD_CONSTRAINTS


def test_every_constraint_shape_is_one_this_module_handles():
    assert set(HARD_CONSTRAINT_SHAPES.values()) <= {
        "int",
        "int_passthrough",
        "fraction",
        "screen_types",
        "time_blocks",
        "id_list",
    }


def test_display_type_keys_are_not_enforced_constraints():
    """They are stripped, so they must not also be in the enforced vocabulary."""
    assert not (DISPLAY_TYPE_NON_CONSTRAINTS & ENFORCED_HARD_CONSTRAINTS)


# --------------------------------------------------------------- as_str_list


def test_a_bare_string_becomes_a_one_item_list_not_one_item_per_letter():
    """THE bug: `list("metro_station")` is 13 screen types that match no screen."""
    assert coerce.as_str_list("metro_station", field="t") == ["metro_station"]


def test_a_separated_string_becomes_its_parts():
    assert coerce.as_str_list("young_professionals, commuters", field="t") == [
        "young_professionals",
        "commuters",
    ]
    assert coerce.as_str_list("LH-ZONE-001; LH-ZONE-002", field="t") == [
        "LH-ZONE-001",
        "LH-ZONE-002",
    ]


def test_a_json_array_written_as_a_string_is_parsed():
    assert coerce.as_str_list('["2", "5"]', field="t") == ["2", "5"]
    # A model that has been thinking in Python emits repr, which is not JSON.
    assert coerce.as_str_list("['2', '5']", field="t") == ["2", "5"]


def test_numbers_become_the_strings_the_vocabularies_are_written_in():
    """Time blocks arrive as `[2, 3]` and `[2.0, 3.0]` about as often as `["2", "3"]`."""
    assert coerce.as_str_list([2, 3], field="t") == ["2", "3"]
    assert coerce.as_str_list([2.0, 3.0], field="t") == ["2", "3"]


def test_the_ways_a_model_writes_nothing_all_mean_nothing():
    for empty in (None, "", "none", "N/A", "null", "[]"):
        assert coerce.as_str_list(empty, field="t") == []


def test_duplicates_collapse_but_order_is_kept():
    assert coerce.as_str_list(["b", "a", "b"], field="t") == ["b", "a"]


def test_a_vocabulary_normalizes_case_and_separators():
    assert coerce.as_str_list("Metro Station", field="t", vocabulary=SCREEN_TYPES) == [
        "metro_station"
    ]
    assert coerce.as_str_list(["METRO-STATION"], field="t", vocabulary=SCREEN_TYPES) == [
        "metro_station"
    ]


def test_an_off_vocabulary_value_raises_rather_than_passing_through():
    """An unmatched term does not fail loudly downstream; it neutralizes a sub-score."""
    with pytest.raises(ArgumentError, match="does not accept"):
        coerce.as_str_list("billboards", field="t", vocabulary=SCREEN_TYPES)


def test_the_error_names_the_field_and_the_allowed_values():
    with pytest.raises(ArgumentError) as exc:
        coerce.as_str_list("fintech", field="industry_vertical", vocabulary=AUDIENCE_TERMS)
    message = str(exc.value)
    assert "industry_vertical" in message
    assert "young_professionals" in message


def test_a_wrapper_object_is_refused_with_the_shape_it_wants():
    with pytest.raises(ArgumentError, match="takes a list of strings"):
        coerce.as_str_list({"screen_types": ["bus"]}, field="allowed_screen_types")


# --------------------------------------------------------------- as_dict


def test_a_json_object_written_as_a_string_is_parsed():
    assert coerce.as_dict('{"min_screens": 20}', field="hard_constraints") == {"min_screens": 20}
    assert coerce.as_dict("{'min_screens': 20}", field="hard_constraints") == {"min_screens": 20}


def test_a_list_where_an_object_was_declared_is_refused():
    with pytest.raises(ArgumentError, match="takes an object"):
        coerce.as_dict(["min_screens"], field="hard_constraints")


def test_as_dict_list_accepts_a_single_object():
    """A model asked for one question inside a list sends the question."""
    assert coerce.as_dict_list({"field": "budget"}, field="questions") == [{"field": "budget"}]


def test_as_dict_list_parses_a_serialized_list_and_serialized_entries():
    assert coerce.as_dict_list('[{"field": "budget"}]', field="questions") == [{"field": "budget"}]
    assert coerce.as_dict_list(['{"field": "budget"}'], field="questions") == [{"field": "budget"}]


# --------------------------------------------------------------- numbers


def test_a_number_written_as_prose_is_read():
    assert coerce.as_int("50,000", field="budget") == 50000
    assert coerce.as_int("$1_200", field="budget") == 1200


def test_a_flag_and_a_fraction_are_refused_rather_than_truncated():
    """`int(True)` is 1 and `int(1.5)` is 1 -- coercion would enforce a nonsense value."""
    with pytest.raises(ArgumentError, match="flag"):
        coerce.as_int(True, field="max_slots_per_day")
    with pytest.raises(ArgumentError, match="whole number"):
        coerce.as_int(1.5, field="max_slots_per_day")


def test_a_percentage_becomes_a_fraction_because_the_other_reading_is_unsatisfiable():
    assert coerce.as_fraction(90, field="min_budget_utilization") == pytest.approx(0.9)
    assert coerce.as_fraction("90%", field="min_budget_utilization") == pytest.approx(0.9)
    assert coerce.as_fraction(0.9, field="min_budget_utilization") == pytest.approx(0.9)
    assert coerce.as_fraction(1, field="min_budget_utilization") == pytest.approx(1.0)


def test_a_share_above_one_hundred_percent_is_refused():
    with pytest.raises(ArgumentError):
        coerce.as_fraction(900, field="min_budget_utilization")


def test_clamp_reports_rather_than_rejecting():
    """A rejected call in an agent loop is a retry against a per-minute rate limit."""
    value, note = coerce.clamp_int(9000, field="top_n", low=1, high=2000)
    assert value == 2000
    assert note and "9000" in note and "2000" in note

    value, note = coerce.clamp_int(250, field="top_n", low=1, high=2000)
    assert (value, note) == (250, None)

    assert coerce.clamp_int(None, field="top_n", low=1, high=2000) == (None, None)


# --------------------------------------------------------------- hard constraints


def test_a_single_screen_type_as_a_string_is_read_as_a_one_item_list():
    """The whole point: this used to empty the candidate pool and blame the brief."""
    out, notes = coerce.normalize_hard_constraints({"allowed_screen_types": "metro_station"})
    assert out == {"allowed_screen_types": ["metro_station"]}
    assert notes and "allowed_screen_types" in notes[0]


def test_the_whole_constraints_dict_may_arrive_as_a_string():
    out, _ = coerce.normalize_hard_constraints('{"max_slots_per_day": 1}')
    assert out == {"max_slots_per_day": 1}


def test_each_key_is_coerced_to_its_declared_type():
    out, _ = coerce.normalize_hard_constraints(
        {
            "min_screens": "12",
            "min_budget_utilization": "85%",
            "required_time_blocks": [2, 5],
            "excluded_screen_types": "bus",
        }
    )
    assert out["min_screens"] == 12
    assert out["min_budget_utilization"] == pytest.approx(0.85)
    assert out["required_time_blocks"] == ["2", "5"]
    assert out["excluded_screen_types"] == ["bus"]


def test_an_unknown_key_is_rejected_at_intake_naming_the_vocabulary():
    """Cheaper here than at verification, where the rep already believes it was honoured."""
    with pytest.raises(ArgumentError, match="no stage enforces"):
        coerce.normalize_hard_constraints({"max_cost_per_screen": 500})


def test_an_off_vocabulary_screen_type_is_rejected():
    with pytest.raises(ArgumentError, match="does not accept"):
        coerce.normalize_hard_constraints({"allowed_screen_types": ["billboard"]})


def test_a_daypart_name_in_required_time_blocks_is_rejected():
    with pytest.raises(ArgumentError) as exc:
        coerce.normalize_hard_constraints({"required_time_blocks": ["morning"]})
    assert "1" in str(exc.value) and "6" in str(exc.value)


def test_a_display_type_key_is_dropped_and_disclosed_not_rejected():
    """Every screen is digital, so the constraint is already satisfied by all of them."""
    key = min(DISPLAY_TYPE_NON_CONSTRAINTS)
    out, notes = coerce.normalize_hard_constraints({key: True, "min_screens": 5})
    assert out == {"min_screens": 5}
    assert any(key in note for note in notes)


def test_an_empty_value_is_dropped_rather_than_enforced_as_an_empty_filter():
    out, notes = coerce.normalize_hard_constraints({"allowed_screen_types": []})
    assert out == {}
    assert notes


def test_time_block_vocabulary_covers_the_six_real_blocks():
    assert TIME_BLOCK_IDS == ("1", "2", "3", "4", "5", "6")


# ================================================================ at the tool boundary
#
# The coercion above is only worth having if the tools actually use it. These call the
# real tools with the shapes a model sends, and assert the two properties that matter:
# a usable shape gets through, and an unusable one comes back as a RESULT rather than an
# exception. A raise here would abort the SSE stream and cost the rep the turn.

from app.services import local_db, run_state  # noqa: E402
from app.tools import master_tools, or_agent_tools, relevance_tools  # noqa: E402

_BRIEF = {
    "campaign_objective": "Launch awareness for a retail client",
    "optimization_goal": "reach",
    "start_date": "2026-10-01",
    "duration_days": 30,
    "budget": 50000.0,
}


def _spec(**overrides) -> dict:
    return master_tools.create_campaign_spec.invoke({**_BRIEF, **overrides})


def test_intake_accepts_the_shapes_a_model_actually_sends():
    """Geography as a bare string, terms comma-separated, constraints as a JSON string."""
    out = _spec(
        city_ids="LH",
        audience_terms="young_professionals, commuters",
        preferred_time_blocks=[2, 5],
        hard_constraints='{"allowed_screen_types": "metro_station", "max_slots_per_day": 1}',
    )
    assert out["status"] == "ok"
    spec = run_state.get_spec(out["run_id"])
    assert spec.city_ids == ["LH"]
    assert spec.audience_terms == ["young_professionals", "commuters"]
    assert spec.preferred_time_blocks == ["2", "5"]
    # The one that used to empty the pool: a list, not thirteen characters.
    assert spec.hard_constraints["allowed_screen_types"] == ["metro_station"]
    assert spec.hard_constraints["max_slots_per_day"] == 1


def test_intake_reports_what_it_reshaped():
    out = _spec(city_ids="LH", hard_constraints={"excluded_screen_types": "bus"})
    assert out["status"] == "ok"
    assert any("excluded_screen_types" in note for note in out["argument_notes"])


def test_intake_rejects_an_unenforced_constraint_key_without_raising():
    out = _spec(city_ids=["LH"], hard_constraints={"max_cost_per_screen": 500})
    assert out["status"] == "invalid"
    assert "max_cost_per_screen" in out["errors"]


def test_intake_rejects_an_off_vocabulary_audience_term_without_raising():
    out = _spec(city_ids=["LH"], audience_terms="tourists")
    assert out["status"] == "invalid"
    assert "young_professionals" in out["errors"]


def test_screen_type_mix_is_reachable_from_intake():
    """It is a real CampaignSpec field, and no agent could set it until now."""
    out = _spec(city_ids=["LH"], screen_type_mix="metro_station, bus")
    assert out["status"] == "ok"
    assert out["normalized_spec"]["screen_type_mix"] == ["metro_station", "bus"]
    assert run_state.get_spec(out["run_id"]).screen_type_mix == ["metro_station", "bus"]


def test_geography_resolution_takes_a_bare_place_name():
    out = master_tools.resolve_geography_terms.invoke({"terms": "Las Hackland"})
    assert out.get("city_ids") == ["LH"]


def test_geography_resolution_splits_a_separated_string():
    out = master_tools.resolve_geography_terms.invoke({"terms": "Las Hackland, Downtown Core"})
    assert out.get("city_ids") == ["LH"]
    assert out.get("zone_ids")


def test_check_explanations_takes_a_single_screen_id_as_a_string():
    out = _spec(city_ids=["LH"])
    result = master_tools.check_explanations.invoke(
        {"run_id": out["run_id"], "explained_screen_ids": "SCR-000001"}
    )
    # No package on this run, but the ARGUMENT was accepted rather than exploding.
    assert result["status"] == "error"
    assert "No package" in result["detail"]


def test_clarifying_questions_reject_a_session_that_does_not_exist():
    """Stored against an unknown session, the rep sees a request and no options."""
    out = master_tools.ask_clarifying_questions.invoke(
        {
            "session_id": "ses-does-not-exist",
            "understood": "A retail launch in Las Hackland.",
            "questions": [
                {
                    "field": "audience_terms",
                    "question": "Who is this for?",
                    "option_a": "commuters",
                    "option_b": "students",
                    "recommended": "A",
                    "recommendation_reason": "the brief mentions the morning peak",
                }
            ],
        }
    )
    assert out["status"] == "invalid"
    assert "ses-does-not-exist" in out["detail"]


def test_clarifying_questions_accept_a_single_object_and_normalize_its_values():
    session = local_db.insert(local_db.SESSIONS, {"title": "t"})
    out = master_tools.ask_clarifying_questions.invoke(
        {
            "session_id": session["id"],
            "understood": "A retail launch in Las Hackland.",
            # Not wrapped in a list, and the values are title-cased. Both used to fail.
            "questions": {
                "field": "audience_terms",
                "question": "Who is this for?",
                "option_a": "Young Professionals",
                "option_b": "Students",
                "recommended": "A",
                "recommendation_reason": "the brief mentions the morning peak",
            },
        }
    )
    assert out["status"] == "asked"
    pending = clarifications.get_open(session["id"])
    values = [o.value for o in pending.questions[0].options if o.kind == "answer"]
    # Canonical tokens, so what the rep clicks is what the spec accepts.
    assert values == ["young_professionals", "students"]


def test_candidate_pool_size_is_bounded_and_disclosed():
    out = _spec(city_ids=["LH"])
    result = relevance_tools.build_screen_candidates.invoke(
        {"run_id": out["run_id"], "top_n": 99999}
    )
    assert result["status"] == "ok"
    assert result["candidates_selected"] <= relevance_tools.MAX_CANDIDATE_POOL
    assert any("top_n" in note for note in result["argument_notes"])


def test_compare_objectives_rejects_an_unknown_objective_without_raising():
    out = _spec(city_ids=["LH"])
    result = or_agent_tools.compare_objectives.invoke(
        {"run_id": out["run_id"], "objectives": "footfall"}
    )
    assert result["status"] == "invalid"
    assert "footfall" in result["errors"]


# --------------------------------------------------------------- levers merge


def test_levers_merge_across_calls_instead_of_resetting_what_was_not_passed():
    """A lever the rep set two turns ago is still their instruction."""
    run_id = _spec(city_ids=["LH"])["run_id"]

    first = master_tools.set_pricing_levers.invoke(
        {"run_id": run_id, "commercial_multiplier": 0.95, "note": "competing bid"}
    )
    assert first["status"] == "ok"

    second = master_tools.set_pricing_levers.invoke({"run_id": run_id, "seasonality_weight": 0.0})
    levers = run_state.get_pricing_levers(run_id)
    assert levers.commercial_multiplier == 0.95, "an omitted lever was silently reset"
    assert levers.seasonality_weight == 0.0
    assert second["levers_you_set_this_call"] == ["seasonality_weight"]
    assert "commercial_multiplier" in second["carried_from_earlier_calls"]


def test_reset_clears_earlier_levers():
    run_id = _spec(city_ids=["LH"])["run_id"]
    master_tools.set_pricing_levers.invoke({"run_id": run_id, "commercial_multiplier": 0.95})
    master_tools.set_pricing_levers.invoke({"run_id": run_id, "reset": True})
    assert run_state.get_pricing_levers(run_id).is_default()


def test_demand_premium_weight_is_reachable_from_the_tool():
    """The documented way to price purely off comparables, previously unreachable."""
    run_id = _spec(city_ids=["LH"])["run_id"]
    out = master_tools.set_pricing_levers.invoke({"run_id": run_id, "demand_premium_weight": 0.0})
    assert out["status"] == "ok"
    assert run_state.get_pricing_levers(run_id).demand_premium_weight == 0.0
    assert "demand_premium_weight=0" in out["changes_from_default"]


# --------------------------------------------------------------- agent tool surfaces


def test_the_master_is_not_told_to_call_a_tool_it_does_not_have():
    """The prompt named `compare_objectives`, which is the OR agent's tool."""
    from app.agents.prompts import MASTER_SYSTEM_PROMPT

    master_tool_names = {t.name for t in master_tools.TOOLS} | {
        t.name for t in relevance_tools.TOOLS
    }
    specialist_only = {t.name for t in or_agent_tools.TOOLS} - master_tool_names
    for name in specialist_only:
        if name in MASTER_SYSTEM_PROMPT:
            # Naming it to say it is NOT the Master's is fine; instructing a call is not.
            assert "You cannot call it" in MASTER_SYSTEM_PROMPT, (
                f"the Master prompt references {name}, which is not on its tool surface"
            )
