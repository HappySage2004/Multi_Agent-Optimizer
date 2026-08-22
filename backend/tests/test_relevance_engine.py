"""Invariants the audience relevance engine must hold.

No LLM and no API key. These assert the properties that were actually broken or dangerous
during integration, rather than restating the implementation:

  * contract bounds hold on real data (family_score reached 1.140 before renormalization)
  * geography is a HARD filter, not a soft penalty — the validator fails packages otherwise
  * pool_key deduplication materially changes the audience number
  * the exposure model is applied exactly once, and both sides of the reach min() are in
    viewed units
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.models.campaign import AUDIENCE_TERMS
from app.models.screens import ScreenCandidate
from app.optimize import exposure
from app.services.artifact_store import read_models
from app.tools import master_tools, ml_agent_tools, relevance_tools
from app.tools.relevance_tools import (
    ALL_DAY_TYPES,
    ALL_TIME_BLOCKS,
    SLOTS_PER_BLOCK,
    get_relevance_engine,
)

START = (date.today() + timedelta(days=14)).isoformat()


def _spec(**overrides) -> dict:
    payload = {
        "campaign_objective": "reach young commuters",
        "optimization_goal": "reach",
        "start_date": START,
        "duration_days": 30,
        "budget": 50_000.0,
        "city_ids": ["LH"],
        "zone_ids": ["LH-ZONE-001"],
        "industry_vertical": "technology",
        "audience_terms": ["young_professionals", "commuters"],
    }
    payload.update(overrides)
    return master_tools.create_campaign_spec.invoke(payload)


@pytest.fixture(scope="module")
def engine():
    return get_relevance_engine()


# --- feature layer ------------------------------------------------------------


def test_every_screen_is_profiled_with_all_twelve_impression_columns(engine):
    assert engine.screens == 11_163
    for block in ALL_TIME_BLOCKS:
        for day_type in ALL_DAY_TYPES:
            col = relevance_tools.block_column(block, day_type)
            assert col in engine.profile.columns
            assert engine.profile[col].isna().sum() == 0


def test_pool_key_is_never_null(engine):
    """It is the reach denominator. A null would silently make a screen its own audience."""
    assert engine.profile["pool_key"].isna().sum() == 0
    assert engine.profile["pool_key"].nunique() == 1_004


def test_block_one_is_empty_and_that_is_a_known_gap(engine):
    """Documented limitation, asserted so it cannot change without a decision.

    No scheduled service starts between 00:00 and 04:00, and the model has no ambient
    term, so block 1 is zero for every screen — despite block 1 having real bookings.
    """
    for day_type in ALL_DAY_TYPES:
        col = relevance_tools.block_column(1, day_type)
        assert engine.profile[col].sum() == 0.0


def test_audience_scores_stay_inside_the_contract_bounds(engine):
    """family_score peaked at 1.140 before renormalization and broke artifact writes."""
    for col in (
        "income_score",
        "professional_score",
        "young_adult_score",
        "student_score",
        "family_score",
        "commuter_score",
    ):
        assert engine.profile[col].min() >= 0.0, col
        assert engine.profile[col].max() <= 1.0, col


def test_every_audience_term_maps_to_a_real_score_column_and_blocks(engine):
    """The closed vocabulary and the engine's lookup tables must not drift apart."""
    for term in AUDIENCE_TERMS:
        columns = relevance_tools.AUDIENCE_TERM_TO_SCORE_COLUMN[term]
        assert columns, term
        for col in columns:
            assert col in engine.profile.columns, f"{term} -> {col}"
        assert relevance_tools.AUDIENCE_TO_PREFERRED_BLOCKS[term], term


# --- scoring ------------------------------------------------------------------


def test_relevance_and_every_subscore_stay_in_zero_to_one():
    run_id = _spec(audience_terms=["families"])["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 100})
    assert out["status"] == "ok"

    # Re-validating through the contract is the real assertion: the bounds live there.
    candidates = _candidates(out)
    assert len(candidates) == 100
    for c in candidates:
        assert 0.0 <= c.relevance_score <= 1.0
        assert 0.0 <= c.audience_match_score <= 1.0
        assert 0.0 <= c.time_of_day_score <= 1.0
        assert 0.0 <= c.historical_performance_score <= 1.0


def test_candidates_are_ranked_best_first():
    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 50})
    scores = [c.relevance_score for c in _candidates(out)]
    assert scores == sorted(scores, reverse=True)


def test_geography_is_a_hard_filter_not_a_penalty():
    """Every candidate must be inside the requested geography.

    The validator fails a package containing an ineligible screen, so a soft geography
    penalty would produce packages that cannot pass verification.
    """
    from app.data.reference import eligible_screen_ids

    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 250})
    eligible = eligible_screen_ids(["LH"], ["LH-ZONE-001"], [])
    assert out["candidates_selected"] == 250
    for c in _candidates(out):
        assert c.screen_id in eligible


def test_allowed_screen_types_is_enforced_before_scoring():
    run_id = _spec(hard_constraints={"allowed_screen_types": ["bus_stop"]})["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id})
    assert {c.screen_type for c in _candidates(out)} == {"bus_stop"}


def test_audience_terms_change_the_target_blocks_and_the_ranking():
    """The same inventory, scored for two audiences, must not produce the same answer."""
    commuters = relevance_tools.build_screen_candidates.invoke(
        {"run_id": _spec(audience_terms=["commuters"])["run_id"], "top_n": 50}
    )
    families = relevance_tools.build_screen_candidates.invoke(
        {"run_id": _spec(audience_terms=["families"])["run_id"], "top_n": 50}
    )
    assert commuters["target_time_blocks"] == [2, 5]
    assert families["target_time_blocks"] == [4, 5]
    assert [c.screen_id for c in _candidates(commuters)] != [
        c.screen_id for c in _candidates(families)
    ]


def test_unknown_audience_terms_are_rejected_deterministically():
    """An invented segment must not score as a neutral 0.5 — it must be refused."""
    rejected = _spec(audience_terms=["gen_z_skateboarders"])
    assert rejected["status"] == "invalid"
    assert "gen_z_skateboarders" in rejected["errors"]
    assert rejected["audience_terms_allowed"] == list(AUDIENCE_TERMS)


def test_missing_audience_terms_are_reported_not_hidden():
    run_id = _spec(audience_terms=[])["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 20})
    assert any("audience_terms" in note for note in out["defaults_applied"])


def test_reasons_cite_real_feature_values():
    """SOLUTION.md section 25: no generic 'highly relevant' text."""
    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 10})
    for c in _candidates(out):
        assert c.reasons
        joined = " ".join(c.reasons).lower()
        assert "highly relevant" not in joined
        # Every candidate cites at least one number.
        assert any(ch.isdigit() for ch in joined)


# --- pooling ------------------------------------------------------------------


def test_pooled_audience_is_far_below_the_naive_sum():
    """The whole reason pool_key exists. If these ever converge, dedupe has broken."""
    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 250})
    audience = out["audience"]
    assert audience["distinct_audience_pools"] < 250
    assert audience["pooled_daily_audience"] < audience["naive_daily_audience"]
    # On this inventory the gap is an order of magnitude, not a rounding difference.
    assert audience["naive_daily_audience"] > 5 * audience["pooled_daily_audience"]


def test_every_candidate_carries_a_pool_key():
    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 60})
    assert all(c.pool_key for c in _candidates(out))


# --- audience unit conversion -------------------------------------------------


def test_day_type_mix_counts_real_calendar_days():
    """A 30-day flight is not 30 weekdays; weekend ridership is ~6x lower."""
    mix = ml_agent_tools._day_type_mix(date(2026, 10, 5), 14)  # a Monday
    assert mix == {"weekday": 10, "weekend": 4}
    assert sum(ml_agent_tools._day_type_mix(date(2026, 10, 5), 30).values()) == 30


def test_viewed_exposures_apply_loop_passes_and_viewability_once():
    """One slot owns 1/6 of a block's continuously cycling airtime, and a passer-by gets
    LOOP_PASSES_PER_TRIP passes while in range — so exposures per slot are
    audience x 8/6 x viewability, and the reach ceiling is audience x viewability."""
    candidate = ScreenCandidate(
        screen_id="LH-SCR-000001",
        relevance_score=0.5,
        impressions_by_block={
            f"{b}_{dt}": (1200.0 if dt == "weekday" else 600.0)
            for b in ALL_TIME_BLOCKS
            for dt in ALL_DAY_TYPES
        },
    )
    mix = {"weekday": 20, "weekend": 10}
    daily = ml_agent_tools._daily_audience(candidate, "5", mix)
    assert daily == pytest.approx((20 * 1200.0 + 10 * 600.0) / 30)

    # Static inventory: 0.35 of passers-by look.
    assert exposure.viewability("metro_station") == 0.35
    assert exposure.viewed_exposures_per_slot_per_day(daily, "metro_station") == pytest.approx(
        daily * 8 / SLOTS_PER_BLOCK * 0.35
    )
    assert exposure.reachable_daily_audience(daily, "metro_station") == pytest.approx(daily * 0.35)

    # In-vehicle inventory is captive for the whole ride, so more of it looks.
    assert exposure.viewability("bus") == 0.65
    assert exposure.viewed_exposures_per_slot_per_day(daily, "bus") == pytest.approx(
        daily * 8 / SLOTS_PER_BLOCK * 0.65
    )

    # An unknown screen type takes the lower factor and says so, rather than being flattered.
    assert exposure.viewability("hovercraft") == 0.35
    assert exposure.is_viewability_assumed("hovercraft")
    assert not exposure.is_viewability_assumed("bus_stop")


def test_reach_ceiling_and_exposures_are_in_the_same_viewed_units():
    """Both sides of the downstream reach min() must be discounted, or a saturated plan
    claims every passer-by when only a third of them look."""
    audience = 10_000.0
    for screen_type in ("metro_station", "bus"):
        ceiling = exposure.reachable_daily_audience(audience, screen_type)
        per_slot = exposure.viewed_exposures_per_slot_per_day(audience, screen_type)
        assert ceiling < audience, "the ceiling must be viewed people, not everyone passing"
        assert per_slot == pytest.approx(ceiling * 8 / SLOTS_PER_BLOCK)


def test_economics_carry_audience_volume_and_the_pool_key():
    run_id = _spec()["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 40})
    out = ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    assert out["status"] == "ok"
    assert out["viewed_exposures_per_slot_per_day_mean"] > 0

    from app.models.economics import ScreenEconomics
    from app.services import run_state

    economics = read_models(run_state.require_artifact(run_id, "screen_economics"), ScreenEconomics)
    assert economics
    for e in economics:
        assert e.pool_key
        # People passing -> people who look -> exposures one slot earns in a day.
        assert e.reachable_daily_audience == pytest.approx(
            e.daily_unique_audience * (e.viewability_factor or 0.0)
        )
        assert e.viewed_exposures_per_slot_per_day == pytest.approx(
            e.reachable_daily_audience * 8 / SLOTS_PER_BLOCK
        )
        # The ceiling is a count of PEOPLE, so it can never exceed the crowd that passed.
        assert e.reachable_daily_audience <= e.daily_unique_audience
        if e.demand_forecast is not None:
            assert e.demand_forecast.demand_index >= 0.0


def _candidates(tool_output: dict) -> list[ScreenCandidate]:
    artifact_id = tool_output["artifact"].split("artifact_id=")[1].split(" ")[0]
    return read_models(artifact_id, ScreenCandidate)
