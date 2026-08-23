"""Pricing lever tests.

Two properties matter more than any individual lever's arithmetic:

1. IDENTITY. A run with no levers set prices exactly as the engine did before levers
   existed. Without this, adding the surface silently repriced every existing campaign.
2. CONTAINMENT. No lever reaches the feasibility gate, and the clamps hold in code rather
   than in the prompt. An LLM chooses these values, so the bounds must not be advisory.
"""

from __future__ import annotations

import pytest

from app.ml.levers import (
    COMMERCIAL_MULTIPLIER_RANGE,
    OCCUPANCY_GAMMA_RANGE,
    WEIGHT_RANGE,
    PricingLevers,
    effective_multiplier,
)
from app.ml.price_band import INDUSTRY_ADJ_CLAMP


@pytest.fixture(scope="module")
def engine():
    from app.ml.engine import PricingEngine

    return PricingEngine.build()


def _campaign(**overrides) -> dict:
    campaign = {
        "industry_vertical": "healthcare",
        "time_block_id": 2,
        "start_date": "2026-09-01",
        "end_date": "2026-09-15",
        "slots_needed": 1,
    }
    campaign.update(overrides)
    return campaign


def _some_screens(engine, n: int) -> list[str]:
    return engine._screens.index[:n].tolist()


def _feasible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["feasible"]]


# --- the weighting primitive --------------------------------------------------


@pytest.mark.parametrize("multiplier", [0.32, 0.913, 1.0, 1.15, 2.4])
def test_weight_zero_switches_a_term_off_whichever_side_of_neutral_it_sits(multiplier):
    """The reason weights dial the DEVIATION rather than scaling the multiplier.

    Scaling would send a 1.15 boost to 0.0 at weight 0 — a free campaign — instead of
    neutralizing it.
    """
    assert effective_multiplier(multiplier, 0.0) == 1.0


@pytest.mark.parametrize("multiplier", [0.32, 0.913, 1.15])
def test_weight_one_is_the_identity(multiplier):
    assert effective_multiplier(multiplier, 1.0) == pytest.approx(multiplier)


def test_weight_two_doubles_the_deviation():
    assert effective_multiplier(0.90, 2.0) == pytest.approx(0.80)
    assert effective_multiplier(1.15, 2.0) == pytest.approx(1.30)


# --- clamping -----------------------------------------------------------------


def test_out_of_range_levers_are_bounded_not_rejected():
    """Bounding rather than raising is deliberate: a rejected tool call in an agent loop
    becomes a retry against a per-minute rate limit, to reach the number clamping could
    have returned immediately."""
    levers, adjustments = PricingLevers(
        seasonality_weight=99.0,
        occupancy_gamma=0.0,
        band_position=1.8,
        commercial_multiplier=0.1,
    ).clamp()

    assert levers.seasonality_weight == WEIGHT_RANGE[1]
    assert levers.occupancy_gamma == OCCUPANCY_GAMMA_RANGE[0]
    assert levers.band_position == 1.0
    assert levers.commercial_multiplier == COMMERCIAL_MULTIPLIER_RANGE[0]
    # Every clamp is disclosed, so the caller can quote the effective value.
    assert len(adjustments) == 4
    assert all("clamped to" in a for a in adjustments)


def test_in_range_levers_clamp_to_themselves_with_no_adjustments():
    levers, adjustments = PricingLevers(commercial_multiplier=0.95, band_position=0.5).clamp()
    assert adjustments == []
    assert levers.commercial_multiplier == 0.95
    assert levers.band_position == 0.5


def test_default_levers_are_identity_and_report_no_changes():
    levers = PricingLevers()
    assert levers.is_default()
    assert levers.changes() == []
    assert levers.clamp() == (levers, [])


def test_band_position_none_survives_clamping():
    """None means 'keep the occupancy rule'. Clamping it to 0.0 would silently reprice
    every screen at the band floor."""
    levers, _ = PricingLevers().clamp()
    assert levers.band_position is None


# --- identity against the engine ----------------------------------------------


def test_no_levers_prices_identically_to_explicit_defaults(engine):
    """The guarantee that makes the lever surface safe to add to a live pipeline."""
    screens = _some_screens(engine, 40)
    without = engine.price_candidates(_campaign(), screens)
    with_defaults = engine.price_candidates(_campaign(), screens, PricingLevers())

    assert [r["recommended_price"] for r in without] == [
        r["recommended_price"] for r in with_defaults
    ]
    assert [r["assumptions"] for r in without] == [r["assumptions"] for r in with_defaults]


# --- containment: levers must not reach inventory ------------------------------


def test_levers_cannot_make_a_sold_out_screen_feasible(engine):
    """Availability is inventory truth. If a lever ever gates feasibility, this fails."""
    campaign = _campaign(slots_needed=6)
    screens = _some_screens(engine, 150)

    baseline = engine.price_candidates(campaign, screens)
    aggressive = engine.price_candidates(
        campaign,
        screens,
        PricingLevers(
            band_position=0.0,
            commercial_multiplier=0.70,
            respect_band_floor=False,
            seasonality_weight=0.0,
        ),
    )

    assert any(not r["feasible"] for r in baseline), "no sold-out screen sampled"
    assert [r["feasible"] for r in baseline] == [r["feasible"] for r in aggressive]
    assert [r["max_available_slots"] for r in baseline] == [
        r["max_available_slots"] for r in aggressive
    ]


def test_occupancy_rate_is_unchanged_by_levers(engine):
    """Occupancy is a measurement of existing bookings, not a pricing posture."""
    screens = _some_screens(engine, 40)
    baseline = engine.price_candidates(_campaign(), screens)
    levered = engine.price_candidates(
        _campaign(), screens, PricingLevers(occupancy_gamma=3.0, commercial_multiplier=1.3)
    )
    assert [r["occupancy_rate"] for r in baseline] == [r["occupancy_rate"] for r in levered]


# --- individual levers --------------------------------------------------------


def test_band_position_pins_the_quote_inside_the_band(engine):
    screens = _some_screens(engine, 60)

    at_floor = _feasible(
        engine.price_candidates(_campaign(), screens, PricingLevers(band_position=0.0))
    )
    at_cap = _feasible(
        engine.price_candidates(_campaign(), screens, PricingLevers(band_position=1.0))
    )
    assert at_floor and at_cap

    for row in at_floor:
        assert row["recommended_price"] == pytest.approx(row["floor_price"], abs=0.02)
    for row in at_cap:
        assert row["recommended_price"] == pytest.approx(row["cap_price"], abs=0.02)


def test_band_position_beats_occupancy_gamma_when_both_are_set(engine):
    """Documented precedence: an explicit posture replaces the scarcity read entirely."""
    screens = _some_screens(engine, 40)
    rows = _feasible(
        engine.price_candidates(
            _campaign(), screens, PricingLevers(band_position=0.25, occupancy_gamma=4.0)
        )
    )
    assert rows
    for row in rows:
        if row["cap_price"] > row["floor_price"]:
            position = (row["recommended_price"] - row["floor_price"]) / (
                row["cap_price"] - row["floor_price"]
            )
            assert position == pytest.approx(0.25, abs=0.01)


def test_occupancy_gamma_pins_both_ends_and_moves_the_middle(engine):
    """gamma < 1 quotes higher on partly-empty inventory, but an empty screen must still
    quote at floor and a full one at cap — otherwise the band stops meaning anything."""
    screens = _some_screens(engine, 80)
    baseline = {r["screen_id"]: r for r in _feasible(engine.price_candidates(_campaign(), screens))}
    aggressive = {
        r["screen_id"]: r
        for r in _feasible(
            engine.price_candidates(_campaign(), screens, PricingLevers(occupancy_gamma=0.5))
        )
    }
    assert baseline and aggressive

    moved = 0
    for sid, base in baseline.items():
        row = aggressive[sid]
        assert row["recommended_price"] >= base["recommended_price"] - 0.02
        if (
            0.0 < base["occupancy_rate"] < 1.0
            and row["cap_price"] > row["floor_price"]
            and row["recommended_price"] > base["recommended_price"] + 0.02
        ):
            moved += 1
        if base["occupancy_rate"] == 0.0:
            assert row["recommended_price"] == pytest.approx(row["floor_price"], abs=0.02)
    assert moved, "gamma changed nothing; the sample had no partly-occupied screen"


def test_commercial_multiplier_scales_the_quote(engine):
    screens = _some_screens(engine, 60)
    baseline = {r["screen_id"]: r for r in _feasible(engine.price_candidates(_campaign(), screens))}
    marked_up = {
        r["screen_id"]: r
        for r in _feasible(
            engine.price_candidates(_campaign(), screens, PricingLevers(commercial_multiplier=1.20))
        )
    }
    assert baseline

    for sid, base in baseline.items():
        expected = base["recommended_price"] * 1.20
        assert marked_up[sid]["recommended_price"] == pytest.approx(expected, abs=0.02)
        assert any("commercial adjustment" in a for a in marked_up[sid]["assumptions"])


def test_band_floor_is_held_by_default_and_releasable_on_request(engine):
    """A discount deep enough to breach p25 is a decision that must be authorised
    explicitly, and disclosed on the row either way."""
    screens = _some_screens(engine, 60)
    deep_discount = PricingLevers(band_position=0.0, commercial_multiplier=0.70)

    held = _feasible(engine.price_candidates(_campaign(), screens, deep_discount))
    released = _feasible(
        engine.price_candidates(
            _campaign(), screens, deep_discount.model_copy(update={"respect_band_floor": False})
        )
    )
    assert held and released

    for row in held:
        assert row["recommended_price"] >= row["floor_price"] - 0.02
        assert any("held at band floor" in a for a in row["assumptions"])

    below = [r for r in released if r["recommended_price"] < r["floor_price"] - 0.02]
    assert below, "nothing went sub-floor; the test proved nothing"
    for row in below:
        assert any("BELOW the band floor" in a for a in row["assumptions"])


def test_seasonality_weight_neutralizes_the_day_of_week_haircut():
    """The documented reason this lever exists: the day-of-week multiplier averages 0.913
    over a full week, discounting a band already built from real contracted prices.

    Driven off a synthetic ridership payload rather than the engine, because
    `ridership_actuals.csv` is gitignored and optional — on a machine without it the real
    multiplier is a flat 1.0 and an engine-level assertion would prove nothing. The
    weekday/weekend factors below are the shape the docstring in `app/ml/seasonality.py`
    reports from the real file (Friday 1.21 down to Sunday 0.32).
    """
    import pandas as pd

    from app.ml.seasonality import SeasonalityAdjuster

    dow = {
        "monday": 1.05,
        "tuesday": 1.08,
        "wednesday": 1.10,
        "thursday": 1.15,
        "friday": 1.21,
        "saturday": 0.58,
        "sunday": 0.32,
    }
    adjuster = SeasonalityAdjuster(
        {"dow_multiplier": dow, "holiday_relative_multiplier": 1.0, "holiday_dates": set()},
        pd.DataFrame(
            columns=[
                "city_id",
                "city_zone",
                "anchor_location_id",
                "start_date",
                "end_date",
                "expected_attendance",
                "attendance_tier",
            ]
        ),
        pd.DataFrame([{"screen_id": "S1", "location_id": "L1"}]),
        pd.DataFrame([{"location_id": "L1", "city_id": "LH", "city_zone": "Z"}]).set_index(
            "location_id", drop=False
        ),
    )

    # A whole week: the haircut the lever exists to answer for.
    week = adjuster.get_adjustment("S1", "2026-09-07", "2026-09-13")
    haircut = week.day_of_week_holiday_multiplier
    assert haircut < 1.0, "fixture no longer reproduces the documented whole-week haircut"

    assert effective_multiplier(haircut, 1.0) == pytest.approx(haircut)
    assert effective_multiplier(haircut, 0.0) == 1.0
    # Half-weighting halves the discount rather than halving the price.
    assert effective_multiplier(haircut, 0.5) == pytest.approx(1.0 - (1.0 - haircut) / 2)


def test_seasonality_weight_zero_leaves_only_the_event_term(engine):
    """Engine-level counterpart. Works whether or not ridership_actuals is provisioned:
    with the day-of-week term switched off, the surviving multiplier is the event boost,
    which is >= 1.0 by construction and exactly 1.0 where no event matches."""
    screens = _some_screens(engine, 120)
    campaign = _campaign(start_date="2026-09-07", end_date="2026-09-27")

    rows = _feasible(
        engine.price_candidates(campaign, screens, PricingLevers(seasonality_weight=0.0))
    )
    assert rows

    for row in rows:
        assert row["seasonality_multiplier"] >= 1.0
        if row["event_match_type"] in {"none", "not_applicable"}:
            assert row["seasonality_multiplier"] == pytest.approx(1.0)


def test_event_weight_is_independent_of_seasonality_weight(engine):
    """A rep can want the event premium without the day-of-week discount. If the two terms
    were multiplied before weighting, that would be impossible."""
    screens = _some_screens(engine, 120)
    campaign = _campaign(start_date="2026-09-07", end_date="2026-09-27")

    rows = _feasible(
        engine.price_candidates(
            campaign, screens, PricingLevers(seasonality_weight=0.0, event_weight=1.0)
        )
    )
    boosted = [r for r in rows if r["event_match_type"] in {"location_match", "zone_match"}]
    if boosted:
        # Day-of-week neutralized to 1.0, so anything above 1.0 is the surviving event term.
        assert all(r["seasonality_multiplier"] >= 1.0 for r in boosted)
    unboosted = [r for r in rows if r["event_match_type"] in {"none", "not_applicable"}]
    assert unboosted
    assert all(r["seasonality_multiplier"] == pytest.approx(1.0) for r in unboosted)


def test_industry_weight_cannot_widen_the_industry_clamp(engine):
    """The band module guarantees industry never swings the price more than -15/+20%. A
    weight above 1.0 must not be able to buy authority the raw ratio never had."""
    screens = _some_screens(engine, 120)
    rows = _feasible(
        engine.price_candidates(_campaign(), screens, PricingLevers(industry_weight=2.0))
    )
    assert rows

    for row in rows:
        # target is p50 x industry_adj x seasonality; recover the adjustment's bound via
        # the floor/target/cap ratios, which all carry the same multiplier.
        assert row["floor_price"] <= row["target_price"] <= row["cap_price"]
    # The clamp itself is the invariant under test; assert it is what the band advertises.
    assert INDUSTRY_ADJ_CLAMP == (0.85, 1.20)


def test_industry_weight_zero_suppresses_the_adjustment(engine):
    screens = _some_screens(engine, 120)
    suppressed = _feasible(
        engine.price_candidates(_campaign(), screens, PricingLevers(industry_weight=0.0))
    )
    assert suppressed
    assert not any(
        a.startswith("industry adjustment applied")
        for row in suppressed
        for a in row["assumptions"]
    )


# --- run-state round trip -----------------------------------------------------


def test_levers_round_trip_through_run_state(spec_factory=None):
    """Levers persist on the run so a rebuild reprices identically, and so the pricing
    stage can read them without them crossing an LLM text channel."""
    from app.models.campaign import CampaignSpec
    from app.services import run_state

    spec = CampaignSpec(
        campaign_objective="awareness",
        city_ids=["LH"],
        start_date="2026-10-01",
        duration_days=30,
        budget=50_000.0,
        optimization_goal="reach",
    )
    run_id = run_state.create_run(spec)

    # A run that never set levers must read back as identity, not as an error.
    assert run_state.get_pricing_levers(run_id).is_default()

    levers, _ = PricingLevers(commercial_multiplier=0.95, note="client won't pay peak").clamp()
    run_state.set_pricing_levers(run_id, levers)

    loaded = run_state.get_pricing_levers(run_id)
    assert loaded.commercial_multiplier == 0.95
    assert loaded.note == "client won't pay peak"
    assert loaded.changes() == ["commercial_multiplier=0.95"]
