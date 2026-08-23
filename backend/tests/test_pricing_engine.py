"""Pricing engine tests, ported from the handoff package's standalone scripts.

The originals were print-based scripts with hardcoded `/home/claude/...` paths. The
invariants are preserved as real assertions; paths come from the DuckDB layer.

The engine is a session fixture because `build()` costs ~12 s. Tests that only need
reference data avoid it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.db import query_df
from app.ml.booking_probability import PRICE_DRIVEN_LOSS_REASONS
from app.ml.occupancy import SLOT_CAPACITY, OccupancyEngine
from app.ml.price_band import INDUSTRY_ADJ_CLAMP
from app.ml.seasonality import ATTENDANCE_BOOST


@pytest.fixture(scope="module")
def engine():
    from app.ml.engine import PricingEngine

    return PricingEngine.build()


@pytest.fixture(scope="module")
def screens() -> pd.DataFrame:
    from app.ml import loaders

    return loaders.load_screens()


@pytest.fixture(scope="module")
def bookings() -> pd.DataFrame:
    from app.ml import loaders

    return loaders.load_bookings()


# --- M1 occupancy -------------------------------------------------------------


def test_capacity_ceiling_is_never_exceeded(engine, bookings):
    """No screen/time-block/day may show more than SLOT_CAPACITY committed slots."""
    rng = np.random.default_rng(42)
    groups = list(engine.occ_engine._index.keys())
    sample = [groups[i] for i in rng.choice(len(groups), size=300, replace=False)]

    max_seen = 0.0
    for sid, tb in sample:
        block = engine.occ_engine._active_bookings(sid, tb)
        if len(block) == 0:
            continue
        daily = engine.occ_engine.get_daily_occupancy(sid, tb, block[:, 0].min(), block[:, 1].max())
        max_seen = max(max_seen, float(daily["occupied_slots"].max()))

    assert max_seen <= SLOT_CAPACITY, f"observed {max_seen} slots committed"


def test_full_exclusivity_booking_blocks_the_screen(engine, bookings):
    """A 6-slot booking consumes the whole rotation for its window."""
    full = bookings[bookings["slots_booked_per_day"] == SLOT_CAPACITY].iloc[0]
    result = engine.occ_engine.check_feasibility(
        full["screen_id"],
        full["time_block_id"],
        full["start_date"],
        full["end_date"],
        slots_needed=1,
    )
    assert result.feasible is False
    assert result.min_available_slots == 0


def test_screen_with_no_bookings_is_fully_available(engine):
    result = engine.occ_engine.check_feasibility(
        "NONEXISTENT-SCR-9999", 3, "2026-01-01", "2026-01-15", slots_needed=3
    )
    assert result.feasible is True
    assert result.min_available_slots == SLOT_CAPACITY


def test_availability_is_the_tightest_day_not_the_average(engine, bookings):
    """The reported figure must be the window minimum. A screen whose occupancy varies
    across the window would otherwise be sold slots it does not have on its busiest day."""
    rng = np.random.default_rng(7)
    groups = list(engine.occ_engine._index.keys())
    checked = 0
    for i in rng.choice(len(groups), size=200, replace=False):
        sid, tb = groups[i]
        result = engine.occ_engine.check_feasibility(sid, tb, "2026-09-01", "2026-09-30")
        daily = result.daily_series
        assert result.min_available_slots == int((SLOT_CAPACITY - daily["occupied_slots"]).min())
        if daily["occupied_slots"].nunique() > 1:
            checked += 1
    assert checked > 0, "no varying-occupancy window in the sample; test proved nothing"


def test_fast_index_matches_a_plain_groupby(bookings):
    """The port replaced a groupby-dict with a sorted-slice build for speed. Guard the
    equivalence, since every availability number depends on it."""
    subset = bookings.head(20_000).copy()
    subset["start_date"] = pd.to_datetime(subset["start_date"])
    subset["end_date"] = pd.to_datetime(subset["end_date"])

    naive = {
        key: group[["start_date", "end_date", "slots_booked_per_day"]].to_numpy()
        for key, group in subset.groupby(["screen_id", "time_block_id"])
    }
    fast = OccupancyEngine(subset)._index

    assert set(naive) == set(fast)
    for key, expected in naive.items():
        got = fast[key]
        assert expected.shape == got.shape
        # Row order within a group is irrelevant -- the consumer sums a masked subset.
        assert sorted(map(tuple, expected)) == sorted(map(tuple, got))


# --- M2 price band ------------------------------------------------------------


def test_band_is_ordered_floor_target_cap(engine, screens):
    for sid in screens["screen_id"].sample(150, random_state=1):
        band = engine.band_engine.get_price_band(sid, "evening", "finance")
        assert band.floor_price <= band.target_price <= band.cap_price, sid


def test_large_screens_price_above_small_ones(engine, screens):
    stations = screens[screens["screen_type"] == "metro_station"]
    large = stations[stations["screen_size"] == "L"].iloc[0]
    small = stations[(stations["screen_size"] == "S") & (stations["city_id"] == large["city_id"])]
    small = small.iloc[0] if len(small) else stations[stations["screen_size"] == "S"].iloc[0]

    band_l = engine.band_engine.get_price_band(large["screen_id"], "morning", "retail")
    band_s = engine.band_engine.get_price_band(small["screen_id"], "morning", "retail")
    assert band_l.target_price > band_s.target_price


def test_fallback_ladder_reports_the_level_it_used(engine, screens):
    """Every band must name the rung it resolved at, and it must be a rung that exists.
    Asserting against the closed set is the point — a typo'd or unreported level would
    otherwise make a quote untraceable to its sample."""
    levels = {
        "zone_with_daypart": 0,
        "zone_no_daypart": 0,
        "full_with_daypart": 0,
        "city_no_daypart": 0,
        "attrs_only": 0,
    }
    sample = screens["screen_id"].sample(150, random_state=1).tolist()
    for sid in sample:
        band = engine.band_engine.get_price_band(sid, "night", "healthcare")
        levels[band.segment_level_used] += 1
    assert sum(levels.values()) == len(sample)


def test_industry_adjustment_stays_inside_its_clamp(engine, screens):
    lo, hi = INDUSTRY_ADJ_CLAMP
    for sid in screens["screen_id"].sample(60, random_state=1):
        for industry in ("finance", "hospitality", "healthcare", "retail", "cpg"):
            adj = engine.band_engine.get_price_band(sid, "morning", industry).industry_adjustment
            assert lo <= adj <= hi, f"{sid}/{industry} -> {adj}"


def test_every_screen_attribute_combination_has_a_band(engine, screens):
    """Level C is the last resort and has no further fallback. Assert the combinations in
    `screens` are all covered by `bookings`, which is what makes that safe."""
    combos = (
        screens.assign(position=screens["position"].fillna("not_applicable"))
        .groupby(["screen_size", "screen_type", "position"])
        .size()
        .index
    )
    missing = [c for c in combos if c not in engine.band_engine._level_c.index]
    assert not missing, f"attribute combinations with no historical pricing: {missing}"


def test_band_uses_the_screens_own_city(engine, screens):
    """Regression on a port fix: the upstream signature took the campaign's city, so a
    multi-city campaign priced out-of-city screens off the wrong band."""
    dat = screens[screens["city_id"] == "DAT"]["screen_id"].iloc[0]
    implicit = engine.band_engine.get_price_band(dat, "morning", "retail")
    explicit = engine.band_engine.get_price_band(dat, "morning", "retail", city_id="DAT")
    wrong_city = engine.band_engine.get_price_band(dat, "morning", "retail", city_id="LH")

    assert implicit == explicit
    assert (implicit.target_price, implicit.segment_n) != (
        wrong_city.target_price,
        wrong_city.segment_n,
    )


# --- M3 booking probability ---------------------------------------------------


def test_price_coefficient_is_negative(engine):
    """THE critical check. A positive coefficient means premium screens' higher prices and
    higher booking rates are not disentangled, and the model must not price anything."""
    report = engine.training_report
    assert report.price_coefficient < 0, report.price_coefficient
    assert report.price_coefficient_sign_ok


def test_model_is_calibrated_against_the_true_base_rate(engine):
    report = engine.training_report
    assert report.calibration_ok
    assert abs(report.mean_predicted_prob - report.true_base_rate) < 0.01


def test_negative_class_is_only_price_driven_losses(engine):
    """Non-price loss reasons (ghosted, competitor, timing) must not become negatives --
    they say nothing about price elasticity."""
    from app.ml import loaders

    lost = loaders.load_lost_leads()
    expected = lost[
        lost["loss_reason"].isin(PRICE_DRIVEN_LOSS_REASONS)
        & lost["quoted_price_per_slot_per_day"].notnull()
    ]
    # The model additionally drops rows whose anchor screen has no attributes.
    assert engine.training_report.n_lost <= len(expected)
    assert engine.training_report.n_lost > 0


def test_probability_curve_decreases_with_price(engine, screens):
    station = screens[screens["screen_type"] == "metro_station"].iloc[0]
    curve = engine.prob_model.predict_proba_curve(
        np.linspace(20, 250, 30),
        station["screen_size"],
        station["screen_type"],
        station["position"] if pd.notnull(station["position"]) else "not_applicable",
        station["city_id"],
        "retail",
    )
    assert (np.diff(curve) <= 1e-9).mean() > 0.9


def test_probabilities_stay_in_bounds_at_extreme_prices(engine, screens):
    station = screens[screens["screen_type"] == "metro_station"].iloc[0]
    curve = engine.prob_model.predict_proba_curve(
        [1, 5, 500, 2000, 10000],
        station["screen_size"],
        station["screen_type"],
        "platform",
        station["city_id"],
        "retail",
    )
    assert all(0.0 <= p <= 1.0 for p in curve)


# --- M4 / M6 / engine surface -------------------------------------------------


def test_dayparts_are_derived_from_dim_slot(engine):
    """Port fix: upstream accepted `daypart` and `time_block_id` independently. In
    `bookings` they are a strict 1:1 function, so the engine derives one from the other."""
    observed = query_df(
        "SELECT DISTINCT time_block_id, daypart FROM bookings ORDER BY time_block_id"
    )
    assert len(observed) == observed["time_block_id"].nunique(), "mapping is not 1:1"
    for row in observed.itertuples():
        assert engine.daypart_for(row.time_block_id) == row.daypart


def test_unknown_screen_returns_the_full_schema(engine):
    """Consumers rely on every key being present on every row."""
    known = engine.price_candidates(_campaign(), _some_screens(engine, 1))
    rows = engine.price_candidates(_campaign(), ["NONEXISTENT-SCR-0000"])

    assert set(rows[0]) == set(known[0])
    assert rows[0]["feasible"] is False
    assert "not found" in rows[0]["assumptions"][0]
    assert rows[0]["recommended_price"] is None


def test_infeasible_rows_are_retained_with_diagnostics(engine):
    rows = engine.price_candidates(_campaign(slots_needed=6), _some_screens(engine, 120))
    infeasible = [r for r in rows if not r["feasible"] and r["recommended_price"] is None]
    assert infeasible, "no sold-out screen in the sample; test proved nothing"
    for row in infeasible:
        # Excluded, but the caller can still see why and what would have fit.
        assert row["min_free_slots"] is not None
        assert row["occupancy_rate"] is not None
        assert any("infeasible" in a for a in row["assumptions"])


def test_price_by_slot_count_is_flat_and_bounded_by_availability(engine):
    rows = engine.price_candidates(_campaign(), _some_screens(engine, 60))
    feasible = [r for r in rows if r["feasible"]]
    assert feasible
    for row in feasible:
        curve = row["price_by_slot_count"]
        for n in range(1, SLOT_CAPACITY + 1):
            if n <= row["max_available_slots"]:
                assert curve[n] == row["recommended_price"]
            else:
                assert curve[n] is None


def test_recommended_price_sits_inside_the_band(engine):
    rows = engine.price_candidates(_campaign(), _some_screens(engine, 60))
    for row in (r for r in rows if r["feasible"]):
        assert row["floor_price"] <= row["recommended_price"] <= row["cap_price"]
        # Occupancy is what positions the price inside the band.
        if row["cap_price"] > row["floor_price"]:
            position = (row["recommended_price"] - row["floor_price"]) / (
                row["cap_price"] - row["floor_price"]
            )
            assert abs(position - row["occupancy_rate"]) < 0.01


def test_time_block_is_echoed_on_every_row(engine):
    ids = _some_screens(engine, 5) + ["NONEXISTENT-SCR-1"]
    rows = engine.price_candidates(_campaign(), ids)
    assert all(r["time_block_id"] == 2 for r in rows)


def test_expected_revenue_is_price_times_probability(engine):
    for row in engine.price_candidates(_campaign(), _some_screens(engine, 40)):
        if row["feasible"]:
            expected = row["recommended_price"] * row["booking_probability"]
            assert abs(row["expected_revenue"] - expected) < 0.02


def test_mobile_screens_get_no_event_boost(engine, screens):
    """Mobile screens have no location_id, so the event join cannot be evaluated. That
    must read as not_applicable, not as a silent 1.0 meaning 'no event nearby'."""
    mobile = screens[screens["location_id"].isna()]["screen_id"].iloc[0]
    adjustment = engine.seasonality_adjuster.get_adjustment(mobile, "2026-09-01", "2026-09-15")
    assert adjustment.event_match_type == "not_applicable"
    assert adjustment.event_boost == 1.0


def test_event_boost_never_exceeds_the_largest_tier(engine, screens):
    ceiling = max(ATTENDANCE_BOOST.values())
    for sid in screens["screen_id"].sample(80, random_state=3):
        adjustment = engine.seasonality_adjuster.get_adjustment(sid, "2026-09-01", "2026-09-15")
        assert 1.0 <= adjustment.event_boost <= ceiling
        assert adjustment.event_match_type in {
            "location_match",
            "zone_match",
            "none",
            "not_applicable",
        }


def test_reach_is_not_claimed_by_this_engine(engine):
    """The impressions figure is a pricing-internal diagnostic. If it ever starts being
    presented as campaign reach, this is the test that should fail first."""
    for row in engine.price_candidates(_campaign(), _some_screens(engine, 20)):
        assert row["reach_owner"] == "audience_engine"


# --- helpers ------------------------------------------------------------------


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
