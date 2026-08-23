"""Demand-value model + the zone rungs of the price band.

The properties worth pinning are the ones that make the premium defensible rather than
merely computed: the model must not be able to see a price, the gates must actually
withhold, and the premium must be bounded and disclosed on every row it touches.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.db import query_df
from app.ml.demand_value import (
    DEMAND_VALUE_SQL,
    MAX_PREMIUM,
    MIN_ABSORPTION_RANK,
    MIN_MERIT,
    MIN_PRICE_HISTORY,
    MIN_RESIDUAL,
    W_DAYTIME,
    W_INCOME,
    W_POI,
    W_VOLUME,
    DemandValueModel,
)
from app.ml.levers import PricingLevers


@pytest.fixture(scope="module")
def engine():
    from app.ml.engine import PricingEngine

    return PricingEngine.build()


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    return query_df(DEMAND_VALUE_SQL)


def _campaign(**overrides) -> dict:
    campaign = {
        "industry_vertical": "retail",
        "time_block_id": 2,
        "start_date": "2026-10-01",
        "end_date": "2026-10-30",
        "slots_needed": 1,
    }
    campaign.update(overrides)
    return campaign


# --- the merit index ----------------------------------------------------------


def test_merit_weights_sum_to_one():
    assert pytest.approx(1.0) == W_VOLUME + W_INCOME + W_DAYTIME + W_POI


def test_merit_is_bounded_and_defined_for_every_screen(scored):
    assert len(scored) == 11_163
    assert scored["merit"].between(0.0, 1.0).all()
    assert scored["merit"].notna().all()


def test_merit_is_computed_without_any_price_input(scored):
    """The property the whole model rests on.

    A merit score fitted to price would learn a systematically underpriced category as
    correct and report no mispricing at all. This asserts the inputs are structurally
    price-free: merit must be reproducible from the four feature ranks alone.
    """
    recomputed = (
        W_VOLUME * scored["pr_volume"]
        + W_INCOME * scored["pr_income"]
        + W_DAYTIME * scored["pr_daytime"]
        + W_POI * scored["pr_poi"]
    )
    pd.testing.assert_series_equal(scored["merit"], recomputed, check_names=False, rtol=1e-9)


def test_price_rank_is_ranked_only_among_screens_that_have_a_price(scored):
    """Regression on a real bug, and the sharpest edge in this model.

    `pr_price` must be a percentile among screens that HAVE a price index. Ranking the
    NULL-price screens in the same window partition squeezed 181 priced bus_stop/ACS
    screens into ranks 0.000-0.257 instead of 0.000-1.000 — every price rank depressed,
    every residual inflated, and screens transacting ABOVE their own comparables flagged as
    underpriced (811 premiums against the correct 385).

    The invariant: within any group that has enough priced screens, the ranks must span
    close to the full 0-1 range regardless of how many unpriced screens sit alongside them.
    """
    priced = scored[scored["pr_price"].notna()]
    groups = priced.groupby(["screen_type", "city_id"])

    checked = 0
    for (stype, city), group in groups:
        if len(group) < 50:
            continue
        checked += 1
        unpriced = len(
            scored[
                (scored["screen_type"] == stype)
                & (scored["city_id"] == city)
                & scored["pr_price"].isna()
            ]
        )
        assert group["pr_price"].max() > 0.98, (
            f"{stype}/{city}: {len(group)} priced screens rank only up to "
            f"{group['pr_price'].max():.3f} — {unpriced} unpriced screens are diluting "
            f"the ranking partition"
        )
        assert group["pr_price"].min() == pytest.approx(0.0)
    assert checked >= 3, f"only {checked} groups were large enough to check"


def test_a_screen_priced_above_its_comparables_is_ranked_above_the_median(scored):
    """The consequence of the bug above, stated as a property. A screen whose price index
    exceeds its peers' median cannot sit in the bottom half of the price rank."""
    priced = scored[scored["pr_price"].notna()]
    for (stype, city), group in priced.groupby(["screen_type", "city_id"]):
        if len(group) < 50:
            continue
        median_index = group["price_index"].median()
        dearer = group[group["price_index"] > median_index]
        assert (dearer["pr_price"] >= 0.49).all(), (
            f"{stype}/{city}: a screen above the median price index ranked below 0.49"
        )


def test_merit_agrees_with_the_market_without_reproducing_it(scored):
    """Merit must correlate positively with realized price — zero would mean it is noise —
    but well short of 1.0, or there would be no headroom for it to find."""
    fixed = scored[
        (scored["inventory_class"] == "fixed")
        & (scored["price_n"] >= MIN_PRICE_HISTORY)
        & scored["pr_price"].notna()
    ]
    corr = fixed["merit"].corr(fixed["pr_price"])
    assert 0.15 < corr < 0.75, f"merit vs realized price rank correlation was {corr:.3f}"


# --- the gates ----------------------------------------------------------------


def test_mobile_inventory_never_receives_a_premium(engine):
    """Mobile audience correlates NEGATIVELY with what actually sells (-0.25 bus, -0.21
    metro_rail_coach), and mobile screens have no zone demographics at all. Pricing off
    that would be pricing off a gap in the data."""
    model = engine.demand_value
    mobile = engine._screens[engine._screens["location_id"].isna()].index[:200]
    assert len(mobile) > 0

    for sid in mobile:
        value = model.for_screen(sid)
        assert value.premium == 1.0
        assert "excluded" in value.reason


def test_a_screen_that_does_not_sell_is_withheld_however_good_it_looks(scored):
    """The gate that earns its place: it withholds screens carrying the highest modelled
    audience in the inventory, precisely because nobody is buying them."""
    model = DemandValueModel(scored)
    withheld = [
        v for v in (model.for_screen(s) for s in scored["screen_id"]) if "does not sell" in v.reason
    ]
    assert withheld, "no screen hit the absorption gate; the test proved nothing"
    for value in withheld:
        assert value.premium == 1.0
        assert value.absorption_rank < MIN_ABSORPTION_RANK
        assert value.merit >= MIN_MERIT


def test_a_cheap_but_weak_screen_is_not_called_underpriced(scored):
    """A weak screen priced weakly is the market being right. Without this gate the
    residual alone flags 234 screens sitting below the median merit for their own type
    and city."""
    model = DemandValueModel(scored)
    values = [model.for_screen(s) for s in scored["screen_id"]]
    below_median_merit = [v for v in values if v.merit < MIN_MERIT]
    assert below_median_merit
    assert all(v.premium == 1.0 for v in below_median_merit)


def test_a_screen_with_thin_history_gets_no_premium(scored):
    model = DemandValueModel(scored)
    thin = scored[scored["price_n"] < MIN_PRICE_HISTORY]["screen_id"]
    assert len(thin) > 0
    for sid in thin[:300]:
        assert model.for_screen(sid).premium == 1.0


def test_every_premium_screen_cleared_every_gate(scored):
    model = DemandValueModel(scored)
    granted = [v for v in (model.for_screen(s) for s in scored["screen_id"]) if v.has_premium]
    assert granted, "no screen received a premium at all"

    for value in granted:
        assert value.merit >= MIN_MERIT
        assert value.price_rank is not None
        assert value.residual >= MIN_RESIDUAL
        assert value.absorption_rank >= MIN_ABSORPTION_RANK
        assert 1.0 < value.premium <= 1.0 + MAX_PREMIUM + 1e-9
        assert "underpriced" in value.reason


def test_the_premium_ramps_from_the_gate_rather_than_jumping_at_it(scored):
    """A screen that only just clears MIN_RESIDUAL must get ~nothing, or the gate becomes a
    cliff where a rounding difference is worth several percent of price."""
    model = DemandValueModel(scored)
    values = [model.for_screen(s) for s in scored["screen_id"]]
    marginal = [
        v
        for v in values
        if v.residual is not None and MIN_RESIDUAL <= v.residual < MIN_RESIDUAL + 0.02
    ]
    if marginal:
        assert all(v.premium < 1.0 + MAX_PREMIUM * 0.10 for v in marginal)


def test_an_unknown_screen_is_neutral_rather_than_an_error(scored):
    value = DemandValueModel(scored).for_screen("NOT-A-SCREEN")
    assert value.premium == 1.0
    assert value.reason


# --- the premium through the engine -------------------------------------------


def test_premium_is_applied_and_disclosed_on_the_row(engine):
    rows = engine.price_candidates(_campaign(), engine._screens.index[:400].tolist())
    feasible = [r for r in rows if r["feasible"]]
    premium_rows = [r for r in feasible if (r["demand_premium"] or 1.0) > 1.0]
    assert premium_rows, "no premium in the sample; the test proved nothing"

    for row in premium_rows:
        assert row["demand_premium"] <= 1.0 + MAX_PREMIUM + 1e-9
        assert row["demand_value_index"] >= MIN_MERIT
        assert row["historical_price_index"] is not None
        assert "underpriced" in row["demand_value_reason"]
        # The price change must be visible in the row's own audit trail, not only in a
        # field a consumer has to know to look at.
        assert any("demand premium" in a for a in row["assumptions"])


def test_only_the_demand_premium_can_carry_a_quote_above_the_band_cap(engine):
    """Deliberate: an underpriced screen's own comparables are what understate it, so
    clamping the premium back inside the band would delete the correction. Nothing ELSE
    may exceed the cap, and the excess is bounded by the premium."""
    rows = engine.price_candidates(_campaign(), engine._screens.index[:400].tolist())
    for row in (r for r in rows if r["feasible"]):
        if row["recommended_price"] > row["cap_price"] + 0.01:
            premium = row["demand_premium"] or 1.0
            assert premium > 1.0, f"{row['screen_id']} exceeded cap with no demand premium"
            assert row["recommended_price"] <= row["cap_price"] * premium + 0.02


def test_disabling_the_premium_returns_the_engine_to_comparables_only(engine):
    screens = engine._screens.index[:400].tolist()
    on = {r["screen_id"]: r for r in engine.price_candidates(_campaign(), screens) if r["feasible"]}
    off = {
        r["screen_id"]: r
        for r in engine.price_candidates(
            _campaign(), screens, PricingLevers(demand_premium_weight=0.0)
        )
        if r["feasible"]
    }
    assert on and off

    raised = 0
    for sid, row in on.items():
        premium = row["demand_premium"] or 1.0
        assert off[sid]["recommended_price"] == pytest.approx(
            row["recommended_price"] / premium, abs=0.02
        )
        # With the premium off, nothing may sit above the band cap any more.
        assert off[sid]["recommended_price"] <= off[sid]["cap_price"] + 0.02
        if premium > 1.0:
            raised += 1
    assert raised, "no premium was active, so switching it off proved nothing"


def test_premium_does_not_change_what_is_purchasable(engine):
    """A pricing judgement must never move inventory."""
    screens = engine._screens.index[:300].tolist()
    on = engine.price_candidates(_campaign(), screens)
    off = engine.price_candidates(_campaign(), screens, PricingLevers(demand_premium_weight=0.0))
    assert [r["feasible"] for r in on] == [r["feasible"] for r in off]
    assert [r["max_available_slots"] for r in on] == [r["max_available_slots"] for r in off]


# --- Layer A: the zone rungs --------------------------------------------------


def test_zone_rungs_carry_most_of_the_inventory(engine):
    """Zone sits above city because the within-city zone spread is 1.87x-2.52x. If coverage
    ever collapsed, the ladder would silently degrade back to city bands."""
    import collections

    levels = collections.Counter()
    for sid in engine._screens.index[:600]:
        levels[engine.band_engine.get_price_band(sid, "morning", "retail").segment_level_used] += 1

    zone_levels = levels["zone_with_daypart"] + levels["zone_no_daypart"]
    assert zone_levels > 0.5 * sum(levels.values()), dict(levels)


def test_mobile_screens_fall_through_the_zone_rungs_without_a_branch(engine):
    """Zone is NULL for all 2,615 vehicle-mounted screens. The zone keys simply never
    match, which is why there is no inventory-class branch in the ladder."""
    mobile = engine._screens[engine._screens["location_id"].isna()].index[:80]
    assert len(mobile) > 0
    for sid in mobile:
        band = engine.band_engine.get_price_band(sid, "morning", "retail")
        assert band.zone_id is None
        assert band.segment_level_used in {"full_with_daypart", "city_no_daypart", "attrs_only"}


def test_zone_band_differs_from_the_city_blend_it_replaced(engine):
    """The point of the change: two screens with identical physical attributes in the same
    city, in different zones, must no longer be quoted from the same band."""
    fixed = engine._screens[engine._screens["location_id"].notna()].copy()
    bands = {}
    for sid in fixed.index[:800]:
        band = engine.band_engine.get_price_band(sid, "morning", "retail")
        if band.segment_level_used.startswith("zone"):
            row = fixed.loc[sid]
            key = (row["screen_size"], row["screen_type"], row["position"], row["city_id"])
            bands.setdefault(key, {})[band.zone_id] = band.target_price

    multi_zone = {k: v for k, v in bands.items() if len(v) > 1}
    assert multi_zone, "sample held no segment spanning two zones"
    spreads = [max(v.values()) / min(v.values()) for v in multi_zone.values()]
    assert max(spreads) > 1.10, f"zone made no material difference: max spread {max(spreads):.3f}"


# --- Layer A: the deal-shape split -------------------------------------------


def test_deal_shape_is_omitted_by_default_and_reproduces_the_blended_band(engine):
    """`is_bundle=None` must mean 'do not split', not 'guess'. A caller that does not know
    the deal shape gets the blended band, exactly as before the split existed."""
    for sid in engine._screens.index[:150]:
        implicit = engine.band_engine.get_price_band(sid, "morning", "retail")
        explicit = engine.band_engine.get_price_band(sid, "morning", "retail", is_bundle=None)
        assert implicit == explicit
        assert not implicit.segment_level_used.endswith(("_bundle", "_single_screen"))


def test_single_screen_comparables_price_above_bundled_ones(engine):
    """At zone grain, single-screen deals sit x1.065-x1.090 above bundled ones. A package
    quoted off single-screen comparables would be overpriced, and vice versa — which is the
    whole reason the split exists."""
    ratios = []
    for sid in engine._screens.index[:400]:
        bundle = engine.band_engine.get_price_band(sid, "morning", "retail", is_bundle=True)
        single = engine.band_engine.get_price_band(sid, "morning", "retail", is_bundle=False)
        # Only compare where BOTH resolved at a shape-split rung; a blended fallback on
        # either side is measuring the same cell twice.
        if bundle.segment_level_used.endswith("_bundle") and single.segment_level_used.endswith(
            "_single_screen"
        ):
            ratios.append(single.target_price / bundle.target_price)

    assert len(ratios) > 20, f"only {len(ratios)} cells had both shapes; sample too thin"
    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio > 1.0, f"single-screen priced at or below bundle: x{mean_ratio:.4f}"


def test_the_rung_name_discloses_which_deal_shape_was_used(engine):
    """A quoted price has to be traceable to the sample it came from, including whether the
    comparables were the same deal shape or a blended fallback."""
    seen = set()
    for sid in engine._screens.index[:400]:
        for shape in (True, False):
            level = engine.band_engine.get_price_band(
                sid, "morning", "retail", is_bundle=shape
            ).segment_level_used
            seen.add(level)
            if level.endswith("_bundle"):
                assert shape is True
            elif level.endswith("_single_screen"):
                assert shape is False

    assert any(x.endswith("_bundle") for x in seen)
    assert any(x.endswith("_single_screen") for x in seen)


def test_deal_shape_is_surrendered_before_zone(engine):
    """Ordering is the design: shape is worth 6.5-9%, zone 87-152%. When a shape-split cell
    is too thin, the ladder must fall back to the BLENDED band at the SAME location depth
    rather than descending to a coarser location."""
    fell_back = 0
    for sid in engine._screens.index[:500]:
        blended = engine.band_engine.get_price_band(sid, "night", "retail")
        shaped = engine.band_engine.get_price_band(sid, "night", "retail", is_bundle=True)
        if not shaped.segment_level_used.endswith("_bundle"):
            # Shape was dropped. The location depth must be no coarser than the blended
            # lookup reached.
            assert shaped.segment_level_used == blended.segment_level_used
            assert any("deal shape dropped" in a for a in shaped.assumptions)
            fell_back += 1
    assert fell_back, "no shape fallback in the sample; the ordering rule was not exercised"


def test_a_single_screen_campaign_prices_off_single_screen_comparables(engine):
    """The seam: `requested_num_screens == 1` is the only case the spec states a deal shape
    unambiguously before the optimizer has picked anything."""
    ids = engine._screens.index[:200].tolist()
    base = dict(_campaign())

    as_bundle = engine.price_candidates({**base, "is_bundle": True}, ids)
    as_single = engine.price_candidates({**base, "is_bundle": False}, ids)

    bundle_prices = {r["screen_id"]: r["recommended_price"] for r in as_bundle if r["feasible"]}
    single_prices = {r["screen_id"]: r["recommended_price"] for r in as_single if r["feasible"]}
    shared = [k for k in bundle_prices if k in single_prices and bundle_prices[k]]
    assert shared

    ratio = sum(single_prices[k] / bundle_prices[k] for k in shared) / len(shared)
    assert ratio > 1.0, f"single-screen quote came out at or below the bundled one: x{ratio:.4f}"


def test_band_is_still_ordered_at_every_rung(engine):
    for sid in engine._screens.index[:400]:
        band = engine.band_engine.get_price_band(sid, "evening", "finance")
        assert band.floor_price <= band.target_price <= band.cap_price, sid
