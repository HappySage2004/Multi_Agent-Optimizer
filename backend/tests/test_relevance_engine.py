"""Invariants the audience relevance engine must hold.

No LLM and no API key. These assert the properties that were actually broken or dangerous
during integration, rather than restating the implementation:

  * contract bounds hold on real data (family_score reached 1.140 before its weights were
    made to sum to 1.0)
  * geography is a HARD filter, not a soft penalty — the validator fails packages otherwise
  * pool_key is the SITE, not the location_id, and deduplication materially changes the
    audience number
  * a route's riders are distributed across its stops, never multiplied by them
  * the exposure model is applied exactly once, and both sides of the reach min() are in
    viewed units
  * an assumption (block 1, the mix floor) never leaks into a measured figure
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.data.db import query_df
from app.models.campaign import AUDIENCE_TERMS, INDUSTRY_VERTICALS, CampaignSpec
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


def _sizes(engine, candidates) -> list[str]:
    lookup = engine.profile.set_index("screen_id")["screen_size"]
    return [lookup.get(c.screen_id) for c in candidates]


def _engine_spec(**overrides) -> CampaignSpec:
    """A CampaignSpec built directly, for engine-level tests. Creates no run."""
    payload = _bare_spec(
        industry_vertical="technology",
        audience_terms=["young_professionals", "commuters"],
    )
    payload.update(overrides)
    return CampaignSpec.model_validate(payload)


def _engine_eligible() -> set[str]:
    from app.data.reference import eligible_screen_ids

    return eligible_screen_ids(["LH"], ["LH-ZONE-001"], [])


def _bare_spec(**overrides) -> dict:
    """A CampaignSpec payload, for validator tests that must not create a run."""
    payload = {
        "campaign_objective": "reach young commuters",
        "optimization_goal": "reach",
        "start_date": START,
        "duration_days": 30,
        "budget": 50_000.0,
        "city_ids": ["LH"],
        "zone_ids": ["LH-ZONE-001"],
    }
    payload.update(overrides)
    return payload


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
    # 878 physical SITES + 94 corridors. Was 1,004 when the key was the raw location_id,
    # which split single stations across platform rows and let reach count a crowd twice.
    assert engine.profile["pool_key"].nunique() == 972


def test_block_one_is_empty_and_that_is_a_known_gap(engine):
    """Documented limitation, asserted so it cannot change without a decision.

    No scheduled service starts between 00:00 and 04:00, and the model has no ambient
    term, so block 1 is zero for every screen — despite block 1 having real bookings.
    """
    for day_type in ALL_DAY_TYPES:
        col = relevance_tools.block_column(1, day_type)
        assert engine.profile[col].sum() == 0.0


def test_audience_scores_stay_inside_the_contract_bounds(engine):
    """family_score peaked at 1.140 before its weights were made to sum to 1.0."""
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


# --- Fix 1: a route's riders are shared between its stops, never multiplied ----


def test_stop_shares_sum_to_one_per_route():
    """The correction itself. Each stop takes a SHARE of its route's riders.

    Before this, every stop on a route was credited with the route's WHOLE ridership, so a
    rider on one line was counted once per stop they passed — summing the stops along a
    corridor came to 20.4x (median) the corridor's own measured ridership.
    """
    totals = query_df(
        "SELECT route_id, sum(stop_share) AS total FROM v_route_stop_weight GROUP BY 1"
    )
    assert len(totals) == 188
    assert totals["total"].min() == pytest.approx(1.0, abs=1e-9)
    assert totals["total"].max() == pytest.approx(1.0, abs=1e-9)


def test_stations_on_a_corridor_sum_to_that_corridors_ridership_exactly():
    """The acceptance test for Fix 1: 1.00x, not ~16x.

    Restricted to the routes belonging to the corridor, and measured against the same
    source on both sides. Across sources (scheduled estimate vs observed actuals) the ratio
    is ~0.97x, which is a property of the data rather than of the accounting.
    """
    ratios = query_df(
        """
        WITH corridor AS (
            SELECT rs.corridor_id, rb.time_block_id, rb.day_type,
                   sum(rb.avg_daily_ridership) AS riders
            FROM (SELECT DISTINCT route_id, corridor_id FROM route_stops) rs
            JOIN v_route_block_demand rb ON rb.route_id = rs.route_id
            GROUP BY 1, 2, 3
        ),
        stops AS (
            SELECT rs.corridor_id, rb.time_block_id, rb.day_type,
                   sum(rb.avg_daily_ridership * w.stop_share) AS riders
            FROM (SELECT DISTINCT route_id, corridor_id FROM route_stops) rs
            JOIN v_route_stop_weight w   ON w.route_id = rs.route_id
            JOIN v_route_block_demand rb ON rb.route_id = rs.route_id
            GROUP BY 1, 2, 3
        )
        SELECT s.riders / c.riders AS ratio
        FROM stops s JOIN corridor c USING (corridor_id, time_block_id, day_type)
        """
    )
    assert len(ratios) > 900
    assert ratios["ratio"].min() == pytest.approx(1.0, abs=1e-6)
    assert ratios["ratio"].max() == pytest.approx(1.0, abs=1e-6)


def test_mobile_volume_is_untouched_by_the_stop_share_correction(engine):
    """Fix 1 applies to stop-mounted screens only. The vehicle path is a different model,
    and moving both at once would have made the fixed:mobile gap unmeasurable."""
    mobile = engine.profile[engine.profile["inventory_class"] == "mobile"]
    assert len(mobile) == 2_615
    # These medians were measured before the correction and must not have moved.
    by_type = mobile.groupby("screen_type")["total_impressions"].median().round(1)
    assert by_type["bus"] == pytest.approx(597.6, abs=0.1)
    assert by_type["metro_rail_coach"] == pytest.approx(6147.8, abs=0.1)


# --- Fix 2: the pool is a site, not a location row ----------------------------


def test_locations_sharing_a_site_share_a_pool_key():
    """A site is (city, name, serving corridors). Screens on two platforms of one station
    see the same people, so they must land in one pool or reach counts that crowd twice."""
    violations = query_df(
        """
        WITH corridor_set AS (
            SELECT location_id,
                   array_to_string(list_sort(list_distinct(list(corridor_id))), ',') AS corridors
            FROM route_stops GROUP BY 1
        )
        SELECT count(*) AS n FROM (
            SELECT l.city_id, l.name, coalesce(c.corridors, '') AS corridors
            FROM v_screen_profile p
            JOIN locations l ON l.location_id = p.location_id
            LEFT JOIN corridor_set c ON c.location_id = p.location_id
            WHERE p.inventory_class = 'fixed'
            GROUP BY 1, 2, 3
            HAVING count(DISTINCT p.pool_key) > 1
        )
        """
    )
    assert int(violations["n"][0]) == 0


def test_pool_key_does_not_over_merge_on_name_alone(engine):
    """Grouping on (city, name) alone collapses 910 locations to 626 — a 31% over-merge,
    because station names are a low-cardinality template that unrelated stops share. The
    corridor set is what keeps two genuinely different stops apart."""
    fixed = engine.profile[engine.profile["inventory_class"] == "fixed"]
    assert fixed["pool_key"].nunique() == 878
    assert fixed["location_id"].nunique() == 910
    by_name = query_df(
        """
        SELECT count(*) AS n FROM (
            SELECT DISTINCT l.city_id, l.name FROM locations l
            WHERE l.location_id IN (
                SELECT DISTINCT location_id FROM v_screen_profile WHERE inventory_class='fixed'
            )
        )
        """
    )
    assert int(by_name["n"][0]) == 626


# --- Fix 3: mobile screens are not penalized for having no POI ----------------


def test_no_mobile_screen_takes_the_poi_mismatch_penalty():
    """`v_screen_poi` joins on location_id, which a vehicle does not have. Scoring mobile
    screens against a POI set handed all 2,615 the 0.2 mismatch penalty for an
    architectural reason, not a measured one."""
    # Restricted to vehicle-mounted inventory, because a city-wide cut is filled by
    # higher-scoring fixed screens long before it reaches a bus.
    run_id = _spec(
        zone_ids=[],
        city_ids=["LH"],
        hard_constraints={"allowed_screen_types": list(relevance_tools.MOBILE_SCREEN_TYPES)},
    )["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 300})
    mobile = [c for c in _candidates(out) if c.screen_type in relevance_tools.MOBILE_SCREEN_TYPES]
    assert len(mobile) == 300
    assert all(c.contextual_score == 0.5 for c in mobile)
    assert not any(c.contextual_score == 0.2 for c in mobile)
    # And it is disclosed, not silent.
    assert all(any("not applicable" in d for d in c.defaults_applied) for c in mobile)


# --- Fix 4: graded occupation affinity ----------------------------------------


def test_every_dominant_occupation_value_is_mapped(engine):
    """Five values across 30 zones. An unmapped one scores 0 and warns — that path must
    stay unreachable on this data, or a whole zone silently loses its audience signal."""
    real = set(
        query_df(
            "SELECT DISTINCT dominant_occupation AS o FROM zone_demographics WHERE o IS NOT NULL"
        )["o"]
    )
    assert real == {"mixed", "white_collar", "blue_collar", "retail_service", "student"}
    for affinity in (
        relevance_tools.OCCUPATION_PROFESSIONAL_AFFINITY,
        relevance_tools.OCCUPATION_STUDENT_AFFINITY,
        relevance_tools.OCCUPATION_FAMILY_AFFINITY,
    ):
        assert real == set(affinity)


def test_family_score_needs_no_renormalization_constant(engine):
    """Its weights sum to 1.0 over bounded inputs, so it cannot exceed 1.0 by construction.
    It used to sum a 1.0- and a 0.25-weighted term and peak at 1.140."""
    assert not hasattr(relevance_tools, "FAMILY_SCORE_MAX")
    assert engine.profile["family_score"].max() <= 1.0


def test_high_income_and_young_professionals_are_their_own_scores(engine):
    """Both used to resolve to a column measuring something else: `high_income` to
    professional_score (40% occupation) and `young_professionals` to the MEAN of
    professional_score and student_score, which is a different audience."""
    assert relevance_tools.AUDIENCE_TERM_TO_SCORE_COLUMN["high_income"] == ["high_income_score"]
    assert relevance_tools.AUDIENCE_TERM_TO_SCORE_COLUMN["young_professionals"] == [
        "young_professionals_score"
    ]
    # high_income is income alone; professional_score is not.
    assert (engine.profile["high_income_score"] == engine.profile["income_score"]).all()
    assert not (engine.profile["professional_score"] == engine.profile["income_score"]).all()


def test_mobile_screens_score_zero_on_every_demographic_component(engine):
    """The documented structural floor, asserted so it cannot be mistaken for a signal."""
    mobile = engine.profile[engine.profile["inventory_class"] == "mobile"]
    for col in (
        "income_score",
        "professional_score",
        "young_professionals_score",
        "student_score",
        "high_income_score",
    ):
        assert (mobile[col] == 0.0).all(), col
    # commuter_score is schedule-derived and DOES carry signal for them.
    assert mobile["commuter_score"].nunique() > 1


# --- Fix 6: tie-breaking ------------------------------------------------------


def test_exact_relevance_ties_are_broken_by_size_then_footfall_then_id(engine):
    """Screens at one stop genuinely tie — same zone, same POIs, same traffic — so the
    order among them was decided by screen_id alone. Size breaks it first.

    Grouped on the engine's exact float, not the contract's 4dp rounding: two screens whose
    scores differ in the 5th decimal are not tied, and rounding them together would assert
    an ordering the sort never promised.
    """
    ranked = engine.score(_engine_spec(), _engine_eligible(), 10**6).candidates
    ranks = relevance_tools.SCREEN_SIZE_RANK

    checked = 0
    for _, tied in ranked.groupby("relevance_score", sort=False):
        if len(tied) < 2 or tied["screen_size"].nunique() < 2:
            continue
        seq = [ranks.get(v, 0.0) for v in tied["screen_size"]]
        assert seq == sorted(seq, reverse=True), tied["screen_id"].tolist()
        checked += 1
    assert checked > 50, f"only {checked} mixed-size ties; this test asserted little"


def test_the_candidate_pool_is_reproducible(engine):
    """screen_id stays the final sort key, which is what makes the artifact replayable."""
    run_id = _spec()["run_id"]
    first = _candidates(
        relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 250})
    )
    again = _candidates(
        relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 250})
    )
    assert [c.screen_id for c in again] == [c.screen_id for c in first]


# --- Fix 7: ambient footfall is quarantined ----------------------------------


def test_ambient_footfall_is_carried_but_never_added_into_volume(engine):
    """It correlates only weakly with transit ridership (~0.12-0.26) and can disagree ~20x
    at one location, so it is reported as a separate signal and used only as a tie-break.
    Blending it into impressions would undo Fix 1's correction."""
    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 100})
    candidates = _candidates(out)
    assert any(c.nearby_ambient_footfall > 0 for c in candidates)
    for c in candidates:
        # The day-type totals reconcile to the 12 blocks and to nothing else. Tolerance is
        # the 1dp rounding on 12 contract fields, not slack in the model.
        block_total = sum(c.impressions_by_block.values())
        assert c.impressions_weekday + c.impressions_weekend == pytest.approx(block_total, abs=1.0)
        # A footfall figure large enough to notice is not hiding inside the volume.
        if c.nearby_ambient_footfall > 100:
            assert abs(c.impressions_weekday + c.impressions_weekend - block_total) < 1.0


# --- Fix 8: block 1 is an estimate, kept out of every measured figure --------


def test_block_one_estimate_is_separate_from_every_measured_total(engine):
    """8,544 real bookings sit in block 1, but no scheduled service runs then, so measured
    volume is 0. The 8%-of-block-6 estimate is published separately and must not reach a
    total, an off-peak figure or commuter_score — nothing the validator checks may move
    with an assumed constant."""
    p = engine.profile
    for day_type in ALL_DAY_TYPES:
        assert (p[relevance_tools.block_column(1, day_type)] == 0.0).all()
        assert (p[f"block_1_estimated_{day_type}"] > 0).any()

    # peak + offpeak == total EXACTLY, on measured columns only.
    assert ((p["peak_impressions"] + p["offpeak_impressions"]) == p["total_impressions"]).all()

    # The estimate is real and non-trivial, so leaking it would be visible.
    estimate = p["block_1_estimated_weekday"] + p["block_1_estimated_weekend"]
    assert estimate.sum() > 0
    measured_blocks = sum(
        p[relevance_tools.block_column(b, dt)] for b in ALL_TIME_BLOCKS for dt in ALL_DAY_TYPES
    )
    assert (p["total_impressions"] - measured_blocks).abs().max() < 1e-6


def test_block_one_estimate_reaches_the_contract_as_an_estimate():
    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 40})
    for c in _candidates(out):
        assert set(c.impressions_block_1_estimated) == set(ALL_DAY_TYPES)
        assert c.impressions_by_block["1_weekday"] == 0.0
        assert c.impressions_by_block["1_weekend"] == 0.0


# --- Fix 5 item 4: the corridor pool invariant -------------------------------


def test_a_corridor_pool_is_never_smaller_than_a_station_on_it():
    """`v_corridor_block_demand` divides by vehicles and `optimize/contract.py` multiplies
    back by pool_partition_count — a round trip across two modules that nothing checked.
    A station's riders on a corridor are a subset of that corridor's riders."""
    report = relevance_tools.corridor_pool_sanity()
    assert report["violations"] == 0, report["worst"]


# --- Fix 10: a mixed brief must be servable ----------------------------------


def test_allowed_screen_types_alone_cannot_produce_a_mix():
    """Pinning the bug, so nobody 'fixes' it by widening allowed_screen_types again.

    It is a FILTER, not a mix: permitting metro_station and bus lets 806 eligible screens
    through and a single global relevance cut still returns 250 metro_station and 0 bus,
    because bus's best score sits below metro's pool minimum.
    """
    run_id = _spec(hard_constraints={"allowed_screen_types": ["metro_station", "bus"]})["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 250})
    composition = out["pool_composition_by_screen_type"]
    assert composition == {"metro_station": 250}
    assert out["eligible_by_screen_type"]["bus"] > 0


def test_screen_type_mix_stratifies_the_pool_so_both_types_survive(engine):
    """Scored through the engine rather than through intake: `create_campaign_spec` does not
    yet expose `screen_type_mix`, so no agent can set it. That parameter and its docstring
    are a handoff to the Master session — the engine behaviour is what is pinned here."""
    result = engine.score(
        _engine_spec(
            hard_constraints={"allowed_screen_types": ["metro_station", "bus"]},
            screen_type_mix=["metro_station", "bus"],
        ),
        _engine_eligible(),
        250,
    )
    composition = result.pool_by_screen_type
    assert set(composition) == {"metro_station", "bus"}
    assert composition["bus"] > 0 and composition["metro_station"] > 0
    assert sum(composition.values()) == 250
    assert not result.mix_unfilled
    # Still ranked best-first across the strata.
    scores = result.candidates["relevance_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_a_scarce_requested_type_gets_its_whole_supply_and_no_capacity_is_wasted(engine):
    """A type with 75 screens gets 75, not a fixed share of 125, and the rest it cannot
    fill go to the type that can — so a mix request never shrinks the pool."""
    result = engine.score(
        _engine_spec(screen_type_mix=["metro_station", "bus_stop"]), _engine_eligible(), 250
    )
    assert result.pool_by_screen_type["bus_stop"] == result.eligible_by_screen_type["bus_stop"]
    assert sum(result.pool_by_screen_type.values()) == 250


def test_an_unavailable_requested_type_is_reported_not_silently_dropped(engine):
    """The Master cannot tell a rep a type was unservable if it was never told."""
    result = engine.score(
        _engine_spec(
            hard_constraints={"allowed_screen_types": ["metro_station"]},
            screen_type_mix=["metro_station", "bus"],
        ),
        _engine_eligible(),
        100,
    )
    assert "bus" in result.mix_unfilled
    assert result.pool_by_screen_type == {"metro_station": 100}


def test_pool_composition_is_reported_even_when_no_mix_was_requested():
    """A pool that is 100% one screen type is the most consequential fact about it."""
    run_id = _spec()["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 250})
    assert out["pool_composition_by_screen_type"]
    assert out["screen_type_mix_requested"] == []
    assert sum(out["pool_composition_by_screen_type"].values()) == 250


# --- Fix 11: the industry vocabulary --------------------------------------


def test_industry_vertical_vocabulary_matches_the_poi_map():
    """Two sub-scores are keyed on this string — context_fit (0.15) and
    historical_performance (0.10). A value the map does not hold collapses BOTH to 0.5 for
    every screen, pinning 25% of relevance_score to a constant while reporting success."""
    assert set(INDUSTRY_VERTICALS) == set(relevance_tools.INDUSTRY_TO_POI_CONTEXT)
    real = set(query_df("SELECT DISTINCT industry_vertical AS v FROM bookings")["v"])
    assert real == set(INDUSTRY_VERTICALS)


def test_off_vocabulary_industry_verticals_are_rejected():
    """These are the actual values the Master wrote to runs before the field was closed.

    Every one of them silently neutralized context_fit AND historical_performance — 25% of
    relevance_score — while the pipeline reported success.
    """
    for bad in ("AUTOMOTIVE / ELECTRIC VEHICLES", "Consumer Tech", "Fintech", "Beauty / Skincare"):
        with pytest.raises(ValidationError):
            CampaignSpec.model_validate(_bare_spec(industry_vertical=bad))
        # Through the intake tool it comes back as a recoverable result, not an exception.
        out = _spec(industry_vertical=bad)
        assert out.get("status") == "invalid", out
        assert "industry_vertical" in str(out)


def test_case_and_separator_variants_are_normalized_not_rejected():
    """'Real Estate' is unambiguously real_estate; failing it would reject a right answer
    on formatting."""
    for given, expected in (
        ("Real Estate", "real_estate"),
        ("TECHNOLOGY", "technology"),
        ("real-estate", "real_estate"),
    ):
        spec = CampaignSpec.model_validate(_bare_spec(industry_vertical=given))
        assert spec.industry_vertical == expected


def test_a_pool_wide_constant_subscore_is_reported_loudly():
    """The failure mode that hid the industry bug: a sub-score identical across the pool
    contributes nothing but its weight, and every path there looked like success."""
    run_id = _spec(industry_vertical=None, audience_terms=["commuters"])["run_id"]
    out = relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 100})
    reported = " ".join(out["defaults_applied"]) + " ".join(out["constant_subscores"])
    assert "context_fit" in reported
    assert "historical_performance" in reported
