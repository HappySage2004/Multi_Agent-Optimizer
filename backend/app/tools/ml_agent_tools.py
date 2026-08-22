"""Tools for the ML / PRICING AGENT.

Thin wrapper over the pricing engine in `app/ml/`. The tool resolves the run's spec into
per-time-block campaign context, calls the engine, and maps its rows onto the
`ScreenEconomics` contract. No decision logic lives here.

============================== INTEGRATION STATUS ==============================
PRICING -- REAL, and owned here. `app/ml/` implements SOLUTION.md section 8: a market
price band from historical bookings (p25/p50/p90 with a bounded segmentation ladder), a
calibrated booking-probability model trained on bookings vs price-driven lost leads,
day-by-day slot occupancy from live bookings, and a seasonality/event adjustment. The
quoted price is occupancy-driven (`floor + occupancy_rate x (cap - floor)`); the
probability model is a reported diagnostic, not the optimizer of the price. These are
SELLER-SIDE prices.

AUDIENCE VOLUME -- REAL, but owned UPSTREAM. It is computed by the relevance engine
(`app/tools/relevance_tools.py`) and arrives on the `screen_candidates` artifact as daily
riders per time block per day type. This module only converts units: block-daily traffic
for the campaign's actual weekday/weekend mix, then through `app/optimize/exposure.py` to
get `viewed_exposures_per_slot_per_day` and `reachable_daily_audience`. Nothing is
modelled here, and the exposure constants are not duplicated here either -- there is one
implementation, and this is its only call site.

`pricing_internal_reach_proxy` remains carried for pricing diagnostics only and is still
NOT mapped onto any exposure field -- it is a hand-set heuristic whose fixed and mobile
paths are on different units. The relevance engine's ridership figures are the audience
number; the proxy never was.
===============================================================================
"""

from __future__ import annotations

from datetime import date, timedelta

from langchain_core.tools import tool

from app.logging_utils import debug, error, info
from app.models.economics import DemandForecastSummary, PricingRecommendation, ScreenEconomics
from app.models.screens import ScreenCandidate
from app.optimize.exposure import (
    is_viewability_assumed,
    reachable_daily_audience,
    viewability,
    viewed_exposures_per_slot_per_day,
)
from app.services import run_state
from app.services.artifact_store import read_models, write_records
from app.tools.relevance_tools import ALL_DAY_TYPES

ARTIFACT_KIND = "screen_economics"
CANDIDATES_KIND = "screen_candidates"

# Blocks 2, 3 and 5 are the commuter peaks and midday (04:00-08:00, 08:00-12:00,
# 16:00-20:00) per dim_slot. Only reached when neither the spec nor the candidate artifact
# names the campaign's target blocks.
DEFAULT_TIME_BLOCKS = ("2", "3", "5")

AUDIENCE_NOTICE = (
    "Audience volume comes from the relevance engine's transit ridership model, not from "
    "this stage. Three figures, three units: daily_unique_audience is people PASSING, "
    "reachable_daily_audience is the share who look (the reach ceiling), and "
    "viewed_exposures_per_slot_per_day is what one slot earns on one day. Reach is "
    "deduplicated by pool_key downstream and is never the sum of exposures. Volume is "
    "schedule-derived with no ambient term, so time block 1 (00:00-04:00) is zero "
    "everywhere — 'not modelled', never 'nobody there'."
)


@tool
def estimate_screen_economics(
    run_id: str, time_blocks: list[str] | None = None, slots_needed: int = 1
) -> dict:
    """Price each candidate screen for each requested time block, with availability.

    Consumes the `screen_candidates` artifact and produces the `screen_economics`
    artifact the optimizer needs. Returns an artifact reference plus aggregates.

    Args:
        run_id: Handle for the campaign run.
        time_blocks: dim_slot time_block_ids to price. Defaults to the campaign's
            preferred blocks, then to the blocks the relevance engine identified for this
            audience, then to the commuter peaks and midday ("2", "3", "5").
        slots_needed: Slots per day the campaign needs on a screen for it to count as
            available. A screen with fewer free slots on any day of the flight is
            returned as infeasible rather than dropped.
    """
    if (blocked := run_state.missing_prerequisite(run_id, CANDIDATES_KIND)) is not None:
        error(f"STAGE 4 blocked: {blocked['detail']}")
        return blocked

    spec = run_state.get_spec(run_id)
    ref_candidates = run_state.require_artifact(run_id, CANDIDATES_KIND)
    candidates = read_models(ref_candidates, ScreenCandidate)
    if not candidates:
        error(f"STAGE 4 no candidates on run_id={run_id}")
        return {
            "status": "no_candidates",
            "run_id": run_id,
            "detail": "The screen_candidates artifact is empty; nothing to price.",
        }

    from app.ml.engine import get_pricing_engine

    engine = get_pricing_engine()

    # The relevance engine publishes the blocks this campaign's audience is actually active
    # in. Reading it here keeps one authoritative copy of that mapping instead of
    # reimplementing it and drifting apart.
    engine_blocks = ref_candidates.summary.get("preferred_time_blocks") or []
    chosen = time_blocks or spec.preferred_time_blocks or engine_blocks or DEFAULT_TIME_BLOCKS
    block_source = (
        "caller argument"
        if time_blocks
        else "campaign spec"
        if spec.preferred_time_blocks
        else "relevance engine"
        if engine_blocks
        else "commuter-peak default"
    )
    blocks = [str(b) for b in chosen]
    debug(f"STAGE 4 time blocks {blocks} taken from the {block_source}")
    unknown_blocks = [b for b in blocks if b not in engine.dayparts]
    if unknown_blocks:
        error(
            f"STAGE 4 rejected time_block_ids {unknown_blocks} — not in dim_slot "
            f"(valid: {sorted(engine.dayparts)})"
        )
        return {
            "status": "invalid_time_blocks",
            "run_id": run_id,
            "detail": (
                f"time_block_ids {unknown_blocks} are not in dim_slot. Valid ids: "
                f"{sorted(engine.dayparts)}."
            ),
        }

    # Inclusive end date: a 30-day flight starting on the 1st runs through the 30th.
    end_date = spec.start_date + timedelta(days=spec.duration_days - 1)
    screen_ids = [c.screen_id for c in candidates]
    industry = spec.industry_vertical or "retail"
    if not spec.industry_vertical:
        debug("STAGE 4 spec has no industry_vertical; using 'retail' for the price band")

    debug(
        f"STAGE 4 pricing {len(screen_ids)} candidates x {len(blocks)} time blocks "
        f"{blocks}, {spec.start_date}..{end_date}, slots_needed={slots_needed}"
    )

    # Audience volume per screen, keyed by block, from the relevance engine's artifact.
    audience = {c.screen_id: c for c in candidates}
    day_mix = _day_type_mix(spec.start_date, spec.duration_days)
    debug(
        f"STAGE 4 flight day mix: {day_mix['weekday']} weekday(s), "
        f"{day_mix['weekend']} weekend day(s)"
    )

    economics: list[ScreenEconomics] = []
    for block in blocks:
        campaign = {
            "industry_vertical": industry,
            "time_block_id": int(block),
            "start_date": spec.start_date,
            "end_date": end_date,
            "slots_needed": slots_needed,
            # city_id omitted on purpose: the engine uses each screen's own city, which is
            # what a multi-city campaign requires.
        }
        rows = engine.price_candidates(campaign, screen_ids)
        economics.extend(
            _to_contract(row, block, audience.get(row["screen_id"]), day_mix) for row in rows
        )

    _attach_demand_index(economics)

    # Aggregated rather than per row: `exposure.py` is called once per line and stays a
    # pure module, so the fallbacks it took are counted and reported here instead.
    assumed = sum(
        1
        for e in economics
        if e.daily_unique_audience > 0 and is_viewability_assumed(audience[e.screen_id].screen_type)
        if e.screen_id in audience
    )
    no_audience = sum(1 for e in economics if e.daily_unique_audience <= 0)
    debug(
        f"STAGE 4 audience mapping: {len(economics) - no_audience}/{len(economics)} lines "
        f"carry a modelled audience, {no_audience} report zero (block 1 and any block with "
        f"no scheduled service — 'not modelled', never 'nobody there')"
    )
    if assumed:
        info(
            f"STAGE 4 {assumed} line(s) took the default static viewability factor because "
            f"their screen_type is unrecognized — disclosed per row in `assumptions`"
        )

    feasible = [e for e in economics if e.feasible]
    if not feasible:
        error(f"STAGE 4 no feasible screen/block combination for run_id={run_id}")
        ref = write_records(
            ARTIFACT_KIND,
            economics,
            provenance="computed",
            summary={
                "rows": len(economics),
                "feasible_rows": 0,
                "time_blocks": sorted(set(blocks)),
                "demand_model": "transit_ridership (relevance engine)",
            },
        )
        run_state.set_artifact(run_id, ARTIFACT_KIND, ref)
        return {
            "status": "no_availability",
            "artifact": ref.as_context(),
            "detail": (
                f"All {len(economics)} screen/time-block combinations are sold out for "
                f"{slots_needed} slot(s)/day across {spec.start_date}..{end_date}. "
                f"No package is purchasable as specified."
            ),
            "relaxation_options": [
                "Reduce slots per day",
                "Shift or shorten the campaign window",
                "Add time blocks",
                "Broaden the geography to enlarge the candidate pool",
            ],
            "notice": AUDIENCE_NOTICE,
        }

    prices = [e.pricing.recommended_price for e in feasible]
    ref = write_records(
        ARTIFACT_KIND,
        economics,
        provenance="computed",
        summary={
            "rows": len(economics),
            "feasible_rows": len(feasible),
            "screens_priced": len({e.screen_id for e in feasible}),
            "time_blocks": sorted(set(blocks)),
            "price_min": round(min(prices), 2),
            "price_mean": round(sum(prices) / len(prices), 2),
            "price_max": round(max(prices), 2),
            "occupancy_mean": round(
                sum(e.occupancy_rate or 0.0 for e in feasible) / len(feasible), 4
            ),
            "booking_probability_mean": round(
                sum(e.pricing.booking_probability for e in feasible) / len(feasible), 4
            ),
            "viewed_exposures_per_slot_per_day_mean": round(
                sum(e.viewed_exposures_per_slot_per_day for e in feasible) / len(feasible), 1
            ),
            "reachable_daily_audience_total_naive": round(
                sum(e.reachable_daily_audience for e in feasible), 1
            ),
            "demand_model": "transit_ridership (relevance engine)",
        },
    )
    run_state.set_artifact(run_id, ARTIFACT_KIND, ref)
    info(
        f"STAGE 4 economics ready: {len(feasible)}/{len(economics)} feasible lines, "
        f"price {ref.summary['price_min']}-{ref.summary['price_max']} "
        f"(mean {ref.summary['price_mean']}), viewed exposures/slot/day mean "
        f"{ref.summary['viewed_exposures_per_slot_per_day_mean']:,.0f}, "
        f"artifact={ref.artifact_id}"
    )

    return {
        "status": "ok",
        "artifact": ref.as_context(),
        "screens_priced": ref.summary["screens_priced"],
        "lines_feasible": len(feasible),
        "lines_infeasible": len(economics) - len(feasible),
        "time_blocks": sorted(set(blocks)),
        "price_band": {
            "min": ref.summary["price_min"],
            "mean": ref.summary["price_mean"],
            "max": ref.summary["price_max"],
        },
        "occupancy_mean": ref.summary["occupancy_mean"],
        "booking_probability_mean": ref.summary["booking_probability_mean"],
        "viewed_exposures_per_slot_per_day_mean": ref.summary[
            "viewed_exposures_per_slot_per_day_mean"
        ],
        "pricing_basis": (
            "seller-side price = floor + occupancy_rate x (cap - floor), where the band is "
            "p25/p50/p90 of comparable historical bookings"
        ),
        "audience_basis": (
            "viewed_exposures_per_slot_per_day = the block's daily riders (weighted by this "
            "flight's weekday/weekend mix) x 8 loop passes / 6 slots x viewability. Reach is "
            "deduplicated by pool_key by the optimizer and capped at "
            "reachable_daily_audience — do not sum exposures to get it."
        ),
        "notice": AUDIENCE_NOTICE,
    }


@tool
def describe_pricing_model(run_id: str) -> dict:
    """Report how the pricing models were fitted and whether they are trustworthy.

    Reference lookup against the fitted engine — no campaign data involved. Use this to
    state *why* a price is credible, or to surface that the booking-probability model
    failed its sign check.

    Args:
        run_id: Handle for the campaign run. Accepted for consistency; unused.
    """
    from app.ml.engine import get_pricing_engine

    report = get_pricing_engine().training_report
    debug(
        f"STAGE 4 pricing model report requested (run_id={run_id}): "
        f"price_coef={report.price_coefficient:+.4f} "
        f"sign_ok={report.price_coefficient_sign_ok} auc={report.auc:.4f} "
        f"calibrated={report.calibration_ok}"
    )
    return {
        "booking_probability_model": {
            "form": "logistic regression on log(price) + screen/city/industry controls",
            "won_examples": report.n_won,
            "lost_examples": report.n_lost,
            "price_coefficient": round(report.price_coefficient, 4),
            "price_coefficient_sign_ok": report.price_coefficient_sign_ok,
            "auc": round(report.auc, 4),
            "calibrated": report.calibration_ok,
            "caveat": (
                "Correctly signed but weak within a segment: predicted P(booked) moves "
                "only ~0.1-0.2% across a segment's floor-cap band, so it is reported as a "
                "diagnostic and does not set the price."
            ),
        },
        "price_band": {
            "form": "p25/p50/p90 of contracted_price_per_slot_per_day",
            "segmentation": "screen_size x screen_type x position x city x daypart, "
            "with bounded fallbacks",
            "industry_adjustment_clamp": [0.85, 1.20],
        },
        "availability": {
            "form": "day-by-day slot occupancy from live bookings",
            "slot_capacity_per_screen_per_block_per_day": 6,
            "reported_figure": "tightest single day across the flight",
        },
        "demand_model": {
            "owner": "audience relevance engine (app/tools/relevance_tools.py)",
            "form": "average daily transit riders per screen x time block x day type",
            "mapped_here": (
                "block daily traffic, weighted by the flight's weekday/weekend mix, then "
                "through app/optimize/exposure.py: x loop passes per trip / 6 slots x "
                "viewability, giving viewed exposures for one slot on one day"
            ),
            "accuracy_metric": (
                "none — no held-out evaluation against a baseline exists yet, which is why "
                "no per-screen confidence is emitted"
            ),
        },
        "notice": AUDIENCE_NOTICE,
    }


def _day_type_mix(start: date, duration_days: int) -> dict[str, int]:
    """How many weekdays and weekend days the flight actually contains.

    Weekday and weekend ridership differ by roughly 6x, so a 30-day flight's audience is
    not 30 x the weekday figure. The mix is counted from real calendar dates rather than
    assumed to be 5/7.
    """
    counts = {"weekday": 0, "weekend": 0}
    for offset in range(duration_days):
        day = start + timedelta(days=offset)
        counts["weekend" if day.weekday() >= 5 else "weekday"] += 1
    return counts


def _daily_audience(
    candidate: ScreenCandidate | None, block: str, day_mix: dict[str, int]
) -> float:
    """Distinct people passing this screen's pool during this block, per day of the flight.

    Weighted by the flight's own weekday/weekend composition. This is the reach ceiling:
    buying more slots or more days shows the ad to these same people more often, it does
    not find new ones.
    """
    if candidate is None or not candidate.impressions_by_block:
        return 0.0
    total_days = sum(day_mix.values()) or 1
    weighted = sum(
        day_mix[dt] * candidate.impressions_by_block.get(f"{block}_{dt}", 0.0)
        for dt in ALL_DAY_TYPES
    )
    return weighted / total_days


def _to_contract(
    row: dict,
    block: str,
    candidate: ScreenCandidate | None,
    day_mix: dict[str, int],
) -> ScreenEconomics:
    """Engine row + candidate audience -> ScreenEconomics. Mapping only; no modelling.

    `availability` is left empty by design: the engine's availability contract is
    `max_slots_per_day`, the tightest single day across the flight, which is the figure
    the optimizer and validator must respect. A per-date list would restate it row by row.

    The audience unit conversion is the one piece of arithmetic here, and it is delegated
    to `app/optimize/exposure.py` so that exactly one module decides what an exposure is.
    People passing x loop passes per trip / 6 slots x viewability gives the viewed
    exposures one slot earns on one day, which scales correctly when the optimizer
    multiplies by slots x days. The same viewability discounts the reach ceiling, so both
    sides of the downstream reach min() are in viewed units.
    """
    pricing = None
    if row["feasible"]:
        pricing = PricingRecommendation(
            floor=row["floor_price"],
            target=row["target_price"],
            cap=row["cap_price"],
            recommended_price=row["recommended_price"],
            booking_probability=row["booking_probability"],
        )

    daily_audience = _daily_audience(candidate, block, day_mix)
    screen_type = candidate.screen_type if candidate else None
    assumptions = list(row["assumptions"])
    if daily_audience > 0 and is_viewability_assumed(screen_type):
        assumptions.append(
            f"unrecognized screen_type {screen_type!r}: viewability defaulted to the "
            f"static factor {viewability(screen_type)}"
        )

    return ScreenEconomics(
        screen_id=row["screen_id"],
        time_block_id=block,
        feasible=row["feasible"],
        max_slots_per_day=row["max_available_slots"] or 0,
        occupancy_rate=row["occupancy_rate"],
        price_by_slot_count=row["price_by_slot_count"] or {},
        viewed_exposures_per_slot_per_day=viewed_exposures_per_slot_per_day(
            daily_audience, screen_type
        ),
        daily_unique_audience=daily_audience,
        reachable_daily_audience=reachable_daily_audience(daily_audience, screen_type),
        viewability_factor=viewability(screen_type) if daily_audience > 0 else None,
        pool_key=candidate.pool_key if candidate else None,
        pricing=pricing,
        expected_revenue=row["expected_revenue"] or 0.0,
        seasonality_multiplier=row["seasonality_multiplier"],
        event_match_type=row["event_match_type"],
        pricing_internal_reach_proxy=row["pricing_internal_reach_proxy"],
        reach_owner=row["reach_owner"],
        assumptions=assumptions,
    )


def _attach_demand_index(economics: list[ScreenEconomics]) -> None:
    """Add the per-row DemandForecastSummary once the whole pool is known.

    `demand_index` is relative: this row's daily audience over the median across the priced
    pool. 1.0 is a typical line, 3.0 is three times the typical audience. It needs the pool
    to exist, so it is a second pass rather than part of the row mapping.

    `confidence` is left at its contract default. Neither the audience model nor the
    pricing model ships a held-out accuracy metric, so any number here would be invented —
    the validation layer skips its confidence check for exactly this reason.
    """
    volumes = sorted(e.daily_unique_audience for e in economics if e.daily_unique_audience > 0)
    if not volumes:
        return
    median = volumes[len(volumes) // 2] or 1.0
    for e in economics:
        e.demand_forecast = DemandForecastSummary(
            viewed_exposures_per_slot_per_day=e.viewed_exposures_per_slot_per_day,
            demand_index=e.daily_unique_audience / median,
        )


TOOLS = [estimate_screen_economics, describe_pricing_model]
