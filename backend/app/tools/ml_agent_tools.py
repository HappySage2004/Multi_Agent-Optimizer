"""Tools for the ML / FORECASTING AGENT.

============================== INTEGRATION POINT ==============================
The ML Agent is owned by a separate implementer. Replace the body of
`_stub_screen_economics` with the real demand-forecast, market-price, and
booking-probability models.

Keep intact:
  * tool names and argument signatures
  * the ScreenEconomics contract  -- app/models/economics.py
  * artifact-reference return shape
  * `provenance="computed"` once real

Expected real behaviour (SOLUTION.md sections 6-9):
  demand      : route_schedules + ridership_actuals (+ events, POI, demographics)
                -> baseline, then LightGBM/XGBoost. Beat `estimated_ridership` as
                the naive baseline.
  price       : bookings.contracted_price_per_slot_per_day as target
  booking prob: lost_leads as the negative class (extreme 0.75% imbalance -- weight
                or calibrate; aggregate bookings to deal_id before pairing)
  expected_revenue = recommended_price * booking_probability
===============================================================================
"""

from __future__ import annotations

from datetime import timedelta

from langchain_core.tools import tool

from app.logging_utils import debug, error, info
from app.models.economics import (
    DemandForecastSummary,
    PricingRecommendation,
    ScreenEconomics,
    TimeSlotAvailability,
)
from app.models.screens import ScreenCandidate
from app.services import run_state
from app.services.artifact_store import read_models, write_records
from app.tools._stub_support import STUB_NOTICE, market_price_anchor, spread, unit

ARTIFACT_KIND = "screen_economics"
CANDIDATES_KIND = "screen_candidates"

# Blocks 2 and 5 are the commuter peaks (04:00-08:00, 16:00-20:00) per dim_slot.
DEFAULT_TIME_BLOCKS = ("2", "3", "5")


@tool
def estimate_screen_economics(run_id: str, time_blocks: list[str] | None = None) -> dict:
    """Forecast demand and recommend pricing for each candidate screen and time block.

    Consumes the `screen_candidates` artifact and produces the `screen_economics`
    artifact the optimizer needs. Returns an artifact reference plus aggregates.

    Args:
        run_id: Handle for the campaign run.
        time_blocks: dim_slot time_block_ids to price. Defaults to the commuter
            peaks and midday ("2", "3", "5"), or the campaign's preferred blocks.
    """
    if (blocked := run_state.missing_prerequisite(run_id, CANDIDATES_KIND)) is not None:
        error(f"STAGE 3 blocked: {blocked['detail']}")
        return blocked

    spec = run_state.get_spec(run_id)
    ref_candidates = run_state.require_artifact(run_id, CANDIDATES_KIND)
    candidates = read_models(ref_candidates, ScreenCandidate)

    blocks = [str(b) for b in (time_blocks or spec.preferred_time_blocks or DEFAULT_TIME_BLOCKS)]
    debug(f"STAGE 3 pricing {len(candidates)} candidates x {len(blocks)} time blocks {blocks}")
    economics = _stub_screen_economics(candidates, blocks, spec.start_date, spec.duration_days)

    ref = write_records(
        ARTIFACT_KIND,
        economics,
        provenance="stub",
        summary={
            "rows": len(economics),
            "screens": len({e.screen_id for e in economics}),
            "time_blocks": sorted(set(blocks)),
            "price_mean": round(
                sum(e.pricing.recommended_price for e in economics) / len(economics), 2
            ),
            "impressions_per_slot_day_mean": round(
                sum(e.expected_impressions for e in economics) / len(economics), 1
            ),
            "confidence_min": round(min(e.confidence for e in economics), 3),
        },
    )
    run_state.set_artifact(run_id, ARTIFACT_KIND, ref)
    info(
        f"STAGE 3 economics ready [STUB]: {len(economics)} rows, "
        f"mean price {ref.summary['price_mean']}, "
        f"min confidence {ref.summary['confidence_min']}, artifact={ref.artifact_id}"
    )

    return {
        "status": "ok",
        "artifact": ref.as_context(),
        "screens_priced": len({e.screen_id for e in economics}),
        "time_blocks": sorted(set(blocks)),
        "price_band": {
            "min": round(min(e.pricing.recommended_price for e in economics), 2),
            "max": round(max(e.pricing.recommended_price for e in economics), 2),
        },
        "warning": STUB_NOTICE,
    }


def _stub_screen_economics(
    candidates: list[ScreenCandidate],
    blocks: list[str],
    start_date,
    duration_days: int,
) -> list[ScreenEconomics]:
    """PLACEHOLDER economics. Replace with real demand + pricing + booking-probability."""
    anchor = market_price_anchor()
    rows: list[ScreenEconomics] = []

    for c in candidates:
        for block in blocks:
            demand_index = spread(unit(c.screen_id, block, "demand"), 0.45, 0.98)
            # Higher relevance and higher demand imply a stronger price, bounded by the
            # observed market band.
            price = spread(
                (demand_index + c.relevance_score) / 2, anchor["floor"] * 1.3, anchor["p90"]
            )
            booking_probability = spread(unit(c.screen_id, block, "bookprob"), 0.45, 0.92)
            confidence = spread(unit(c.screen_id, block, "conf"), 0.55, 0.90)
            impressions = round(
                spread(unit(c.screen_id, block, "impr"), 2_500, 14_000) * demand_index, 1
            )
            max_slots = 1 + int(unit(c.screen_id, block, "slots") * 4)  # 1..4

            rows.append(
                ScreenEconomics(
                    screen_id=c.screen_id,
                    time_block_id=block,
                    availability=[
                        TimeSlotAvailability(
                            date=start_date + timedelta(days=offset),
                            time_block_id=block,
                            available_slots=max_slots,
                        )
                        # Sampled, not the full flight: the optimizer reads
                        # max_slots_per_day, and a full daily grid would balloon the
                        # artifact for no gain.
                        for offset in range(min(duration_days, 3))
                    ],
                    max_slots_per_day=max_slots,
                    demand_forecast=DemandForecastSummary(
                        expected_impressions=impressions,
                        demand_index=round(demand_index, 4),
                        confidence=round(confidence, 3),
                    ),
                    pricing=PricingRecommendation(
                        floor=round(price * 0.82, 2),
                        target=round(price, 2),
                        cap=round(price * 1.25, 2),
                        recommended_price=round(price, 2),
                        booking_probability=round(booking_probability, 3),
                        confidence=round(confidence, 3),
                    ),
                    expected_impressions=impressions,
                    expected_revenue=round(price * booking_probability, 2),
                    confidence=round(confidence, 3),
                )
            )
    return rows


TOOLS = [estimate_screen_economics]
