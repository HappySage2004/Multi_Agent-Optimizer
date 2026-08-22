"""Demand + pricing contracts produced by the ML / Pricing Agent.

Two capabilities share this contract, and both are now populated:

* PRICING -- availability, price band, recommended price, booking probability and
  expected revenue, from the engine in `app/ml/`.
* AUDIENCE VOLUME -- `daily_unique_audience`, `reachable_daily_audience`,
  `viewed_exposures_per_slot_per_day`, `pool_key` and `demand_forecast`, mapped through
  from the relevance engine's ridership model (`app/tools/relevance_tools.py`) in
  `app/tools/ml_agent_tools.py::_to_contract`. Volume is derived entirely from transit
  schedules and ridership actuals; a block with no scheduled service reports zero, with no
  ambient/pedestrian term to fall back on.

THREE AUDIENCE FIGURES, AND THE UNIT IS IN THE NAME. `daily_unique_audience` is people
PASSING the pool. `reachable_daily_audience` is the subset who LOOK at the screen
(x viewability) and is the reach ceiling the optimizer and validator cap against.
`viewed_exposures_per_slot_per_day` is what one purchased slot earns on one day and scales
with slots x days. The single conversion between them lives in `app/optimize/exposure.py`.

`DemandForecast` -- the per screen/date/time-block contract -- remains unpopulated. The
audience model works at day-type granularity (weekday/weekend), not per calendar date, so
there is nothing to fill a per-date row with.

`confidence` is retained but not populated by any stage: neither the pricing nor the
audience model ships a held-out accuracy metric, and the validation layer skips its
confidence check rather than passing on a defaulted number.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class DemandForecast(BaseModel):
    """Per screen / date / time-block prediction. Reserved for the demand model."""

    screen_id: str
    date: date
    time_block_id: str

    predicted_impressions: float
    lower_bound: float
    upper_bound: float

    demand_index: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class DemandForecastSummary(BaseModel):
    viewed_exposures_per_slot_per_day: float
    demand_index: float = Field(ge=0.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PricingRecommendation(BaseModel):
    floor: float
    target: float
    cap: float
    recommended_price: float
    booking_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TimeSlotAvailability(BaseModel):
    date: date
    time_block_id: str
    available_slots: int = Field(ge=0)


class ScreenEconomics(BaseModel):
    """Everything the optimizer needs about one screen + time block.

    One row per candidate screen per time block. Rows where no slot is purchasable are
    retained with `feasible=False` and null pricing rather than dropped, so the caller can
    see what was excluded and why.
    """

    screen_id: str
    time_block_id: str

    feasible: bool = True

    availability: list[TimeSlotAvailability] = []
    max_slots_per_day: int = Field(
        default=0,
        ge=0,
        description=(
            "Slots purchasable every day of the flight — the tightest single day in the "
            "window, not an average. The engine's min_free_slots / max_available_slots."
        ),
    )
    occupancy_rate: float | None = Field(
        default=None, description="Mean committed-slot fraction across the window, 0-1"
    )
    price_by_slot_count: dict[int, float | None] = Field(
        default_factory=dict,
        description=(
            "Absolute price per slot per day at each slot count 1-6, null beyond "
            "availability. Flat by design — see app/ml/price_optimizer.py."
        ),
    )

    # --- audience volume, from the relevance engine's ridership model ---
    demand_forecast: DemandForecastSummary | None = None
    viewed_exposures_per_slot_per_day: float = Field(
        default=0.0,
        description=(
            "VIEWED exposures attributable to ONE purchased slot on ONE day of the flight, "
            "so the optimizer can scale by slots x duration. A time block is a 4-hour "
            "window in which all 6 rotation slots cycle continuously, so holding k slots "
            "puts the creative on k of every 6 loop passes and exposures are LINEAR in k. "
            "= daily_unique_audience x LOOP_PASSES_PER_TRIP / 6 x viewability. Zero means "
            "the audience model produced no figure — never a measured zero."
        ),
    )
    daily_unique_audience: float = Field(
        default=0.0,
        description=(
            "Distinct people PASSING this screen's POOL during this block on a typical day "
            "of the flight. Upstream truth from the relevance engine, carried for "
            "traceability — it is NOT the reach ceiling, because not everyone who passes "
            "looks. Cap against reachable_daily_audience instead."
        ),
    )
    reachable_daily_audience: float = Field(
        default=0.0,
        description=(
            "THE REACH CEILING: distinct people who look at this screen's pool on a typical "
            "day, = daily_unique_audience x viewability. Does not scale with slots or days "
            "— buying more of either raises frequency against these same people. Both sides "
            "of the reach min() are in viewed units, so a saturated plan claims the people "
            "who looked, not everyone who walked past."
        ),
    )
    viewability_factor: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Share of passers-by assumed to look at this screen type (in-vehicle 0.65, "
            "static 0.35). ASSUMED, no ground truth in the source data — recorded per row "
            "so any exposure figure is traceable to it."
        ),
    )
    pool_key: str | None = Field(
        default=None,
        description=(
            "Physical-audience unit carried through from the candidate. Screens sharing "
            "it see the same people; the optimizer must dedupe on it before reporting "
            "reach."
        ),
    )

    pricing: PricingRecommendation | None = None
    expected_revenue: float = 0.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- pricing diagnostics, carried through for explainability ---
    seasonality_multiplier: float | None = None
    event_match_type: str | None = Field(
        default=None,
        description="location_match | zone_match | none | not_applicable",
    )
    pricing_internal_reach_proxy: float | None = Field(
        default=None,
        description=(
            "NOT client-facing and NOT the campaign reach figure. Hand-set "
            "dwell/visibility heuristic with mismatched fixed/mobile units; kept for "
            "pricing diagnostics only. Reach belongs to the demand/audience model."
        ),
    )
    reach_owner: str = "audience_engine"
    assumptions: list[str] = Field(
        default_factory=list,
        description="Which fallbacks fired and which adjustments applied, per row",
    )
