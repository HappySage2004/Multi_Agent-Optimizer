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
    pool_reachable_daily_audience: float = Field(
        default=0.0,
        description=(
            "THE REACH CEILING FOR THE WHOLE POOL: distinct people who look at any screen "
            "sharing this `pool_key` on a typical day. For a stop-mounted screen this equals "
            "`reachable_daily_audience` — every screen at the stop is passed by the same "
            "crowd. For a VEHICLE it is larger: `v_corridor_block_demand` divides a "
            "corridor's riders by its vehicle count to get one vehicle's share, so the "
            "per-screen figure is a fraction of the pool and capping reach against it "
            "understates the corridor by that same factor (up to ~9x). Cap deduplicated "
            "reach against THIS field, never against the per-screen one."
        ),
    )
    viewability_factor: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Share of passers-by assumed to look at this screen type (in-vehicle 0.65, "
            "stop-mounted 0.35). ASSUMED, no ground truth in the source data — recorded per row "
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

    # --- demand value / mispricing, from app/ml/demand_value.py ------------------
    demand_value_index: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "MERIT: what this screen is worth on what it physically delivers — riders, "
            "zone income, daytime activity, POI draw — as a percentile of its own "
            "screen_type x city. Computed WITHOUT ever seeing a price, which is what lets "
            "it disagree with the market instead of reproducing it."
        ),
    )
    historical_price_index: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "What this screen has actually transacted at, relative to its own comparables. "
            "1.0 is exactly its segment median, 0.85 is 15% under. None when the screen has "
            "too little booking history to say."
        ),
    )
    demand_premium: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Multiplier applied because merit exceeds the realized price rank — i.e. the "
            "screen is underpriced for the audience it delivers. 1.0 means no premium. "
            "Capped, gated on the screen actually selling, and fixed inventory only. This "
            "is the ONE adjustment that may carry a quote above the band cap, deliberately: "
            "an underpriced screen's own comparables are what understate it."
        ),
    )
    demand_value_reason: str | None = Field(
        default=None,
        description=(
            "Why a premium was or was not applied, citing the real figures. A screen with "
            "no premium is the interesting case — it says which gate stopped it."
        ),
    )
    reach_owner: str = "audience_engine"
    assumptions: list[str] = Field(
        default_factory=list,
        description="Which fallbacks fired and which adjustments applied, per row",
    )
