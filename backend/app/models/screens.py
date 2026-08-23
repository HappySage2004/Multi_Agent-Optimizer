"""Screen-level contracts produced by the audience relevance engine (stages 2-3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScreenLocation(BaseModel):
    city_id: str
    zone_id: str | None = None
    corridor_id: str | None = None
    location_type: str | None = None


class ScreenAttributes(BaseModel):
    screen_type: str
    position: str | None = None
    screen_size: str | None = None


class AudienceFeatures(BaseModel):
    population: float | None = None
    density: float | None = None
    median_age: float | None = None
    pct_age_18_34: float | None = None
    median_income: float | None = None
    daytime_population_multiplier: float | None = None


class TransitFeatures(BaseModel):
    routes_serving: int | None = None
    estimated_daily_ridership: float | None = None


class ContextFeatures(BaseModel):
    poi_footfall: float | None = None
    nearby_events: int | None = None
    event_attendance: float | None = None


class ScreenProfile(BaseModel):
    """Feature row for one screen. Built deterministically — never by an LLM."""

    screen_id: str
    location: ScreenLocation
    screen_attributes: ScreenAttributes
    audience: AudienceFeatures = Field(default_factory=AudienceFeatures)
    transit: TransitFeatures = Field(default_factory=TransitFeatures)
    context: ContextFeatures = Field(default_factory=ContextFeatures)


class ScreenCandidate(BaseModel):
    """A screen that passed hard filtering, with an explainable relevance score.

    `relevance_score` is the weighted sum of exactly five components from the audience
    relevance engine (`app/tools/relevance_tools.py`):

        0.40 audience_match + 0.20 geography + 0.15 contextual
      + 0.15 time_of_day    + 0.10 historical_performance

    `transit_score` is reported alongside but deliberately NOT in that sum — volume is the
    optimizer's objective quantity, not a measure of fit. Every score is bounded 0-1; the
    engine renormalizes its raw components before they land here.
    """

    screen_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)

    audience_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    geography_score: float = Field(default=0.0, ge=0.0, le=1.0)
    contextual_score: float = Field(default=0.0, ge=0.0, le=1.0)
    transit_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Audience volume as a percentile of the eligible pool. Diagnostic — not part "
            "of relevance_score."
        ),
    )
    time_of_day_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of this screen's traffic falling in the campaign's target blocks",
    )
    historical_performance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Completion rate of past bookings on this screen in the same vertical",
    )

    reasons: list[str] = Field(
        default_factory=list,
        description="Must cite real feature values — no generic 'highly relevant' text",
    )
    defaults_applied: list[str] = Field(
        default_factory=list,
        description="Sub-scores that fell back to a neutral default, and why. Empty is good.",
    )
    hard_constraints_passed: bool = True

    # --- audience volume ---------------------------------------------------------
    pool_key: str | None = Field(
        default=None,
        description=(
            "The physical-audience unit: a synthetic SITE id for stop-mounted screens, "
            "corridor_id for vehicle-mounted ones. A site is (city, name, serving "
            "corridors), not a raw location_id — one physical station is modelled as "
            "several location rows, and 910 stop-mounted location_ids resolve to 878 "
            "sites. Screens sharing a pool_key see the SAME people, so anything summing "
            "audience across screens MUST group by this first or it over-counts."
        ),
    )
    pool_partition_count: int = Field(
        default=1,
        ge=1,
        description=(
            "How many partitions the pool's audience was divided into to produce this "
            "screen's figure. 1 for stop-mounted screens — every screen at a location is "
            "passed by the same crowd, so the per-screen figure IS the pool's. For "
            "vehicle-mounted screens it is the vehicles working the corridor, because the "
            "demand view divides the corridor by them. The optimizer needs the pool's whole "
            "crowd (per-screen x this) or it under-buys in-vehicle inventory by this factor."
        ),
    )
    impressions_by_block: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "PEOPLE PASSING — average riders passing this screen's pool on a typical day, "
            "keyed '{time_block_id}_{weekday|weekend}', all 12 combinations present. NOT "
            "viewed exposures: no viewability discount is applied here, and it is a "
            "whole-block daily figure, not per slot and not per campaign. "
            "`app/optimize/exposure.py` is the only place it becomes an exposure count. "
            "Block 1 (00:00-04:00) is always 0: no scheduled service starts then, which "
            "means 'not modelled', not 'nobody there'."
        ),
    )
    impressions_weekday: float = Field(
        default=0.0, ge=0.0, description="People passing, not viewed exposures"
    )
    impressions_weekend: float = Field(
        default=0.0, ge=0.0, description="People passing, not viewed exposures"
    )
    impressions_block_1_estimated: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "ESTIMATE, NOT A MEASUREMENT. Block 1 (00:00-04:00) has no scheduled transit "
            "service, so its measured volume is exactly 0 for every screen — while "
            "`bookings` holds 8,544 real block-1 bookings, so inventory there demonstrably "
            "sells. This is an 8% baseline assumption relative to the same screen's block-6 "
            "volume, keyed '{weekday|weekend}'. Kept strictly separate from "
            "impressions_by_block['1_*'] (which stays at the measured 0) and excluded from "
            "every total, peak, off-peak and commuter_score figure, so no validated number "
            "moves with the assumption. Quote it only as an estimate."
        ),
    )
    nearby_ambient_footfall: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Foot-traffic proxy from nearby POIs. Independently verified to correlate "
            "weakly with transit ridership and can disagree by up to ~20x at individual "
            "locations. Supplementary signal only — not a reach or pricing input."
        ),
    )

    # Carried through so the optimizer and validator need no extra lookups.
    city_id: str | None = None
    zone_id: str | None = None
    location_name: str | None = Field(
        default=None,
        description=(
            "The stop or station name — 'Grant Rd & Kingsley Rd', 'East Commons Station'. "
            "From `locations.name`. This is the label a CLIENT understands, so it is what "
            "anything client-facing shows, ahead of the zone. Null for vehicle-mounted "
            "screens, which have no fixed location; their corridor names them. Also the "
            "only human-readable handle on a pool: `pool_key` is a synthetic site id, so "
            "a reason string names the site through this field, never the key."
        ),
    )
    zone_name: str | None = Field(
        default=None,
        description=(
            "The zone's real name — 'Financial Row', not 'LH-ZONE-005'. Null for "
            "vehicle-mounted screens, which have no zone; their geography is the corridor. "
            "Anything shown to a salesperson should prefer this over zone_id."
        ),
    )
    corridor_id: str | None = None
    screen_type: str | None = None
