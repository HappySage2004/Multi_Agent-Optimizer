"""Screen-level contracts produced by the Data Intelligence Agent."""

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
    """A screen that passed hard filtering, with an explainable relevance score."""

    screen_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)

    audience_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    geography_score: float = Field(default=0.0, ge=0.0, le=1.0)
    contextual_score: float = Field(default=0.0, ge=0.0, le=1.0)
    transit_score: float = Field(default=0.0, ge=0.0, le=1.0)

    reasons: list[str] = Field(
        default_factory=list,
        description="Must cite real feature values — no generic 'highly relevant' text",
    )
    hard_constraints_passed: bool = True

    # Carried through so the optimizer and validator need no extra lookups.
    city_id: str | None = None
    zone_id: str | None = None
    screen_type: str | None = None
