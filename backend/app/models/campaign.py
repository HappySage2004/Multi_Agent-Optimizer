"""CampaignSpec — the normalized campaign brief every downstream stage consumes."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

OptimizationGoal = Literal["reach", "frequency", "awareness", "conversion"]

DayTypeFocus = Literal["weekday", "weekend"]

AUDIENCE_TERMS: tuple[str, ...] = (
    "young_professionals",
    "professionals",
    "students",
    "families",
    "high_income",
    "commuters",
)
"""Closed vocabulary the audience relevance engine scores against.

Intake picks from this list; anything else is rejected deterministically rather than
scored as a near-miss. The engine maps each term onto specific demographic score columns
and preferred time blocks (`app/tools/relevance_tools.py`), so an unrecognized term has
no meaning there — it would silently collapse the audience sub-score to a 0.5 default.
"""


class AudienceTarget(BaseModel):
    age_range: tuple[int, int] | None = None
    income_range: tuple[float, float] | None = None
    occupations: list[str] = []
    commuter: bool | None = None
    other_attributes: dict[str, Any] = {}

    @field_validator("age_range")
    @classmethod
    def _age_order(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None and v[0] > v[1]:
            raise ValueError("age_range lower bound exceeds upper bound")
        return v


class CampaignSpec(BaseModel):
    """Normalized campaign brief. Produced by brief intake, consumed by every stage."""

    campaign_objective: str
    industry_vertical: str | None = None
    ad_type: str | None = None

    city_ids: list[str] = []
    zone_ids: list[str] = []
    corridor_ids: list[str] = []

    target_audience: AudienceTarget = Field(default_factory=AudienceTarget)
    audience_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Audience segments from AUDIENCE_TERMS, chosen by intake from the brief. The "
            "relevance engine scores against these; an empty list makes the audience "
            "sub-score fall back to a neutral 0.5 for every screen."
        ),
    )

    start_date: date
    duration_days: int
    budget: float

    requested_num_screens: int | None = None

    preferred_dayparts: list[str] = []
    preferred_time_blocks: list[str] = []
    day_type_focus: DayTypeFocus | None = Field(
        default=None,
        description=(
            "Score relevance against weekday or weekend traffic only. None scores both. "
            "Weekday and weekend ridership differ ~6x, so a weekend-weighted brief scored "
            "day-agnostically ranks against traffic it will not buy. This affects scoring "
            "only — the flight still runs every day in its window."
        ),
    )

    optimization_goal: OptimizationGoal

    hard_constraints: dict[str, Any] = {}
    soft_preferences: dict[str, Any] = {}

    original_query: str | None = Field(
        default=None, description="Verbatim user input, kept for traceability"
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Fields the intake stage could not determine. Never silently invented.",
    )

    @field_validator("budget")
    @classmethod
    def _budget_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("budget must be > 0")
        return v

    @field_validator("duration_days")
    @classmethod
    def _duration_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("duration_days must be > 0")
        return v

    @field_validator("requested_num_screens")
    @classmethod
    def _screens_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("requested_num_screens must be > 0 when provided")
        return v

    @field_validator("audience_terms")
    @classmethod
    def _audience_terms_known(cls, v: list[str]) -> list[str]:
        """Reject off-vocabulary terms in code rather than letting them score as 0.5.

        An LLM chooses these from the closed list in the tool docstring; this is the
        deterministic gate that stops an invented term from silently neutralizing the
        audience component of every score.
        """
        unknown = [t for t in v if t not in AUDIENCE_TERMS]
        if unknown:
            raise ValueError(
                f"unknown audience_terms {unknown}; allowed values are {list(AUDIENCE_TERMS)}"
            )
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _geography_present(self) -> CampaignSpec:
        if not (self.city_ids or self.zone_ids or self.corridor_ids):
            raise ValueError("at least one of city_ids / zone_ids / corridor_ids must be resolved")
        return self

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.duration_days - 1)
