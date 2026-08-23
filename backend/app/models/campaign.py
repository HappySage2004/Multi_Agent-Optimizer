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

INDUSTRY_VERTICALS: tuple[str, ...] = (
    "retail",
    "finance",
    "technology",
    "cpg",
    "entertainment",
    "auto",
    "telecom",
    "real_estate",
    "education",
    "healthcare",
    "hospitality",
    "nonprofit",
    "government",
)
"""The 13 real values of `bookings.industry_vertical`, lowercase snake_case.

Closed for the same reason as AUDIENCE_TERMS, and it was NOT closed for months. The field
was a bare `str | None`, and every value the Master actually wrote to a run failed to match
anything: `'AUTOMOTIVE / ELECTRIC VEHICLES'`, `'Consumer Tech'`, `'Fintech'`,
`'Beauty / Skincare'`.

Two sub-scores are keyed on this one string — `context_fit` (weight 0.15, via
`INDUSTRY_TO_POI_CONTEXT`) and `historical_performance` (weight 0.10, via the booking
history lookup). An unmatched value collapsed BOTH to a neutral 0.5 for every screen in
the pool, pinning 25% of `relevance_score` to a constant while the tool reported success.
The `conversion` objective made it worse: the MILP weights `conv_fit` (= `contextual_score`)
at 0.40, so 40% of what the solver maximized was a flat 0.5.

A rejected value costs one turn. An unmatched one silently deletes a quarter of the model.
"""

SCREEN_TYPES: tuple[str, ...] = ("bus_stop", "metro_station", "bus", "metro_rail_coach")
"""The four real values of `screens.screen_type`. Two fixed, two vehicle-mounted."""

DISPLAY_TYPE_NON_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "digital_screens_only",
        "digital_only",
        "digital",
        "static_screens_only",
        "exclude_static_screens",
        "screen_display_type",
    }
)
"""Keys that describe a display type, dropped from `hard_constraints` on sight.

EVERY SCREEN IN THIS NETWORK IS DIGITAL, so none of these can select anything. There is no
digital/static attribute in `screens.csv` to filter on, and the inventory model settles it
anyway: 6 ad slots rotating continuously through a 4-hour block is not a printed poster.

They are stripped rather than rejected because a brief saying "digital screens only" is
asking for something already true, and failing a whole run over a redundant phrase is
worse than ignoring it. Recording one, though, was worse still: it is not in
`ENFORCED_HARD_CONSTRAINTS`, so `validation._hard_constraint_checks` failed the package and
the agent reported the plan as "blocked by the digital-only constraint" — a constraint that
never meant anything. Unknown keys in general still fail loudly; only these are inert."""

TIME_BLOCK_IDS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
"""`dim_slot.time_block_id` as the strings every consumer compares against.

Six 4-hour windows. Stored as strings rather than ints because that is what
`CampaignSpec.preferred_time_blocks`, `hard_constraints["required_time_blocks"]` and
`Allocation.time_block_id` all carry, and a mixed-type comparison silently matches
nothing. `relevance_tools.ALL_TIME_BLOCKS` is the int-keyed twin used for column names.
"""

HARD_CONSTRAINT_SHAPES: dict[str, str] = {
    "min_screens": "int",
    "max_screens": "int",
    "min_zone_coverage": "int",
    # Type-coerced at intake but NOT range-checked there, because `contract.resolve_slot_cap`
    # is its single enforcement site and owns the rule that 0 and 7 are not purchasable
    # depths. Checking a range in two places is how one of them ends up bounding a value the
    # caller was never told about; a value this cannot read is passed through untouched so
    # the site that owns the constraint is the site that explains it.
    "max_slots_per_day": "int_passthrough",
    "min_budget_utilization": "fraction",
    "allowed_screen_types": "screen_types",
    "excluded_screen_types": "screen_types",
    "excluded_zone_ids": "id_list",
    "excluded_positions": "id_list",
    "required_time_blocks": "time_blocks",
    # A legacy alias for the line above, accepted by both consumers but deliberately NOT
    # advertised in `create_campaign_spec`'s docstring — one canonical name at intake, two
    # accepted on read, so an older spec still verifies.
    "time_blocks": "time_blocks",
}
"""The declared VALUE TYPE of every enforced hard constraint, keyed by name.

The types are here because they were nowhere, and an untyped constraint fails silently in
the worst possible direction. `hard_constraints={"allowed_screen_types": "metro_station"}`
is what a model writes when the brief names one screen type, and the consumer's
`list(allowed)` turned that single string into thirteen one-character screen types:
`.isin()` matched nothing, the pool emptied, and stage 2 reported that the spec's hard
constraints had eliminated every eligible screen. The constraint was satisfiable; the
value was the wrong shape.

`tools/coerce.normalize_hard_constraints` reads this map to reshape each value at intake
and rejects a key that is not in it. Adding an entry here without the code that enforces
it re-creates the bug `ENFORCED_HARD_CONSTRAINTS` exists to prevent.
"""

ENFORCED_HARD_CONSTRAINTS: frozenset[str] = frozenset(HARD_CONSTRAINT_SHAPES)
"""Every `hard_constraints` key some stage actually enforces.

Derived from `HARD_CONSTRAINT_SHAPES` so the vocabulary and the type map cannot drift —
a key that is enforced but untyped is exactly the state that shipped a broken filter.

`hard_constraints` is a free-form dict, and that freedom cost a real package: a brief
declaring "1 rotating slot per screen" was recorded as `max_slots_per_day: 1`, persisted,
echoed back to the Master in `normalized_spec` and in `get_active_run`'s
`campaign_inputs` — and then read by nobody, because every consumer matches against its
own hardcoded key list. The package shipped 3 slots/day and verification passed clean.

So the vocabulary is closed and `validation._hard_constraint_checks` FAILS a package whose
spec carries a key outside it. A constraint the brief declared and no stage enforces is
worse than a rejected one: the rep believes it was honoured. Adding a key here without
adding the code that enforces it re-creates exactly the bug this set exists to prevent.

Where each is enforced:
    min_screens, max_screens, min_zone_coverage,
    min_budget_utilization, max_slots_per_day     tools/or_agent_tools.py -> optimize/
    required_time_blocks / time_blocks            optimize/contract.py
    allowed_screen_types, excluded_*              tools/relevance_tools.py (stage 2 cuts)

Same pattern as AUDIENCE_TERMS above, one stage later: intake picks from a closed list,
and code — not an LLM — decides what an off-vocabulary entry means.
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
    industry_vertical: str | None = Field(
        default=None,
        description=(
            "One of INDUSTRY_VERTICALS, or None. Drives context_fit and "
            "historical_performance, which together carry 25% of relevance_score — so a "
            "value outside the list is rejected rather than scored as a near-miss."
        ),
    )
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

    screen_type_mix: list[str] = Field(
        default_factory=list,
        description=(
            "Screen types the brief wants REPRESENTED in the package, from SCREEN_TYPES. "
            "Different from hard_constraints['allowed_screen_types'], which only permits "
            "types: permitting bus and metro_station returned a pool that was 100% "
            "metro_station, because a single global relevance cut kept 250 of 4,629 and "
            "bus's best score sat below metro's worst. This field makes the candidate pool "
            "stratified per named type, so a mixed brief is servable at all. Empty means "
            "no mix was requested."
        ),
    )

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

    @field_validator("screen_type_mix")
    @classmethod
    def _screen_type_mix_known(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in SCREEN_TYPES]
        if unknown:
            raise ValueError(
                f"unknown screen_type_mix {unknown}; allowed values are {list(SCREEN_TYPES)}"
            )
        return list(dict.fromkeys(v))

    @field_validator("industry_vertical")
    @classmethod
    def _industry_vertical_known(cls, v: str | None) -> str | None:
        """Reject off-vocabulary verticals in code, exactly like `_audience_terms_known`.

        Case and separator variants are normalized first — 'Real Estate' and 'REAL_ESTATE'
        are unambiguously the vocabulary's `real_estate`, and failing those would reject a
        correct answer on formatting. Anything still unmatched is genuinely a different
        concept ('Fintech' is not `finance` by rule, and guessing is how the model went
        quiet in the first place), so it raises and the Master picks again.
        """
        if v is None:
            return None
        normalized = v.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in INDUSTRY_VERTICALS:
            return normalized
        raise ValueError(
            f"unknown industry_vertical {v!r}; allowed values are "
            f"{list(INDUSTRY_VERTICALS)}. Pick the closest one or leave it None — an "
            f"unmatched value neutralizes 25% of every relevance score."
        )

    @model_validator(mode="after")
    def _geography_present(self) -> CampaignSpec:
        if not (self.city_ids or self.zone_ids or self.corridor_ids):
            raise ValueError("at least one of city_ids / zone_ids / corridor_ids must be resolved")
        return self

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.duration_days - 1)
