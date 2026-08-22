"""Optimization contracts produced by the OR / Optimization Agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SolveStatus = Literal["optimal", "feasible", "infeasible", "error"]

REASON_CODES = (
    "BUDGET_CONSTRAINT",
    "INSUFFICIENT_INVENTORY",
    "GEOGRAPHY_UNAVAILABLE",
    "DATES_UNAVAILABLE",
    "TOO_MANY_SCREENS_REQUESTED",
    "DAYPART_UNAVAILABLE",
    "CONFLICTING_HARD_CONSTRAINTS",
    "NO_CANDIDATES",
)


class Allocation(BaseModel):
    screen_id: str
    time_block_id: str

    slots_per_day: int = Field(ge=1)
    duration_days: int = Field(ge=1)

    price_per_slot_per_day: float = Field(ge=0)

    viewed_exposures: float = Field(
        ge=0,
        description=(
            "Gross VIEWED exposures this line delivers over the flight, "
            "= viewed_exposures_per_slot_per_day x slots x days. Exposures, not people."
        ),
    )
    expected_revenue: float = Field(ge=0)

    @property
    def line_cost(self) -> float:
        return self.price_per_slot_per_day * self.slots_per_day * self.duration_days


class OptimizedPackage(BaseModel):
    """The package the optimizer selected.

    Reach and impressions are different quantities and the gap is large. Never report
    `gross_impressions_viewed` as a number of people.
    """

    allocations: list[Allocation] = []

    total_cost: float = 0.0
    gross_impressions_viewed: float = Field(
        default=0.0,
        description=(
            "Total VIEWED exposures over the flight — the sum over allocations. Scales with "
            "slots x days. Internal and never client-facing as an audience size."
        ),
    )
    expected_reach: float = Field(
        default=0.0,
        description=(
            "DISTINCT PEOPLE, deduplicated by (pool_key, time block) and capped at each "
            "pool's reachable_daily_audience. Saturates: more slots, days or screens in the "
            "same pool buy frequency, not reach. Recomputed independently by the validator."
        ),
    )
    expected_frequency: float = 0.0

    budget_utilization: float = 0.0

    constraint_status: dict[str, bool] = {}

    objective_value: float = 0.0
    optimization_method: str = "unspecified"

    # --- solver diagnostics, additive and never load-bearing -----------------
    curve_reach_diagnostic: float | None = Field(
        default=None,
        description=(
            "Reach under the solver's internal saturation curve, "
            "P x (1 - exp(-lambda x E / P)) with lambda ASSUMED at 0.9. Reported for "
            "comparison ONLY: the definition this system stands behind is the lambda-free "
            "min() in expected_reach. Guarded by curve_reach <= min(sum E, sum P)."
        ),
    )
    unmet_coverage: dict[str, float] = Field(
        default_factory=dict,
        description="Coverage groups the plan could not satisfy, and by how much. Report it.",
    )
    wear_out_exposures_over_cap: float = Field(
        default=0.0,
        description=(
            "Viewed exposures beyond the advisory wear-out cap. Non-zero is expected on any "
            "long flight — the floor is LOOP_PASSES_PER_TRIP / 6 x days regardless of what "
            "the optimizer chooses, so the cap constrains stacking, not total exposure."
        ),
    )

    @property
    def screen_ids(self) -> list[str]:
        return sorted({a.screen_id for a in self.allocations})


class InfeasibilityReport(BaseModel):
    """Returned instead of a package when no feasible solution exists. Never fabricate one."""

    status: Literal["infeasible", "error"] = "infeasible"
    reason_codes: list[str] = []
    explanation: str
    relaxation_options: list[str] = []


class OptimizationResult(BaseModel):
    """Discriminated result — exactly one of `package` / `infeasibility` is set."""

    status: SolveStatus
    package: OptimizedPackage | None = None
    infeasibility: InfeasibilityReport | None = None
    solver_log: list[str] = []
