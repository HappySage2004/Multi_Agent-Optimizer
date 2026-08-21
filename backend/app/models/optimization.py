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

    expected_impressions: float = Field(ge=0)
    expected_revenue: float = Field(ge=0)

    @property
    def line_cost(self) -> float:
        return self.price_per_slot_per_day * self.slots_per_day * self.duration_days


class OptimizedPackage(BaseModel):
    allocations: list[Allocation] = []

    total_cost: float = 0.0
    expected_impressions: float = 0.0
    expected_reach: float = 0.0
    expected_frequency: float = 0.0

    budget_utilization: float = 0.0

    constraint_status: dict[str, bool] = {}

    objective_value: float = 0.0
    optimization_method: str = "unspecified"

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
