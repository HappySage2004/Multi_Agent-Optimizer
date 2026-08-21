"""Validation + final recommendation contracts. Owned by the Master Agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.artifacts import Provenance
from app.models.optimization import OptimizedPackage

CheckStatus = Literal["pass", "fail", "skipped"]


class ValidationCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str
    expected: str | None = None
    observed: str | None = None


class ValidationResult(BaseModel):
    """Deterministic verification of specialist output. Computed in code, never by an LLM."""

    passed: bool
    checks: list[ValidationCheck] = []

    @property
    def failures(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "fail"]

    def summary(self) -> str:
        if self.passed:
            return f"All {len(self.checks)} validation checks passed."
        names = ", ".join(c.name for c in self.failures)
        return f"{len(self.failures)} of {len(self.checks)} checks FAILED: {names}"


class ScreenExplanation(BaseModel):
    screen_id: str
    explanation: str
    supporting_factors: list[str] = []


class AlternativePackage(BaseModel):
    name: str
    description: str
    package: OptimizedPackage
    tradeoffs: list[str] = []


class CampaignRecommendation(BaseModel):
    """Final sales-ready output. Explains analytical values; never recalculates them."""

    executive_summary: str

    recommended_package: OptimizedPackage

    key_recommendations: list[str] = []
    screen_explanations: list[ScreenExplanation] = []

    pricing_explanation: str = ""
    audience_explanation: str = ""
    optimization_explanation: str = ""

    risks: list[str] = []
    alternatives: list[AlternativePackage] = []

    validation: ValidationResult
    provenance: Provenance = "computed"
    provenance_note: str | None = Field(
        default=None,
        description="Set when any upstream specialist was a stub. Surfaced in the UI.",
    )
