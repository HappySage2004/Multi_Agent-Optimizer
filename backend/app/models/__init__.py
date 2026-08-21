"""Canonical inter-agent contracts. These interfaces stay stable as internals are replaced."""

from app.models.artifacts import ArtifactReference, Provenance
from app.models.campaign import AudienceTarget, CampaignSpec, OptimizationGoal
from app.models.economics import (
    DemandForecast,
    DemandForecastSummary,
    PricingRecommendation,
    ScreenEconomics,
    TimeSlotAvailability,
)
from app.models.optimization import (
    REASON_CODES,
    Allocation,
    InfeasibilityReport,
    OptimizationResult,
    OptimizedPackage,
)
from app.models.recommendation import (
    AlternativePackage,
    CampaignRecommendation,
    ScreenExplanation,
    ValidationCheck,
    ValidationResult,
)
from app.models.screens import ScreenCandidate, ScreenProfile

__all__ = [
    "REASON_CODES",
    "Allocation",
    "AlternativePackage",
    "ArtifactReference",
    "AudienceTarget",
    "CampaignRecommendation",
    "CampaignSpec",
    "DemandForecast",
    "DemandForecastSummary",
    "InfeasibilityReport",
    "OptimizationGoal",
    "OptimizationResult",
    "OptimizedPackage",
    "PricingRecommendation",
    "Provenance",
    "ScreenCandidate",
    "ScreenEconomics",
    "ScreenExplanation",
    "ScreenProfile",
    "TimeSlotAvailability",
    "ValidationCheck",
    "ValidationResult",
]
