"""Demand + pricing contracts produced by the ML / Forecasting Agent."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class DemandForecast(BaseModel):
    """Per screen / date / time-block prediction."""

    screen_id: str
    date: date
    time_block_id: str

    predicted_impressions: float
    lower_bound: float
    upper_bound: float

    demand_index: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class DemandForecastSummary(BaseModel):
    expected_impressions: float
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
    """Everything the optimizer needs about one screen. Consolidated by the ML Agent."""

    screen_id: str
    time_block_id: str

    availability: list[TimeSlotAvailability] = []
    max_slots_per_day: int = Field(default=1, ge=0)

    demand_forecast: DemandForecastSummary
    pricing: PricingRecommendation

    expected_impressions: float = Field(
        description="Per slot per day, so the optimizer can scale by slots x duration"
    )
    expected_revenue: float = 0.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
