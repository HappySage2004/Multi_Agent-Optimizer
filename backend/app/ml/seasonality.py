"""M6 -- Seasonality & Event Adjuster.

Two independent adjustments, both deliberately scoped down (only ~6 months of ridership
history exists, so broader multi-year seasonality is out of reach):

1. Day-of-week + holiday multiplier, from ridership_actuals.
2. Event-overlap boost: screen's location near an overlapping event, weighted by
   attendance_tier.

KNOWN LIMITATION, stated explicitly rather than glossed over: there are no lat/lon
coordinates anywhere in this dataset, so a true radius-based "within impact_radius_km"
join is not possible. This uses exact location_id match (screen's location == event's
anchor location) as a strong signal, and city_zone match as a weaker fallback, instead of
true geographic distance. Mobile screens (no location_id) get no event boost at all --
flagged as not_applicable rather than silently defaulted to 1.0, so callers can tell the
difference between "no event nearby" and "we can't check for this screen type."

TWO CAVEATS CARRIED OVER FROM THE UPSTREAM IMPLEMENTATION, both verified against the data
and both intentionally left as-is:

* The combined multiplier scales PRICE, not demand. Ridership day-of-week factors run
  Friday 1.21 down to Sunday 0.32, and the mean over a full week is 0.913 -- so every
  campaign spanning whole weeks gets a ~9% haircut off a band that was already derived
  from actual contracted prices. A weekend-only campaign prices at ~0.37x. Whether the
  historical band already reflects this weekday/weekend mix (in which case the
  multiplier double-counts it) is an open question for the demand model to settle.
* The holiday multiplier is effectively inert. `ridership_actuals` spans 2026-02-19 to
  2026-08-19 and contains exactly TWO holiday dates, both in the past, so no future
  campaign window can match one. The code path is retained for when the ridership feed
  extends forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.logging_utils import debug, info

ATTENDANCE_BOOST = {"small": 1.03, "medium": 1.08, "large": 1.15}
ZONE_MATCH_DAMPING = 0.5  # weaker match -- apply only half the boost


@dataclass
class SeasonalityAdjustment:
    screen_id: str
    day_of_week_holiday_multiplier: float
    event_boost: float
    event_match_type: str  # 'location_match' | 'zone_match' | 'none' | 'not_applicable'
    combined_multiplier: float


class SeasonalityAdjuster:
    def __init__(
        self,
        ridership_seasonality: dict | None,
        events_df: pd.DataFrame,
        screens_df: pd.DataFrame,
        locations_df: pd.DataFrame,
    ):
        """`ridership_seasonality` is the aggregate payload from
        `loaders.load_ridership_seasonality()`, or None when the optional 124 MB
        ridership file was not provisioned -- in which case the day-of-week/holiday
        multiplier degrades to a neutral 1.0 and says so in the adjustment."""
        if ridership_seasonality is None:
            self._dow_multiplier: dict[str, float] = {}
            self._holiday_relative_multiplier = 1.0
            self._holiday_dates: set = set()
            self.ridership_available = False
        else:
            self._dow_multiplier = ridership_seasonality["dow_multiplier"]
            self._holiday_relative_multiplier = ridership_seasonality["holiday_relative_multiplier"]
            self._holiday_dates = ridership_seasonality["holiday_dates"]
            self.ridership_available = True

        # --- event lookups ---
        events_df = events_df.copy()
        events_df["start_date"] = pd.to_datetime(events_df["start_date"])
        events_df["end_date"] = pd.to_datetime(events_df["end_date"])
        self._events = events_df

        screens_df = screens_df.copy()
        self._screens = screens_df.set_index("screen_id")
        self._locations = locations_df.set_index("location_id")

        if self.ridership_available:
            debug(
                f"seasonality: day-of-week multipliers over "
                f"{len(self._dow_multiplier)} day(s) "
                f"[{min(self._dow_multiplier.values()):.3f}..."
                f"{max(self._dow_multiplier.values()):.3f}], holiday factor "
                f"{self._holiday_relative_multiplier:.3f} on "
                f"{len(self._holiday_dates):,} known holiday date(s)"
            )
        else:
            info(
                "seasonality: no ridership_actuals — day-of-week / holiday multiplier "
                "is a neutral 1.0 and is declared as such in every adjustment"
            )
        debug(
            f"seasonality: {len(self._events):,} event(s) indexed for location and zone "
            f"matching, attendance boosts={ATTENDANCE_BOOST}, "
            f"zone_match_damping={ZONE_MATCH_DAMPING}"
        )

    def _dow_holiday_multiplier(self, date_range: pd.DatetimeIndex) -> float:
        """Average of per-day multipliers across the requested window. Each day
        contributes (day-of-week factor) x (holiday factor if that specific date is a
        known holiday, else 1.0)."""
        multipliers = []
        for d in date_range:
            m = self._dow_multiplier.get(d.day_name().lower(), 1.0)
            if d.normalize() in self._holiday_dates:
                m *= self._holiday_relative_multiplier
            multipliers.append(m)
        return float(np.mean(multipliers)) if multipliers else 1.0

    def _event_boost(self, screen_id: str, start_date, end_date) -> tuple[float, str]:
        row = self._screens.loc[screen_id]
        location_id = row.get("location_id")
        if pd.isnull(location_id):
            return 1.0, "not_applicable"  # mobile screen -- no location to check

        start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
        overlapping = self._events[
            (self._events["start_date"] <= end_date) & (self._events["end_date"] >= start_date)
        ]
        if len(overlapping) == 0:
            return 1.0, "none"

        # strong match: exact location
        loc_matches = overlapping[overlapping["anchor_location_id"] == location_id]
        if len(loc_matches) > 0:
            best = loc_matches.loc[loc_matches["expected_attendance"].idxmax()]
            return ATTENDANCE_BOOST.get(best["attendance_tier"], 1.0), "location_match"

        # weak match: same zone. Matched on the `city_zone` display name, as upstream --
        # all 30 zone names happen to be unique across the three cities, so this does not
        # currently leak across cities, but it is keyed on a label rather than zone_id.
        if location_id in self._locations.index:
            zone = self._locations.loc[location_id, "city_zone"]
            zone_matches = overlapping[overlapping["city_zone"] == zone]
            if len(zone_matches) > 0:
                best = zone_matches.loc[zone_matches["expected_attendance"].idxmax()]
                boost = ATTENDANCE_BOOST.get(best["attendance_tier"], 1.0)
                damped = 1.0 + (boost - 1.0) * ZONE_MATCH_DAMPING
                return damped, "zone_match"

        return 1.0, "none"

    def get_adjustment(self, screen_id: str, start_date, end_date) -> SeasonalityAdjustment:
        date_range = pd.date_range(start_date, end_date, freq="D")
        dow_mult = self._dow_holiday_multiplier(date_range)
        event_mult, match_type = self._event_boost(screen_id, start_date, end_date)

        return SeasonalityAdjustment(
            screen_id=screen_id,
            day_of_week_holiday_multiplier=round(dow_mult, 4),
            event_boost=round(event_mult, 4),
            event_match_type=match_type,
            combined_multiplier=round(dow_mult * event_mult, 4),
        )
