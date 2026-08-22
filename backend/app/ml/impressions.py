"""M5 -- Impressions Estimator (pricing-internal diagnostic only).

Estimated Impressions = Base Traffic Volume x Dwell/Exposure Factor(screen_type)
                          x Visibility Factor(screen_size, position)

Two disjoint join paths, since screens split into two populations (verified: the
location_id / vehicle_id split is exactly complementary -- 8,548 fixed, 2,615 mobile, no
overlap and no gaps):
  - Fixed screens  (bus_stop, metro_station): location_id -> nearby POI footfall
  - Mobile screens (bus, metro_rail_coach):   vehicle_id -> corridor_id ->
                                              route_schedules.estimated_ridership

=========================== NOT A DEMAND FORECAST ===========================
This is descriptive context, NOT an input to the price and NOT the campaign's
impressions/reach figure. It is deliberately not wired into
any `ScreenEconomics` exposure field, and the OR stage does not read it. Reach and
impressions belong to the demand model, which is not integrated yet.

Two reasons it must not be promoted without work:
  * DWELL_FACTOR and VISIBILITY_* are DOCUMENTED ASSUMPTIONS, not fitted parameters --
    there is no ground-truth exposure data in this dataset to fit them against.
  * The fixed and mobile paths are on DIFFERENT UNITS. `est_daily_footfall` is per day;
    `route_schedules.estimated_ridership` averaged per corridor is per DEPARTURE (~139
    departures per corridor per weekday). Measured base traffic is 57-48,388 for fixed
    screens against 22-260 for mobile -- a ~36x gap that would make the 2,615 mobile
    screens invisible to any impressions-per-dollar ranking. Summing rather than
    averaging per corridor/day_type puts both on a daily basis (median 2,890 weekday vs
    2,059-2,774 fixed) and is the likely fix when this is reconciled.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.logging_utils import debug

# Documented assumption: how much of "traffic near the screen" actually looks at it, by
# physical screen type.
DWELL_FACTOR = {
    "metro_station": 0.70,  # people wait on platforms, often idle
    "bus_stop": 0.40,  # shorter dwell, more distracted
    "bus": 0.85,  # captive audience for the ride duration
    "metro_rail_coach": 0.85,
}

# Documented assumption: relative visibility by screen size, anchored loosely to the
# observed price-ratio evidence (L/M/S priced ~1.0/0.7/0.45x)
VISIBILITY_SIZE = {"L": 1.00, "M": 0.70, "S": 0.45}

# Documented assumption: relative visibility by position
VISIBILITY_POSITION = {
    "entrance_exit": 1.00,
    "platform": 0.85,
    "top": 0.75,
    "left": 0.65,
    "right": 0.65,
    "back": 0.40,
    "not_applicable": 0.80,  # in-vehicle screens -- no fixed position concept
}


@dataclass
class ImpressionsEstimate:
    screen_id: str
    base_traffic: float
    dwell_factor: float
    visibility_factor: float
    estimated_impressions: float
    method: str  # 'fixed_poi' | 'mobile_ridership' | 'fallback_no_data'


class ImpressionsEstimator:
    def __init__(
        self,
        screens_df: pd.DataFrame,
        poi_footfall_by_location: pd.Series,
        vehicles_df: pd.DataFrame,
        corridor_ridership: pd.Series,
    ):
        screens_df = screens_df.copy()
        screens_df["position"] = screens_df["position"].fillna("not_applicable")
        self.screens = screens_df.set_index("screen_id")

        self.poi_footfall_by_location = poi_footfall_by_location
        self.ridership_by_corridor = corridor_ridership
        self.vehicles = vehicles_df.set_index("vehicle_id")

        # Pricing diagnostics only — this figure is NEVER the campaign audience. See the
        # module docstring on the fixed/mobile unit mismatch.
        debug(
            f"impressions (pricing diagnostic only): {len(self.screens):,} screens, "
            f"POI footfall for {len(self.poi_footfall_by_location):,} location(s), "
            f"ridership for {len(self.ridership_by_corridor):,} corridor(s), "
            f"{len(self.vehicles):,} vehicle(s)"
        )

    def estimate(self, screen_id: str) -> ImpressionsEstimate:
        row = self.screens.loc[screen_id]
        size, stype, pos = row["screen_size"], row["screen_type"], row["position"]

        dwell = DWELL_FACTOR.get(stype, 0.6)  # generic fallback if a new type appears
        vis = VISIBILITY_SIZE.get(size, 0.6) * VISIBILITY_POSITION.get(pos, 0.6)

        if pd.notnull(row.get("location_id")):
            # fixed screen path
            base_traffic = self.poi_footfall_by_location.get(row["location_id"], np.nan)
            method = "fixed_poi"
            if pd.isnull(base_traffic):
                base_traffic = 0.0
                method = "fallback_no_data"
        elif pd.notnull(row.get("vehicle_id")):
            # mobile screen path
            vehicle = self.vehicles.loc[row["vehicle_id"]]
            corridor_id = vehicle["corridor_id"]
            base_traffic = self.ridership_by_corridor.get(corridor_id, np.nan)
            method = "mobile_ridership"
            if pd.isnull(base_traffic):
                base_traffic = 0.0
                method = "fallback_no_data"
        else:
            base_traffic = 0.0
            method = "fallback_no_data"

        estimated_impressions = base_traffic * dwell * vis

        return ImpressionsEstimate(
            screen_id=screen_id,
            base_traffic=round(float(base_traffic), 1),
            dwell_factor=dwell,
            visibility_factor=round(vis, 3),
            estimated_impressions=round(float(estimated_impressions), 1),
            method=method,
        )

    def estimate_batch(self, screen_ids) -> pd.DataFrame:
        rows = [self.estimate(sid) for sid in screen_ids]
        return pd.DataFrame(
            [
                {
                    "screen_id": r.screen_id,
                    "base_traffic": r.base_traffic,
                    "dwell_factor": r.dwell_factor,
                    "visibility_factor": r.visibility_factor,
                    "estimated_impressions": r.estimated_impressions,
                    "method": r.method,
                }
                for r in rows
            ]
        )
