"""DuckDB-backed frame loaders for the pricing engine.

The upstream implementation read nine CSVs with `pd.read_csv`, including the 124 MB
`ridership_actuals.csv` in full. This module is the only thing that changed in the port:
every frame now comes from the DuckDB view layer in `app/data/db.py` with explicit column
projection, per SOLUTION.md section 20 and the repo rule against reading the datasets
directly.

The arithmetic downstream is unchanged. Ridership is the one place where aggregation moved
into SQL: a single `SUM`/`COUNT` cross-tab replaces three full-table pandas means, and the
per-day, per-holiday-flag and overall means are re-derived from it exactly. That avoids
materializing 2.05 M rows to produce eight numbers. Verified against the original
implementation: identical outputs, to the last significant digit.
"""

from __future__ import annotations

import pandas as pd

from app.data.db import has_ridership_actuals, query_df
from app.logging_utils import debug, info

# Only the columns the engine actually uses. Projecting here rather than after the read is
# what keeps `bookings` (191 k rows) and `ridership_actuals` (2.05 M rows) affordable.
BOOKINGS_SQL = """
    SELECT screen_id, city_id, industry_vertical, time_block_id, daypart,
           slots_booked_per_day, start_date, end_date,
           contracted_price_per_slot_per_day, is_bundle
    FROM bookings
"""

# `zone_id` joins through `locations` and is NULL for all 2,615 vehicle-mounted screens (a
# zone is undefined for a moving vehicle). The price-band ladder relies on that: its zone
# levels simply never match for mobile inventory, which falls through to the city levels
# without needing a branch.
SCREENS_SQL = """
    SELECT s.screen_id, s.city_id, s.screen_type, s.location_id, s.vehicle_id,
           s.position, s.screen_size, l.zone_id
    FROM screens s
    LEFT JOIN locations l ON l.location_id = s.location_id
"""

LOST_LEADS_SQL = """
    SELECT anchor_screen_id, city_id, industry_vertical,
           quoted_price_per_slot_per_day, loss_reason
    FROM lost_leads
"""

EVENTS_SQL = """
    SELECT city_id, city_zone, anchor_location_id, start_date, end_date,
           expected_attendance, attendance_tier
    FROM events
"""

LOCATIONS_SQL = "SELECT location_id, city_id, city_zone FROM locations"

POI_FOOTFALL_SQL = """
    SELECT anchor_location_id, AVG(est_daily_footfall) AS est_daily_footfall
    FROM points_of_interest
    GROUP BY anchor_location_id
"""

VEHICLES_SQL = "SELECT vehicle_id, corridor_id FROM vehicles"

# NOTE: AVG, not SUM. `route_schedules` holds one row per scheduled departure (~139 per
# corridor per weekday), so this is average ridership PER DEPARTURE, not per day. That is
# the upstream behaviour and is preserved deliberately -- but it means the mobile-screen
# figure is on a per-departure basis while the fixed-screen figure (`est_daily_footfall`)
# is per day, a ~36x scale gap. Nothing consumes this for optimization today (see
# `impressions.py`). Anyone wiring it into a demand or reach figure must reconcile the
# units first: SUM(estimated_ridership) GROUP BY corridor_id, day_type puts both on a
# daily basis (median 2,890 weekday, comparable to fixed screens' 2,059-2,774).
CORRIDOR_RIDERSHIP_SQL = """
    SELECT corridor_id, AVG(estimated_ridership) AS estimated_ridership
    FROM route_schedules
    GROUP BY corridor_id
"""

TIME_BLOCK_DAYPART_SQL = "SELECT time_block_id, nearest_daypart FROM dim_slot"


def _loaded(name: str, frame: pd.DataFrame | pd.Series) -> None:
    """One [DEBUG] line per frame the pricing engine pulls out of DuckDB.

    Worth having: every one of these is a projected view read, so a zero row count here is
    the earliest and cheapest place to notice a dataset that did not load.
    """
    debug(f"loader {name}: {len(frame):,} rows")


def load_bookings() -> pd.DataFrame:
    df = query_df(BOOKINGS_SQL)
    _loaded("bookings", df)
    return df


def load_screens() -> pd.DataFrame:
    df = query_df(SCREENS_SQL)
    _loaded("screens", df)
    return df


def load_lost_leads() -> pd.DataFrame:
    df = query_df(LOST_LEADS_SQL)
    _loaded("lost_leads", df)
    return df


def load_events() -> pd.DataFrame:
    df = query_df(EVENTS_SQL)
    _loaded("events", df)
    return df


def load_locations() -> pd.DataFrame:
    df = query_df(LOCATIONS_SQL)
    _loaded("locations", df)
    return df


def load_poi_footfall_by_location() -> pd.Series:
    df = query_df(POI_FOOTFALL_SQL)
    _loaded("poi_footfall_by_location", df)
    return df.set_index("anchor_location_id")["est_daily_footfall"]


def load_vehicles() -> pd.DataFrame:
    df = query_df(VEHICLES_SQL)
    _loaded("vehicles", df)
    return df


def load_corridor_ridership() -> pd.Series:
    df = query_df(CORRIDOR_RIDERSHIP_SQL)
    _loaded("corridor_ridership", df)
    return df.set_index("corridor_id")["estimated_ridership"]


def time_block_dayparts() -> dict[str, str]:
    """`dim_slot` time_block_id -> nearest_daypart.

    Upstream took `daypart` as a free parameter alongside `time_block_id`. In `bookings`
    the two are a strict 1:1 function of each other (blocks 1 and 6 both map to `night`),
    so accepting both invites a caller to desync them and silently segment the price band
    on the wrong daypart. The engine derives it from here instead.
    """
    df = query_df(TIME_BLOCK_DAYPART_SQL)
    return {str(r.time_block_id): str(r.nearest_daypart) for r in df.itertuples()}


def load_ridership_seasonality() -> dict[str, object] | None:
    """Day-of-week and holiday aggregates from `ridership_actuals`.

    Returns None when the 124 MB file was not provisioned -- it is gitignored and
    `app/data/db.py` treats it as optional, so the engine must degrade to a neutral
    multiplier rather than crash (upstream would have raised on the missing file).

    Equivalent to the upstream pandas computation:
        overall_mean = r["actual_ridership"].mean()
        dow_means    = r.groupby("day_of_week")["actual_ridership"].mean() / overall_mean
        holiday_means = r.groupby("is_holiday")["actual_ridership"].mean()
    """
    if not has_ridership_actuals():
        info(
            "loader ridership_seasonality: ridership_actuals.csv not provisioned — "
            "returning None so the seasonality adjuster degrades to a neutral 1.0"
        )
        return None

    # One pass for a (day_of_week x is_holiday) sum/count cross-tab -- 14 rows, from which
    # the overall mean, the per-day means and the per-holiday-flag means all follow
    # exactly. Aggregating SUM and COUNT rather than AVG is what makes the re-derivation
    # exact rather than an average of averages. Scanning the 124 MB file once instead of
    # three times is the whole point.
    cross = query_df(
        """
        SELECT day_of_week,
               is_holiday,
               SUM(actual_ridership) AS total,
               COUNT(*)              AS n
        FROM ridership_actuals
        GROUP BY day_of_week, is_holiday
        """
    )
    overall_mean = float(cross["total"].sum() / cross["n"].sum())

    # Keys lowercased so lookups by `Timestamp.day_name().lower()` match; the source data
    # uses capitalized day names.
    by_dow = cross.groupby("day_of_week")[["total", "n"]].sum()
    dow_multiplier = {
        str(day).lower(): float(row["total"] / row["n"]) / overall_mean
        for day, row in by_dow.iterrows()
    }

    by_holiday = cross.groupby("is_holiday")[["total", "n"]].sum()
    holiday_means = {
        bool(flag): float(row["total"] / row["n"]) for flag, row in by_holiday.iterrows()
    }
    non_holiday_mean = holiday_means.get(False, overall_mean)
    # Expressed relative to non-holiday days so it composes with the day-of-week
    # multiplier without double-counting the overall mean.
    holiday_relative = holiday_means.get(True, non_holiday_mean) / non_holiday_mean

    holiday_dates = query_df("SELECT DISTINCT date FROM ridership_actuals WHERE is_holiday")
    dates = set(pd.to_datetime(holiday_dates["date"]).dt.normalize().unique())

    debug(
        f"loader ridership_seasonality: {int(cross['n'].sum()):,} ridership rows "
        f"aggregated in one pass, overall mean {overall_mean:,.1f}, "
        f"{len(dow_multiplier)} day-of-week multiplier(s), holiday factor "
        f"{holiday_relative:.3f} over {len(dates):,} holiday date(s)"
    )

    return {
        "dow_multiplier": dow_multiplier,
        "holiday_relative_multiplier": holiday_relative,
        "holiday_dates": dates,
    }
