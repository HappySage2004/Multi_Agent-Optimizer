"""M1 -- Occupancy & Feasibility Engine.

Determines, for a given screen + time_block + date range, how many of the 6 fixed daily
rotation slots are already committed, and whether a new booking of a given size can fit.

Capacity is a KNOWN CONSTANT (6 slots per screen per time_block per day), confirmed
empirically against the bookings table. Occupancy is NOT static -- it changes day by day
as existing bookings start and end, so every query is evaluated across the full requested
date range, not as a single point estimate.

Port note: the occupancy arithmetic is unchanged. The only difference from the upstream
implementation is how the per-group index is built -- see `_build_index`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.logging_utils import debug

SLOT_CAPACITY = 6

_BOOKING_COLS = ["start_date", "end_date", "slots_booked_per_day"]


@dataclass
class OccupancyResult:
    screen_id: str
    time_block_id: int
    feasible: bool
    slots_needed: int
    min_available_slots: int  # worst day in the window
    avg_occupancy_rate: float  # average across window, for pricing
    max_occupancy_rate: float  # worst day, for feasibility framing
    daily_series: pd.DataFrame  # date, occupied_slots, occupancy_rate


def _build_index(df: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """Group bookings into one numpy block per (screen_id, time_block_id).

    Upstream used a `groupby(...).to_numpy()` dict comprehension. That is correct but
    costs ~34 s for the 41,904 groups in this dataset, which is prohibitive for a service
    singleton. Sorting once and slicing on group boundaries produces the *same* blocks in
    ~1.2 s (verified: identical key set, and identical row multisets for all 41,904
    groups). Row order within a group is irrelevant -- the only consumer sums a masked
    subset, which is order-independent.
    """
    if df.empty:
        return {}

    ordered = df.sort_values(["screen_id", "time_block_id"], kind="stable")
    block = ordered[_BOOKING_COLS].to_numpy()
    keys = pd.MultiIndex.from_arrays(
        [ordered["screen_id"].to_numpy(), ordered["time_block_id"].to_numpy()]
    )
    codes, uniques = pd.factorize(keys)
    starts = np.searchsorted(codes, np.arange(len(uniques)), side="left")
    ends = np.r_[starts[1:], len(codes)]
    return {uniques[i]: block[starts[i] : ends[i]] for i in range(len(uniques))}


class OccupancyEngine:
    def __init__(self, bookings_df: pd.DataFrame):
        """Pre-indexes bookings by (screen_id, time_block_id) so that a query never has
        to scan the full bookings table. Built once at load time."""
        df = bookings_df.copy()
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        # keep only the columns needed for occupancy math
        df = df[["screen_id", "time_block_id", *_BOOKING_COLS]]

        self._index = _build_index(df)
        self._n_bookings_indexed = len(df)
        self._n_groups = len(self._index)
        debug(
            f"occupancy: indexed {self._n_bookings_indexed:,} bookings into "
            f"{self._n_groups:,} (screen, time_block) groups, "
            f"{df['start_date'].min():%Y-%m-%d}..{df['end_date'].max():%Y-%m-%d}, "
            f"slot_capacity={SLOT_CAPACITY}/screen/block/day"
        )

    def _active_bookings(self, screen_id: str, time_block_id: int) -> np.ndarray:
        return self._index.get((screen_id, time_block_id), np.empty((0, 3), dtype=object))

    def get_daily_occupancy(
        self, screen_id: str, time_block_id: int, start_date, end_date
    ) -> pd.DataFrame:
        """Day-by-day occupied_slots / occupancy_rate across [start_date, end_date]
        inclusive, for one screen + time_block."""
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        dates = pd.date_range(start_date, end_date, freq="D")

        bookings = self._active_bookings(screen_id, time_block_id)

        occupied = np.zeros(len(dates), dtype=float)
        if len(bookings) > 0:
            b_start = bookings[:, 0]
            b_end = bookings[:, 1]
            b_slots = bookings[:, 2].astype(float)
            for i, d in enumerate(dates):
                mask = (b_start <= d) & (b_end >= d)
                occupied[i] = b_slots[mask].sum()

        return pd.DataFrame(
            {
                "date": dates,
                "occupied_slots": occupied,
                "occupancy_rate": occupied / SLOT_CAPACITY,
            }
        )

    def check_feasibility(
        self,
        screen_id: str,
        time_block_id: int,
        start_date,
        end_date,
        slots_needed: int = 1,
    ) -> OccupancyResult:
        """The main entry point. Feasibility + occupancy summary for the requested
        screen / time_block / date range / slot size."""
        daily = self.get_daily_occupancy(screen_id, time_block_id, start_date, end_date)
        available = SLOT_CAPACITY - daily["occupied_slots"]
        min_available = int(available.min()) if len(available) else SLOT_CAPACITY

        return OccupancyResult(
            screen_id=screen_id,
            time_block_id=time_block_id,
            feasible=bool(min_available >= slots_needed),
            slots_needed=slots_needed,
            min_available_slots=min_available,
            avg_occupancy_rate=float(daily["occupancy_rate"].mean()) if len(daily) else 0.0,
            max_occupancy_rate=float(daily["occupancy_rate"].max()) if len(daily) else 0.0,
            daily_series=daily,
        )

    def check_feasibility_batch(
        self,
        screen_ids: list[str],
        time_block_id: int,
        start_date,
        end_date,
        slots_needed: int = 1,
    ) -> pd.DataFrame:
        """Batch version for a candidate list of screens."""
        rows = []
        for sid in screen_ids:
            r = self.check_feasibility(sid, time_block_id, start_date, end_date, slots_needed)
            rows.append(
                {
                    "screen_id": r.screen_id,
                    "feasible": r.feasible,
                    "min_available_slots": r.min_available_slots,
                    "avg_occupancy_rate": r.avg_occupancy_rate,
                    "max_occupancy_rate": r.max_occupancy_rate,
                }
            )
        return pd.DataFrame(rows)
