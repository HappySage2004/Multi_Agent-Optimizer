"""Helpers shared by the specialist stubs.

Stub numbers are *deterministic* (hash-derived, not random) so a given campaign always
produces the same placeholder package. That keeps the Master Agent's validation layer
testable and the demo reproducible.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from app.data.db import query_df

STUB_NOTICE = (
    "STUB OUTPUT — this stage is not yet implemented. Numbers are deterministic "
    "placeholders derived from screen IDs, not analysis of the data."
)


def unit(*parts: object) -> float:
    """Stable pseudo-random float in [0, 1) derived from the given parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def spread(value: float, lo: float, hi: float) -> float:
    """Map a [0,1) unit value onto [lo, hi]."""
    return lo + value * (hi - lo)


@lru_cache(maxsize=1)
def market_price_anchor() -> dict[str, float]:
    """Real aggregate from bookings, used only to keep stub prices in a plausible band.

    The actual pricing model (market price + booking probability) is the ML Agent's job.
    """
    df = query_df(
        """
        SELECT
            min(contracted_price_per_slot_per_day)                              AS floor,
            avg(contracted_price_per_slot_per_day)                              AS mean,
            quantile_cont(contracted_price_per_slot_per_day, 0.9)               AS p90,
            max(contracted_price_per_slot_per_day)                              AS cap
        FROM bookings
        """
    )
    row = df.iloc[0]
    return {
        "floor": float(row["floor"]),
        "mean": float(row["mean"]),
        "p90": float(row["p90"]),
        "cap": float(row["cap"]),
    }
