"""The exposure model: people passing -> viewed exposures. ONE implementation.

Everything in this system that turns an audience figure into an exposure figure comes
through here. The constants are ASSUMED (`config.py`), so the single call site matters more
than the values: a second copy would let two stages disagree about what an "impression" is
without either of them being wrong on its own.

================================ THE UNIT CHAIN ================================
    v_screen_demand_history.daily_impressions
        riders passing a screen's POOL during a 4-hour block on a typical day
            |
            v  weight by the flight's real weekday/weekend day counts
    ScreenCandidate.impressions_by_block            PEOPLE PASSING, whole block
            |
            +--> x viewability                      -> reachable_daily_audience
            |        distinct people who LOOK at the screen. The reach ceiling.
            |
            +--> x LOOP_PASSES_PER_TRIP / SLOTS_PER_CELL x viewability
                     -> viewed_exposures_per_slot_per_day
                     what ONE slot earns on ONE day. Scales with slots x days.

A time block is a 4-hour window in which all 6 rotation slots cycle continuously. Slot
POSITION is meaningless — `slots_booked_per_day` is share of voice, never which positions —
so holding k of 6 slots means appearing on k of every 6 loop passes and viewed exposures
are strictly LINEAR in k. The only concavity in this system lives at the audience pool,
where a shared crowd genuinely saturates.

================================ WHY VIEWABILITY ================================
Upstream volume is people PASSING. Not all of them look. Applying the discount here rather
than upstream keeps the relevance engine reporting what it actually measures (transit
ridership) and keeps the attention assumption in one auditable place.

Both sides of the reach `min()` are discounted, deliberately. Exposures are viewed
exposures, so the ceiling must be viewed people too — capping viewed exposures at the
undiscounted crowd would let a plan claim it reached every passer-by when only ~35% of them
ever look, an over-claim of ~2.9x on a client-facing number. The handoff bundle discounts
the numerator and not the denominator; that asymmetry is not reproduced here.

Provenance note, because it was cited in support of these constants and does not hold:
`or_engine/config.py` justifies them with "their metro_station median of 227,981 x 0.2275 =
51,866 against our independently derived 48,706". 0.2275 is 0.35 x 0.65 — the static and
in-vehicle factors multiplied together. `metro_station` is static, so that comparison
should have used 0.35 (giving 79,793 against 48,706, a 1.6x miss). The cross-check
therefore validates a compound factor that is not shipped, and neither shipped constant is
evidenced by it. They are held on the argument that a viewability of 1.0 would label
footfall x 8 loop passes as "impressions" — an 8x over-count of people — and that a
wear-out judgement ("44 exposures is past useful") is only meaningful in viewed units.
"""

from __future__ import annotations

from app.optimize import config as C

# screen_type -> in-vehicle. Exact on this inventory: `v_screen_geography.inventory_class`
# is 'mobile' for exactly these two types and 'fixed' for bus_stop and metro_station
# (verified 2,615 mobile / 8,548 fixed). screen_type is used rather than inventory_class
# because it is what the ScreenCandidate contract carries.
IN_VEHICLE_SCREEN_TYPES = frozenset({"bus", "metro_rail_coach"})

STOP_MOUNTED_SCREEN_TYPES = frozenset({"bus_stop", "metro_station"})
"""Stop- and platform-mounted, as opposed to riding inside a vehicle. Nothing to do with
how the screen displays — every screen in this network is digital."""


def viewability(screen_type: str | None) -> float:
    """Share of passers-by who look at this screen type.

    An unrecognized or missing screen_type takes the STOP-MOUNTED factor — the lower of the two,
    so an unknown type is never flattered. Callers that need to disclose the substitution
    can compare against `is_viewability_assumed`.
    """
    if screen_type in IN_VEHICLE_SCREEN_TYPES:
        return C.VIEWABILITY_IN_VEHICLE
    return C.VIEWABILITY_STOP_MOUNTED


def is_viewability_assumed(screen_type: str | None) -> bool:
    """True when `viewability` fell back rather than recognizing the screen type."""
    return (
        screen_type not in IN_VEHICLE_SCREEN_TYPES and screen_type not in STOP_MOUNTED_SCREEN_TYPES
    )


def viewed_exposures_per_slot_per_day(pool_daily_audience: float, screen_type: str | None) -> float:
    """Viewed exposures ONE purchased slot earns on ONE day of the flight.

    `pool_daily_audience` is whole-block daily people passing this screen's pool. Multiply
    the result by slots x days to get a line's gross viewed exposures.
    """
    share_of_voice = C.LOOP_PASSES_PER_TRIP / C.SLOTS_PER_CELL
    return pool_daily_audience * share_of_voice * viewability(screen_type)


def reachable_daily_audience(pool_daily_audience: float, screen_type: str | None) -> float:
    """Distinct people who look at this screen's pool on a typical day — the reach ceiling.

    Does NOT scale with slots or days: buying more of either shows the ad to these same
    people more often. That is frequency.
    """
    return pool_daily_audience * viewability(screen_type)
