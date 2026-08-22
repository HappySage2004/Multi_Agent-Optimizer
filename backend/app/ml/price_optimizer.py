"""M4 -- Price Optimizer.

Ties M1 (occupancy/feasibility), M2 (price band) and M3 (booking probability) together
into the actual pricing decision. Port note: the decision rule, the grid size, the 15%
divergence threshold and the flat per-slot curve are all unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ml.occupancy import SLOT_CAPACITY


@dataclass
class ScreenPricing:
    screen_id: str
    feasible: bool
    time_block_id: int = None  # echoed for safe joins across per-block calls
    floor_price: float = None
    target_price: float = None
    cap_price: float = None
    occupancy_rate: float = None
    min_free_slots: int = None  # tightest single day in window
    max_available_slots: int = None  # == min_free_slots; largest purchasable slot count
    recommended_price: float = None
    booking_probability: float = None
    expected_revenue: float = None
    price_by_slot_count: dict = None  # {1..6: price|None}
    assumptions: list = field(default_factory=list)


class PriceOptimizer:
    def __init__(self, occupancy_engine, price_band_engine, booking_model, n_grid_points=50):
        self.occ = occupancy_engine
        self.band = price_band_engine
        self.prob_model = booking_model
        self.n_grid_points = n_grid_points

    def price_screen(
        self,
        screen_id: str,
        screen_size: str,
        screen_type: str,
        position: str,
        city_id: str,
        time_block_id: int,
        daypart: str,
        industry_vertical: str,
        start_date,
        end_date,
        slots_needed: int = 1,
        price_multiplier: float = 1.0,
    ) -> ScreenPricing:
        assumptions: list[str] = []

        # Step 1: feasibility gate
        occ_result = self.occ.check_feasibility(
            screen_id, time_block_id, start_date, end_date, slots_needed
        )

        # Occupancy diagnostics are reported even when INFEASIBLE, so consumers can
        # distinguish a near-miss (1 slot short) from a fully sold screen, and can see the
        # largest slot count that WOULD have been purchasable.
        if not occ_result.feasible:
            return ScreenPricing(
                screen_id=screen_id,
                feasible=False,
                time_block_id=time_block_id,
                occupancy_rate=round(occ_result.avg_occupancy_rate, 3),
                min_free_slots=occ_result.min_available_slots,
                max_available_slots=occ_result.min_available_slots,
                price_by_slot_count={n: None for n in range(1, SLOT_CAPACITY + 1)},
                assumptions=[
                    (
                        f"infeasible: only {occ_result.min_available_slots} "
                        f"slots available, {slots_needed} needed"
                    )
                ],
            )

        # Step 2: price band, with optional seasonality/event multiplier (computed upstream
        # by M6 and passed in here so M2/M4 stay independent of M6)
        pb = self.band.get_price_band(screen_id, daypart, industry_vertical, city_id)
        assumptions.extend(pb.assumptions)
        floor_price = pb.floor_price * price_multiplier
        target_price = pb.target_price * price_multiplier
        cap_price = pb.cap_price * price_multiplier
        if abs(price_multiplier - 1.0) > 1e-6:
            assumptions.append(f"seasonality/event multiplier applied: x{price_multiplier:.3f}")

        # Step 3: occupancy-adjusted price -- THE PRIMARY DRIVER.
        #
        # Design note (found during testing, not assumed upfront): a pure argmax over the
        # booking-probability curve degenerates to the cap price for every screen.
        # Diagnosed cause: within any one segment's realistic floor-cap band, predicted
        # P(booked) drops by ~0.1-0.2% -- the elasticity signal is correctly signed but far
        # too diffuse to differentiate prices within-segment, because it's estimated from
        # only 393 price-driven-loss examples spread across thousands of
        # screen/city/industry combinations. It only becomes visible across much larger
        # price swings than any single segment actually spans.
        #
        # Occupancy (M1) is comparatively strong, exact, and validated against real
        # bookings -- so it drives the primary recommendation. The probability model is
        # retained as a secondary diagnostic: reported alongside the price, and used to
        # flag cases where it would suggest a materially different price than occupancy
        # did (rare, but worth surfacing rather than silently overriding).
        #
        # These are SELLER-SIDE prices: what the network should quote for the slot.
        occ_rate = occ_result.avg_occupancy_rate
        if cap_price <= floor_price:
            recommended_price = floor_price
            assumptions.append("degenerate price band (cap<=floor), using floor only")
        else:
            recommended_price = floor_price + occ_rate * (cap_price - floor_price)

        booking_probability = self.prob_model.predict_proba(
            recommended_price, screen_size, screen_type, position, city_id, industry_vertical
        )
        expected_revenue = recommended_price * booking_probability

        if cap_price > floor_price:
            price_grid = np.linspace(floor_price, cap_price, self.n_grid_points)
            probs = self.prob_model.predict_proba_curve(
                price_grid, screen_size, screen_type, position, city_id, industry_vertical
            )
            argmax_price = float(price_grid[int(np.argmax(price_grid * probs))])
            if abs(argmax_price - recommended_price) / recommended_price > 0.15:
                assumptions.append(
                    f"probability-only argmax ({argmax_price:.2f}) diverges >15% from "
                    f"occupancy-driven price ({recommended_price:.2f})"
                )

        # Per-slot-count price response.
        #
        # Price is CONSTANT across slot counts, by deliberate design, not oversight. Two
        # candidate models were tested against the data and both were rejected:
        #
        #  (a) Occupancy escalation ("each marginal slot costs more"): occupancy_rate is
        #      derived from PRE-EXISTING bookings only; slots_needed never enters the price
        #      formula. So this effect does not exist in the implementation and asserting
        #      it would be fabricating a curve.
        #  (b) Volume discount (median 1.00 -> 0.91 from 1 to 6 slots): the raw effect
        #      replicates (1.00 -> 0.926) but is largely a composition confound. Larger
        #      slot purchases skew toward cheaper inventory (L-size share 30.9% -> 26.7%;
        #      premium-city LH share 65.8% -> 61.5%). Controlling for the price-band
        #      segment, the residual effect is ~1.6% (1.003 -> 0.984) -- inside noise.
        #
        # A flat per-slot price is therefore the defensible reading. The map is returned
        # explicitly so the interface states this rather than leaving consumers to infer
        # it. If a discount curve is later justified, apply it here -- the shape is the
        # only thing that changes.
        max_avail = int(occ_result.min_available_slots)
        price_by_slot_count = {
            n: (round(float(recommended_price), 2) if n <= max_avail else None)
            for n in range(1, SLOT_CAPACITY + 1)
        }

        return ScreenPricing(
            screen_id=screen_id,
            feasible=True,
            time_block_id=time_block_id,
            floor_price=round(floor_price, 2),
            target_price=round(target_price, 2),
            cap_price=round(cap_price, 2),
            occupancy_rate=round(occ_rate, 3),
            min_free_slots=max_avail,
            max_available_slots=max_avail,
            recommended_price=round(recommended_price, 2),
            booking_probability=round(booking_probability, 4),
            expected_revenue=round(expected_revenue, 2),
            price_by_slot_count=price_by_slot_count,
            assumptions=assumptions,
        )

    def price_candidates(self, candidates: list[dict]) -> list[ScreenPricing]:
        """candidates: list of dicts, each with keys matching price_screen's args."""
        return [self.price_screen(**c) for c in candidates]
