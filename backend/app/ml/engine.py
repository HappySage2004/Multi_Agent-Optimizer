"""The pricing engine -- assembly and the per-query entry point.

This is the port of the upstream `PricingAgent` class. Two things changed:

* Construction reads through the DuckDB view layer (`app/ml/loaders.py`) instead of
  `pd.read_csv` on nine dataset files.
* `daypart` is derived from `dim_slot` rather than accepted as a free parameter, and the
  price band is looked up against each screen's own `city_id`. Both fix ways the upstream
  signature let a caller silently mis-segment the band.

Everything else -- the module wiring, the per-screen decision sequence, and the output
shape -- is unchanged.

Lifecycle: `get_pricing_engine()` is a process-wide singleton. `build()` costs ~8 s
(bookings projection, occupancy index, price-band groupbys, ridership aggregates, and the
logistic fit), so it must be paid once at first use, never per request.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd

from app.logging_utils import debug, info
from app.ml import loaders
from app.ml.booking_probability import BookingProbabilityModel
from app.ml.impressions import ImpressionsEstimator
from app.ml.occupancy import OccupancyEngine
from app.ml.price_band import PriceBandEngine
from app.ml.price_optimizer import PriceOptimizer, ScreenPricing
from app.ml.seasonality import SeasonalityAdjuster

# Every key the per-row output carries, so an unknown screen still yields the full schema.
_NULL_ROW_FIELDS = (
    "floor_price",
    "target_price",
    "cap_price",
    "occupancy_rate",
    "min_free_slots",
    "max_available_slots",
    "recommended_price",
    "booking_probability",
    "expected_revenue",
    "seasonality_multiplier",
    "event_match_type",
    "pricing_internal_reach_proxy",
    "pricing_internal_reach_method",
)


class PricingEngine:
    def __init__(
        self,
        occ_engine: OccupancyEngine,
        band_engine: PriceBandEngine,
        prob_model: BookingProbabilityModel,
        optimizer: PriceOptimizer,
        impressions_estimator: ImpressionsEstimator,
        seasonality_adjuster: SeasonalityAdjuster,
        screens_df: pd.DataFrame,
        dayparts: dict[str, str],
    ):
        self.occ_engine = occ_engine
        self.band_engine = band_engine
        self.prob_model = prob_model
        self.optimizer = optimizer
        self.impressions_estimator = impressions_estimator
        self.seasonality_adjuster = seasonality_adjuster
        self._screens = screens_df.set_index("screen_id")
        self._dayparts = dayparts

    @classmethod
    def build(cls) -> PricingEngine:
        """One-time assembly. Call through `get_pricing_engine()`, not directly."""
        t0 = time.perf_counter()

        bookings = loaders.load_bookings()
        screens = loaders.load_screens()
        lost_leads = loaders.load_lost_leads()
        debug(f"pricing engine: loaded {len(bookings):,} bookings, {len(screens):,} screens")

        occ_engine = OccupancyEngine(bookings)
        band_engine = PriceBandEngine(bookings, screens)

        prob_model = BookingProbabilityModel()
        report = prob_model.fit(bookings, lost_leads, screens)
        if not report.price_coefficient_sign_ok:
            # The upstream module names this as the one check that must pass: a positive
            # price coefficient means the confounders are not controlled and the model is
            # not fit for pricing. Surface it rather than pricing off it silently.
            info(
                "pricing engine WARNING: booking-probability price coefficient is "
                f"{report.price_coefficient:+.4f} (expected negative) — probability "
                "outputs are not trustworthy for this fit"
            )
        debug(
            f"pricing engine: booking model n_won={report.n_won:,} n_lost={report.n_lost:,} "
            f"price_coef={report.price_coefficient:+.4f} auc={report.auc:.4f} "
            f"calibration_ok={report.calibration_ok}"
        )

        optimizer = PriceOptimizer(occ_engine, band_engine, prob_model)

        impressions_estimator = ImpressionsEstimator(
            screens,
            loaders.load_poi_footfall_by_location(),
            loaders.load_vehicles(),
            loaders.load_corridor_ridership(),
        )

        ridership = loaders.load_ridership_seasonality()
        if ridership is None:
            info(
                "pricing engine: ridership_actuals.csv not provisioned — day-of-week / "
                "holiday multiplier degrades to 1.0"
            )
        seasonality_adjuster = SeasonalityAdjuster(
            ridership, loaders.load_events(), screens, loaders.load_locations()
        )

        engine = cls(
            occ_engine,
            band_engine,
            prob_model,
            optimizer,
            impressions_estimator,
            seasonality_adjuster,
            screens,
            loaders.time_block_dayparts(),
        )
        info(f"pricing engine ready in {time.perf_counter() - t0:.1f}s")
        return engine

    # --- introspection -------------------------------------------------------

    @property
    def training_report(self):
        return self.prob_model.training_report

    @property
    def dayparts(self) -> dict[str, str]:
        """`dim_slot` time_block_id -> nearest_daypart, for validating requested blocks."""
        return self._dayparts

    def daypart_for(self, time_block_id: str | int) -> str:
        """`dim_slot.nearest_daypart` for a time block."""
        key = str(time_block_id)
        if key not in self._dayparts:
            raise KeyError(f"Unknown time_block_id '{time_block_id}' — not in dim_slot.")
        return self._dayparts[key]

    def knows_screen(self, screen_id: str) -> bool:
        return screen_id in self._screens.index

    # --- per-query path ------------------------------------------------------

    def price_candidates(
        self, campaign: dict[str, Any], candidate_screens: list[str]
    ) -> list[dict]:
        """Price one (time block x campaign context) against a candidate screen list.

        campaign: {
          industry_vertical, time_block_id, start_date, end_date,
          slots_needed (default 1), city_id (optional -- otherwise each screen's own)
        }

        Returns one dict per candidate screen, input order preserved. Infeasible screens
        are still returned (with feasible=False) rather than silently dropped, so the
        caller can see what was excluded and why.
        """
        results: list[dict] = []
        slots_needed = campaign.get("slots_needed", 1)
        time_block_id = campaign["time_block_id"]
        daypart = campaign.get("daypart") or self.daypart_for(time_block_id)

        t0 = time.perf_counter()
        unknown = 0

        for screen_id in candidate_screens:
            if screen_id not in self._screens.index:
                unknown += 1
                results.append(self._unknown_screen_row(screen_id, time_block_id))
                continue

            row = self._screens.loc[screen_id]
            position = row["position"] if pd.notnull(row["position"]) else "not_applicable"
            # The screen's own city, not the campaign's: a multi-city campaign must not
            # price a DAT screen off LH's band.
            city_id = campaign.get("city_id") or row["city_id"]

            seasonality = self.seasonality_adjuster.get_adjustment(
                screen_id, campaign["start_date"], campaign["end_date"]
            )

            pricing: ScreenPricing = self.optimizer.price_screen(
                screen_id=screen_id,
                screen_size=row["screen_size"],
                screen_type=row["screen_type"],
                position=position,
                city_id=city_id,
                time_block_id=time_block_id,
                daypart=daypart,
                industry_vertical=campaign["industry_vertical"],
                start_date=campaign["start_date"],
                end_date=campaign["end_date"],
                slots_needed=slots_needed,
                price_multiplier=seasonality.combined_multiplier,
            )

            result = _as_row(pricing)

            # Reach is NOT owned by this engine. Its figure rests on hand-set
            # dwell/visibility multipliers with no ground truth and mismatched units
            # between fixed and mobile screens (see impressions.py). It is retained under
            # an explicitly non-client-facing name for pricing diagnostics only, and is
            # deliberately not mapped onto any exposure field.
            if pricing.feasible:
                impressions = self.impressions_estimator.estimate(screen_id)
                result["pricing_internal_reach_proxy"] = impressions.estimated_impressions
                result["pricing_internal_reach_method"] = impressions.method
                result["seasonality_multiplier"] = seasonality.combined_multiplier
                result["event_match_type"] = seasonality.event_match_type
            else:
                result["pricing_internal_reach_proxy"] = None
                result["pricing_internal_reach_method"] = None
                result["seasonality_multiplier"] = None
                result["event_match_type"] = None
            result["reach_owner"] = "audience_engine"  # not this engine

            results.append(result)

        # Aggregated deliberately: this loop runs once per candidate screen per time block
        # (~750 iterations on a default pool), so a per-screen line would bury the trace.
        feasible = [r for r in results if r["feasible"]]
        prices = [r["recommended_price"] for r in feasible]
        debug(
            f"pricing block {time_block_id} ({daypart}): {len(feasible)}/{len(results)} "
            f"feasible for {slots_needed} slot(s)/day"
            + (
                f", price {min(prices):,.2f}..{max(prices):,.2f} "
                f"(mean {sum(prices) / len(prices):,.2f})"
                if prices
                else ""
            )
            + f", {time.perf_counter() - t0:.2f}s"
        )
        if unknown:
            # A screen the audience engine ranked but inventory does not know about — a
            # seam failure between stages, not a data property.
            info(
                f"pricing block {time_block_id}: {unknown} candidate screen(s) absent "
                f"from the screens table, returned as infeasible rather than dropped"
            )

        return results

    def _unknown_screen_row(self, screen_id: str, time_block_id) -> dict:
        """Full schema for a screen absent from inventory -- consumers rely on every key
        being present on every row (contract invariant)."""
        row: dict[str, Any] = {
            "screen_id": screen_id,
            "feasible": False,
            "time_block_id": time_block_id,
            "price_by_slot_count": {n: None for n in range(1, 7)},
            "reach_owner": "audience_engine",
            "assumptions": ["screen_id not found in inventory"],
        }
        for name in _NULL_ROW_FIELDS:
            row.setdefault(name, None)
        return row


def _as_row(pricing: ScreenPricing) -> dict:
    """ScreenPricing -> plain dict. Hand-rolled rather than `dataclasses.asdict` so the
    `price_by_slot_count` int keys survive untouched."""
    return {
        "screen_id": pricing.screen_id,
        "feasible": pricing.feasible,
        "time_block_id": pricing.time_block_id,
        "floor_price": pricing.floor_price,
        "target_price": pricing.target_price,
        "cap_price": pricing.cap_price,
        "occupancy_rate": pricing.occupancy_rate,
        "min_free_slots": pricing.min_free_slots,
        "max_available_slots": pricing.max_available_slots,
        "recommended_price": pricing.recommended_price,
        "booking_probability": pricing.booking_probability,
        "expected_revenue": pricing.expected_revenue,
        "price_by_slot_count": pricing.price_by_slot_count,
        "assumptions": list(pricing.assumptions),
    }


_engine: PricingEngine | None = None
_lock = threading.Lock()


def get_pricing_engine() -> PricingEngine:
    """Process-wide singleton. `build()` is ~8 s; never call it per request."""
    global _engine
    with _lock:
        if _engine is None:
            _engine = PricingEngine.build()
        return _engine


def reset_pricing_engine() -> None:
    """Drop the singleton. For tests only."""
    global _engine
    with _lock:
        _engine = None
