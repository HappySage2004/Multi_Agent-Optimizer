"""M2 -- Price Band Engine.

Derives floor/target/cap prices (p25/p50/p90) from historical bookings, segmented
primarily by screen physical attributes (the strongest price signal in the data) and
city/daypart, with industry_vertical applied only as a smaller secondary adjustment --
never as a primary segmentation key.

Fallback ladder (deterministic, bounded -- no free-form retries):
  Level A: screen_size x screen_type x position x city x daypart  (n>=MIN_SAMPLES)
  Level B: screen_size x screen_type x position x city            (n>=MIN_SAMPLES)
  Level C: screen_size x screen_type x position                   (final floor)

Industry adjustment is applied on top as a bounded multiplier, only when the finer
(segment x industry) slice itself has enough samples to trust.

Port note: segmentation keys, quantiles, thresholds and the clamp are unchanged. Two
things differ, both flagged inline: the band is looked up against the *screen's own*
city_id rather than the campaign's, and Level C has an explicit fallback instead of an
unguarded `.loc`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.logging_utils import debug

MIN_SAMPLES = 30
MIN_INDUSTRY_SAMPLES = 15
INDUSTRY_ADJ_CLAMP = (0.85, 1.20)  # never let industry swing the band more than -15/+20%

_PRICE_COL = "contracted_price_per_slot_per_day"


@dataclass
class PriceBand:
    screen_id: str
    floor_price: float
    target_price: float
    cap_price: float
    segment_level_used: str  # 'full_with_daypart' | 'city_no_daypart' | 'attrs_only'
    segment_n: int
    industry_adjustment: float
    assumptions: list = field(default_factory=list)


def _quantile_agg(grouped) -> pd.DataFrame:
    return grouped.agg(
        floor=lambda x: x.quantile(0.25),
        target=lambda x: x.quantile(0.50),
        cap=lambda x: x.quantile(0.90),
        n="count",
    )


class PriceBandEngine:
    def __init__(self, bookings_df: pd.DataFrame, screens_df: pd.DataFrame):
        # position is null for metro_rail_coach (no entrance/back concept inside a train
        # car) -- treat as its own explicit category rather than crashing or silently
        # dropping rows from the segmentation.
        screens_df = screens_df.copy()
        screens_df["position"] = screens_df["position"].fillna("not_applicable")

        merged = bookings_df.merge(
            screens_df[["screen_id", "screen_type", "screen_size", "position"]],
            on="screen_id",
            how="left",
        )
        self._merged = merged
        self._screens = screens_df.set_index("screen_id")

        # Level A: full depth incl daypart, excl industry
        self._level_a = _quantile_agg(
            merged.groupby(["screen_size", "screen_type", "position", "city_id", "daypart"])[
                _PRICE_COL
            ]
        )

        # Level B: drop daypart
        self._level_b = _quantile_agg(
            merged.groupby(["screen_size", "screen_type", "position", "city_id"])[_PRICE_COL]
        )

        # Level C: screen attributes only (final safety floor)
        self._level_c = _quantile_agg(
            merged.groupby(["screen_size", "screen_type", "position"])[_PRICE_COL]
        )

        # Industry adjustment: ratio of (segment x industry) mean to segment mean, computed
        # at the city-level segment (Level B keys), only kept where the industry slice
        # itself has enough samples.
        seg_mean = merged.groupby(["screen_size", "screen_type", "position", "city_id"])[
            _PRICE_COL
        ].mean()
        seg_industry = merged.groupby(
            ["screen_size", "screen_type", "position", "city_id", "industry_vertical"]
        )[_PRICE_COL].agg(mean="mean", n="count")

        ratios = (
            seg_industry["mean"]
            / seg_mean.reindex(seg_industry.index.droplevel("industry_vertical")).values
        )
        ratios = ratios.clip(*INDUSTRY_ADJ_CLAMP)
        neutralized = int((seg_industry["n"] < MIN_INDUSTRY_SAMPLES).sum())
        ratios[seg_industry["n"] < MIN_INDUSTRY_SAMPLES] = 1.0
        self._industry_adj = ratios  # indexed same as seg_industry

        debug(
            f"price band: ladder built from {len(merged):,} bookings — "
            f"level A (incl daypart) {len(self._level_a):,} segments, "
            f"level B (excl daypart) {len(self._level_b):,}, "
            f"level C (attrs only) {len(self._level_c):,}; "
            f"min_samples={MIN_SAMPLES}"
        )
        debug(
            f"price band: {len(ratios) - neutralized:,} of {len(ratios):,} industry "
            f"adjustments kept (>= {MIN_INDUSTRY_SAMPLES} samples), {neutralized:,} "
            f"neutralized to 1.0, clamp={INDUSTRY_ADJ_CLAMP}"
        )

    def _screen_attrs(self, screen_id: str):
        row = self._screens.loc[screen_id]
        return row["screen_size"], row["screen_type"], row["position"], row.get("city_id")

    def get_price_band(
        self,
        screen_id: str,
        daypart: str,
        industry_vertical: str,
        city_id: str | None = None,
    ) -> PriceBand:
        """Band for one screen.

        `city_id` defaults to the screen's own city. Upstream took the *campaign's*
        city_id, which is wrong whenever a campaign spans more than one city (our
        CampaignSpec.city_ids is a list): the lookup would miss at Levels A and B and
        quietly degrade to the attribute-only band, or land on another city's prices.
        """
        size, stype, pos, screen_city = self._screen_attrs(screen_id)
        city_id = city_id or screen_city
        assumptions: list[str] = []

        # Level A attempt
        key_a = (size, stype, pos, city_id, daypart)
        if key_a in self._level_a.index and self._level_a.loc[key_a, "n"] >= MIN_SAMPLES:
            row = self._level_a.loc[key_a]
            level_used = "full_with_daypart"
        else:
            # Level B attempt
            key_b = (size, stype, pos, city_id)
            if key_b in self._level_b.index and self._level_b.loc[key_b, "n"] >= MIN_SAMPLES:
                row = self._level_b.loc[key_b]
                level_used = "city_no_daypart"
                assumptions.append("daypart dropped: insufficient sample at full depth")
            else:
                # Level C: final floor. All 15 (size, type, position) combinations present
                # in `screens` do appear in `bookings`, so this is populated in practice --
                # but an unguarded `.loc` would take down the whole run if that ever
                # changed, so the miss is reported instead.
                key_c = (size, stype, pos)
                if key_c not in self._level_c.index:
                    raise KeyError(
                        f"No historical pricing for screen attributes {key_c} "
                        f"(screen_id={screen_id}); cannot derive a price band."
                    )
                row = self._level_c.loc[key_c]
                level_used = "attrs_only"
                assumptions.append(
                    "city+daypart dropped: insufficient sample, using screen-attribute-only band"
                )

        # Industry adjustment lookup (independent of which level was used for the band)
        industry_key = (size, stype, pos, city_id, industry_vertical)
        adj = 1.0
        if industry_key in self._industry_adj.index:
            adj = float(self._industry_adj.loc[industry_key])
            if adj != 1.0:
                assumptions.append(f"industry adjustment applied: x{adj:.2f}")

        return PriceBand(
            screen_id=screen_id,
            floor_price=round(row["floor"] * adj, 2),
            target_price=round(row["target"] * adj, 2),
            cap_price=round(row["cap"] * adj, 2),
            segment_level_used=level_used,
            segment_n=int(row["n"]),
            industry_adjustment=adj,
            assumptions=assumptions,
        )

    def get_price_band_batch(
        self, screen_ids, daypart, industry_vertical, city_id: str | None = None
    ) -> pd.DataFrame:
        rows = [self.get_price_band(sid, daypart, industry_vertical, city_id) for sid in screen_ids]
        return pd.DataFrame(
            [
                {
                    "screen_id": r.screen_id,
                    "floor_price": r.floor_price,
                    "target_price": r.target_price,
                    "cap_price": r.cap_price,
                    "segment_level_used": r.segment_level_used,
                    "segment_n": r.segment_n,
                    "industry_adjustment": r.industry_adjustment,
                }
                for r in rows
            ]
        )
