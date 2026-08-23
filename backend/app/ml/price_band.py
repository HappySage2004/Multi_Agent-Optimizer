"""M2 -- Price Band Engine.

Derives floor/target/cap prices (p25/p50/p90) from historical bookings, segmented
primarily by screen physical attributes (the strongest price signal in the data) and
location/daypart, with industry_vertical applied only as a smaller secondary adjustment --
never as a primary segmentation key.

Fallback ladder (deterministic, bounded -- no free-form retries). Each location rung is
tried twice: split by DEAL SHAPE (`is_bundle`) first, then blended.
  Level Z1: screen_size x screen_type x position x ZONE x daypart (n>=MIN_SAMPLES)
  Level Z2: screen_size x screen_type x position x ZONE           (n>=MIN_SAMPLES)
  Level A:  screen_size x screen_type x position x city x daypart (n>=MIN_SAMPLES)
  Level B:  screen_size x screen_type x position x city           (n>=MIN_SAMPLES)
  Level C:  screen_size x screen_type x position                  (final floor, no split)

WHY ZONE SITS ABOVE CITY (measured, not assumed). Holding city, size, type and position
fixed, the median contracted price still varies 1.87x-2.52x ACROSS ZONES of the same city:
DAT S metro_station platform runs from a 33.47 zone median to an 84.30 one over 10 zones
and 6,260 bookings. Segmenting on city alone quotes every one of those from the same
blended band, which undersells the strong zones and oversells the weak ones. Coverage
supports the depth: 95.6% of bookings sit in a Z1 cell with n>=30, and 99.7% in a Z2 cell.

Zone is NULL for all 2,615 vehicle-mounted screens, so Z1/Z2 simply never match for them
and mobile inventory falls through to the city levels. That is why there is no branch on
inventory class here.

WHY DEAL SHAPE IS A DIMENSION, and a measurement worth not misreading. `is_bundle=False`
deals hold exactly one screen (max 1); bundled deals hold a median of 20. A package this
system produces is therefore commercially a bundle, a single-screen quote is not, and they
should not come off the same comparables.

How big the effect is depends on how it is measured, and the two readings disagree:

* By MEAN PRICE INDEX at city+daypart grain the split looks inert: 1.0617 non-bundle
  against 1.0459 bundle, medians 0.9939 and 1.0017 on 55,485 and 135,624 rows.
* By the ACTUAL BAND QUANTILES at zone+daypart grain -- the cells this ladder now resolves
  at -- across the 302 cells where both shapes clear n>=30, single-screen deals sit
  consistently higher: floor x1.090, target x1.079, cap x1.065, on a band 7.5% narrower.

The second reading is the one that governs here, for two reasons. The price formula consumes
QUANTILES (`floor + position x (cap - floor)`), and a mean of per-booking ratios against a
blended median is a different statistic from the quantiles of each population. And zone
grain is where the band actually resolves. Coverage survives the split: 89.5% of bookings
sit in a Z1+shape cell with n>=30, 98.9% at Z2+shape.

Deal shape is nonetheless the FIRST dimension surrendered when a cell runs short, because
6.5-9% is worth less than zone's 87-152%.

ONE DIMENSION TESTED AND REJECTED, so nobody re-adds it on intuition:

* `duration_days`. Real but small once bundle is controlled, and only inside the bundle
  population: 1.0577 (<=14d) -> 1.0648 (15-30) -> 1.0425 (31-60) -> 0.9924 (61-120) ->
  0.9307 (>120). Between the 15-30 and 31-60 buckets that most campaigns fall in, the
  effect is ~2%. The non-bundle path is non-monotone and noisy (1.3988 and 1.4100 on thin
  cells). Not enough signal to justify another dimension.

Port note: quantiles, thresholds and the clamp are unchanged. The zone levels are new. Two
other differences from upstream, both flagged inline: the band is looked up against the
*screen's own* city_id rather than the campaign's, and Level C has an explicit fallback
instead of an unguarded `.loc`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.logging_utils import debug

MIN_SAMPLES = 30
MIN_INDUSTRY_SAMPLES = 15
INDUSTRY_ADJ_CLAMP = (0.85, 1.20)  # never let industry swing the band more than -15/+20%

_PRICE_COL = "contracted_price_per_slot_per_day"
_BUNDLE_COL = "is_bundle"

# The ladder, finest first. Ordering is the whole design: the dimension worth least is the
# one surrendered first when a cell runs short of samples.
_RUNG_KEYS: dict[str, tuple[str, ...]] = {
    "zone_with_daypart": ("screen_size", "screen_type", "position", "zone_id", "daypart"),
    "zone_no_daypart": ("screen_size", "screen_type", "position", "zone_id"),
    "full_with_daypart": ("screen_size", "screen_type", "position", "city_id", "daypart"),
    "city_no_daypart": ("screen_size", "screen_type", "position", "city_id"),
}

# Why a rung fell through, in the order the ladder tries them. Recorded per row so a quoted
# price is traceable to the sample it came from.
_RUNG_NOTES: dict[str, str] = {
    "zone_no_daypart": "daypart dropped: insufficient sample at zone+daypart depth",
    "full_with_daypart": "zone dropped: insufficient sample, using city+daypart band",
    "city_no_daypart": "zone+daypart dropped: insufficient sample, using city band",
}


@dataclass
class PriceBand:
    screen_id: str
    floor_price: float
    target_price: float
    cap_price: float
    segment_level_used: str
    """The ladder rung that actually resolved, finest first: 'zone_with_daypart',
    'zone_no_daypart', 'full_with_daypart', 'city_no_daypart', 'attrs_only' -- each
    optionally suffixed '_bundle' or '_single_screen' when the deal-shape split held
    enough samples to use."""
    segment_n: int
    industry_adjustment: float
    zone_id: str | None = None
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

        # `zone_id` comes off the SCREEN, not the booking: a booking's own city_id is
        # denormalized but it carries no zone, and the screen is what physically sits in
        # the zone. NULL for mobile inventory, which is what makes the zone levels
        # self-skipping rather than needing an inventory-class branch.
        merged = bookings_df.merge(
            screens_df[["screen_id", "screen_type", "screen_size", "position", "zone_id"]],
            on="screen_id",
            how="left",
        )
        self._merged = merged
        self._screens = screens_df.set_index("screen_id")

        # Each rung is built TWICE: once split by deal shape (`is_bundle`) and once
        # blended. The lookup prefers the split cell and falls back to the blended one, so
        # bundle is always the first dimension surrendered when samples run short — it is
        # worth 6.5-9%, against zone's 87-152%.
        zoned = merged[merged["zone_id"].notna()]
        self._tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for name, keys in _RUNG_KEYS.items():
            source = zoned if "zone_id" in keys else merged
            self._tables[name] = (
                _quantile_agg(source.groupby([*keys, _BUNDLE_COL])[_PRICE_COL]),
                _quantile_agg(source.groupby(list(keys))[_PRICE_COL]),
            )

        # Level C: screen attributes only (final safety floor). Kept out of `_tables`
        # because it has no fallback below it and no bundle split -- if the ladder gets
        # this far, the sample is already too thin to divide again.
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

        rungs = ", ".join(
            f"{name} {len(blended):,}/{len(split):,} split"
            for name, (split, blended) in self._tables.items()
        )
        debug(
            f"price band: ladder built from {len(merged):,} bookings "
            f"({len(zoned):,} with a zone) — {rungs}, "
            f"attrs_only {len(self._level_c):,}; min_samples={MIN_SAMPLES}"
        )
        debug(
            f"price band: {len(ratios) - neutralized:,} of {len(ratios):,} industry "
            f"adjustments kept (>= {MIN_INDUSTRY_SAMPLES} samples), {neutralized:,} "
            f"neutralized to 1.0, clamp={INDUSTRY_ADJ_CLAMP}"
        )

    def _screen_attrs(self, screen_id: str):
        row = self._screens.loc[screen_id]
        zone = row.get("zone_id")
        return (
            row["screen_size"],
            row["screen_type"],
            row["position"],
            row.get("city_id"),
            None if pd.isnull(zone) else zone,
        )

    def get_price_band(
        self,
        screen_id: str,
        daypart: str,
        industry_vertical: str,
        city_id: str | None = None,
        industry_weight: float = 1.0,
        is_bundle: bool | None = None,
    ) -> PriceBand:
        """Band for one screen.

        `city_id` defaults to the screen's own city. Upstream took the *campaign's*
        city_id, which is wrong whenever a campaign spans more than one city (our
        CampaignSpec.city_ids is a list): the lookup would miss at Levels A and B and
        quietly degrade to the attribute-only band, or land on another city's prices.

        `industry_weight` dials how much of the industry adjustment survives (see
        `app/ml/levers.py`). The EFFECTIVE adjustment is re-clamped to
        `INDUSTRY_ADJ_CLAMP` afterwards, so the -15/+20% guarantee holds at any weight —
        the lever can suppress the adjustment, never widen its authority.

        `is_bundle` selects comparables of the same DEAL SHAPE: True for a multi-screen
        package, False for a single-screen quote, None to use the blended band (the
        behaviour before the split existed). Every rung is tried split-first, then blended,
        so a thin split cell costs precision on the deal shape rather than on the zone.
        """
        size, stype, pos, screen_city, zone_id = self._screen_attrs(screen_id)
        city_id = city_id or screen_city
        assumptions: list[str] = []

        # Walk the ladder finest-first. A zone rung with no zone (mobile inventory) is
        # skipped by its own missing key rather than by a branch on inventory class.
        # Walking an ordered dict rather than nesting `if/else` is what keeps adding a rung
        # a one-line change instead of another level of indentation.
        available = {
            "screen_size": size,
            "screen_type": stype,
            "position": pos,
            "city_id": city_id,
            "zone_id": zone_id,
            "daypart": daypart,
        }

        row = None
        level_used = ""
        for name, keys in _RUNG_KEYS.items():
            if any(available[k] is None for k in keys):
                continue  # no zone: this rung does not exist for this screen
            split_table, blended_table = self._tables[name]
            base_key = tuple(available[k] for k in keys)

            # Deal shape first: same rung, comparables of the same shape.
            if is_bundle is not None:
                split_key = (*base_key, is_bundle)
                if (
                    split_key in split_table.index
                    and split_table.loc[split_key, "n"] >= MIN_SAMPLES
                ):
                    row = split_table.loc[split_key]
                    level_used = f"{name}_{'bundle' if is_bundle else 'single_screen'}"
                    if name in _RUNG_NOTES:
                        assumptions.append(_RUNG_NOTES[name])
                    break

            if base_key in blended_table.index and blended_table.loc[base_key, "n"] >= MIN_SAMPLES:
                row = blended_table.loc[base_key]
                level_used = name
                if name in _RUNG_NOTES:
                    assumptions.append(_RUNG_NOTES[name])
                if is_bundle is not None:
                    assumptions.append(
                        f"deal shape dropped: too few "
                        f"{'bundled' if is_bundle else 'single-screen'} comparables at this "
                        f"depth, using the blended band"
                    )
                break

        if row is None:
            # Level C: final floor. All 15 (size, type, position) combinations present in
            # `screens` do appear in `bookings`, so this is populated in practice -- but an
            # unguarded `.loc` would take down the whole run if that ever changed, so the
            # miss is reported instead.
            key_c = (size, stype, pos)
            if key_c not in self._level_c.index:
                raise KeyError(
                    f"No historical pricing for screen attributes {key_c} "
                    f"(screen_id={screen_id}); cannot derive a price band."
                )
            row = self._level_c.loc[key_c]
            level_used = "attrs_only"
            assumptions.append(
                "zone+city+daypart dropped: insufficient sample, using screen-attribute-only band"
            )

        # Industry adjustment lookup. Deliberately still keyed at CITY level, independent
        # of which rung the band came from: this is an industry effect, not a location one,
        # and a zone x industry slice would rarely clear MIN_INDUSTRY_SAMPLES.
        industry_key = (size, stype, pos, city_id, industry_vertical)
        adj = 1.0
        if industry_key in self._industry_adj.index:
            raw_adj = float(self._industry_adj.loc[industry_key])
            # Weight the DEVIATION, then re-clamp. Re-clamping is what keeps the module's
            # stated guarantee ("never more than -15/+20%") true for a weight above 1.0.
            weighted = 1.0 + industry_weight * (raw_adj - 1.0)
            adj = min(max(weighted, INDUSTRY_ADJ_CLAMP[0]), INDUSTRY_ADJ_CLAMP[1])
            if adj != 1.0:
                if industry_weight != 1.0:
                    assumptions.append(
                        f"industry adjustment applied: x{adj:.2f} "
                        f"(raw x{raw_adj:.2f} at weight {industry_weight:g})"
                    )
                else:
                    assumptions.append(f"industry adjustment applied: x{adj:.2f}")
            elif raw_adj != 1.0:
                assumptions.append(
                    f"industry adjustment x{raw_adj:.2f} suppressed by "
                    f"industry_weight={industry_weight:g}"
                )

        return PriceBand(
            screen_id=screen_id,
            floor_price=round(row["floor"] * adj, 2),
            target_price=round(row["target"] * adj, 2),
            cap_price=round(row["cap"] * adj, 2),
            segment_level_used=level_used,
            segment_n=int(row["n"]),
            industry_adjustment=adj,
            zone_id=zone_id,
            assumptions=assumptions,
        )

    def get_price_band_batch(
        self,
        screen_ids,
        daypart,
        industry_vertical,
        city_id: str | None = None,
        industry_weight: float = 1.0,
        is_bundle: bool | None = None,
    ) -> pd.DataFrame:
        rows = [
            self.get_price_band(
                sid, daypart, industry_vertical, city_id, industry_weight, is_bundle
            )
            for sid in screen_ids
        ]
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
