"""M7 -- Screen Demand Value, and the mispricing premium it produces.

THE PROBLEM THIS EXISTS FOR. Every other price signal in `app/ml/` answers one question:
what did screens like this one sell for before? That makes the engine a mirror. A screen
that has always been sold cheap is quoted cheap forever, and no amount of history can tell
you the history was wrong.

THE APPROACH. Build a second opinion that NEVER LOOKS AT PRICE. `merit` scores a screen on
what it physically delivers -- riders passing, zone affluence, daytime activity, nearby POI
draw -- and nothing else. Then compare merit against what the screen has actually
transacted at. A screen with high merit and a low realized price is underpriced.

WHY THE MODEL MUST NOT SEE PRICE, stated plainly because the obvious alternative fails
quietly. Fitting a model to PREDICT price from location and audience, then flagging screens
below prediction, sounds equivalent and is not: that model's notion of "correct" IS the
historical average, so a whole category that is systematically underpriced is learned as
correct and reports no mispricing at all. It can find deviation from a norm, never a wrong
norm. Measured on this inventory, bus stops in the top audience quintile carry 6.2x the
riders of the bottom quintile and 1.59x the price -- a price-trained model calls that
correct; a price-blind merit score calls it a 4x gap between value and cost.

WHAT THE NUMBERS SAY (fixed inventory with >= MIN_PRICE_HISTORY bookings, 6,690 screens):

    residual quintile     merit  price rank   avg riders  price index
    1 (most overpriced)   0.418       0.824      158,864        1.292
    3                     0.501       0.535      183,386        0.993
    5 (most underpriced)  0.526       0.235      207,702        0.809

Riders rise monotonically with the residual while the price index falls: the most
underpriced quintile carries 31% MORE audience than the most overpriced one and transacts
at 0.81x its comparables against 1.29x. Merit correlates with realized price rank at 0.52
(bus_stop) and 0.47 (metro_station) -- positive, so merit is not noise; far below 1.0, so
there is real headroom.

THE CORROBORATION GATE, and why it is not optional. A high-merit screen selling cheap has
two possible explanations and only one is an opportunity: the market undervalues it, or
nobody wants it and the merit score is wrong. Absorption -- does this screen actually get
booked -- separates them. It withholds 281 screens averaging 283,334 riders a day, the
HIGHEST of any bucket this model produces, on an absorption rank of 0.166. Those are exactly
the screens a naive version would raise prices on, and they already cannot find a buyer.

MOBILE INVENTORY IS EXCLUDED, on evidence rather than caution. Correlation between merit and
absorption is +0.37 for bus_stop and +0.31 for metro_station, but -0.28 for bus and -0.20
for metro_rail_coach: higher modelled audience, LESS booked. That inversion is a symptom of
the known artifact in the mobile volume figure (corridor ridership divided by vehicle count,
a stated judgement in `app/data/db.py`), not a market fact. Mobile screens also have no zone
demographics at all, so three of the four merit components are structurally zero for them --
which is why their merit/price-rank correlation is 0.02 and 0.14 rather than the ~0.5 the
fixed types show. Pricing off that would be pricing off a gap in the data.

WHAT THIS CANNOT DO. There is no ground truth for what a screen is really worth, so merit
has no held-out accuracy metric and never will from this dataset -- the same gap SOLUTION.md
section 7 records for the audience model. The honest validation is forward-looking: track
whether premium-flagged screens keep their occupancy after a price rise. Until then this
ships as a disclosed recommendation, capped, gated, and reported per row.

The premium is bounded at +MAX_PREMIUM and self-correcting. If a raised price stops a screen
selling, its occupancy falls, and `floor + occupancy x (cap - floor)` pulls the quote back
down on the next run without anyone intervening.

Where the SQL lives: here, with the model, rather than in `loaders.py`. The percentile
weights and the gates ARE the model -- splitting them out would put half of it in a module
whose job is plain frame projection. Same deliberate co-location as
`app/tools/relevance_tools.py`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pandas as pd

from app.data.db import query_df
from app.logging_utils import debug, info

# --- merit weights -----------------------------------------------------------
# A transparent weighted sum, not a fitted model, for the reason in the docstring: there is
# no price-free target to fit against, and a price-fitted one would relearn the mispricing.
# Volume dominates because volume is the thing being sold; the other three describe the
# quality of that audience rather than its size.
W_VOLUME = 0.50
W_INCOME = 0.20
W_DAYTIME = 0.15
W_POI = 0.15

# --- gates -------------------------------------------------------------------

# Screen types whose audience figure is trustworthy enough to price off. See the docstring:
# mobile inventory's volume/absorption correlation is NEGATIVE.
PREMIUM_ELIGIBLE_CLASSES = frozenset({"fixed"})

# Below this, the screen's realized price index is an average of too few bookings to call
# it "what this screen sells for". 8,769 of 11,163 screens clear it.
MIN_PRICE_HISTORY = 10

# Absorption percentile the screen must clear for "underpriced" to beat "unwanted".
MIN_ABSORPTION_RANK = 0.30

# Merit minus price rank. Below this the two agree closely enough that any gap is noise.
MIN_RESIDUAL = 0.20

# The screen must ALSO be good, not merely cheaper than it is good.
#
# The residual on its own is economically coherent -- "priced below what it delivers" -- but
# it is not the claim this model is making. Measured: of 576 screens clearing the residual
# and absorption gates, 191 sit BELOW the median merit for their own screen_type x city.
# Those are weak screens priced weakly, which is the market being right. Requiring merit
# above the median leaves 385 screens averaging 0.632 merit, a +0.352 residual and a 0.835
# price index on 212,903 riders a day -- above-median inventory selling at a 17% discount to
# its own comparables, which is the actual finding.
#
# Raising it further trades volume for nothing: the surviving set's price index climbs toward
# 1.0 (0.864 at 0.60, 0.913 at 0.70), i.e. the screens still qualifying are the ones already
# priced near their comparables. 0.50 keeps the discount deepest.
MIN_MERIT = 0.50

# Residual at which the premium reaches its cap. Between MIN_RESIDUAL and this it ramps
# linearly, so there is no cliff at the gate: a screen that just clears 0.20 gets +0%.
FULL_PREMIUM_RESIDUAL = 0.60

# The cap. Deliberately modest against the size of the measured gap -- at the observed mean
# premium of x1.056 the flagged set moves from an average transacted 64.51 to 68.14, still
# well BELOW the 88.31 average of the screens this model leaves alone, and far below the
# 118.23 of the ones it flags as overpriced. The correction is conservative by construction:
# it narrows a gap rather than closing it.
#
# The commercial risk is low and measured: in `lost_leads`, deals lost to price wanted a
# third off (avg price_gap_pct 0.328 for price_too_high, 0.309 for budget_mismatch), while
# deals lost to a competitor show a gap of 0.025. Nothing in this data was lost over 15%.
MAX_PREMIUM = 0.15

# One row per screen: merit and its inputs, the realized price index, absorption, and the
# percentile ranks that make them comparable. Ranks are computed WITHIN
# (screen_type, city_id) because the realized price index is normalized against a segment
# that already contains both -- ranking globally would reintroduce exactly the pooling
# artifact this model exists to see past (metro_station median volume is ~380x bus, so a
# global volume rank is mostly a screen-type indicator).
DEMAND_VALUE_SQL = f"""
WITH vol AS (
    SELECT screen_id, sum(daily_impressions) AS daily_vol
    FROM v_screen_demand_history
    GROUP BY 1
),
absorbed AS (
    SELECT screen_id, sum(slots_booked_per_day * duration_days) AS slot_days_sold
    FROM bookings
    GROUP BY 1
),
seg AS (
    SELECT bk.screen_id,
           bk.contracted_price_per_slot_per_day AS price,
           s.screen_size, s.screen_type, coalesce(s.position, 'na') AS position,
           bk.city_id, bk.daypart
    FROM bookings bk
    JOIN screens s USING (screen_id)
),
med AS (
    SELECT screen_size, screen_type, position, city_id, daypart, median(price) AS m
    FROM seg GROUP BY 1, 2, 3, 4, 5
),
price_idx AS (
    -- What this screen actually transacts at, relative to its OWN comparables. Dividing by
    -- the segment median first is what makes a $200 metro screen and a $40 bus stop
    -- comparable on the same 1.0-centred scale.
    SELECT screen_id, count(*) AS price_n, avg(seg.price / med.m) AS price_index
    FROM seg JOIN med USING (screen_size, screen_type, position, city_id, daypart)
    GROUP BY 1
),
base AS (
    SELECT p.screen_id, p.screen_type, p.city_id, p.inventory_class,
           coalesce(v.daily_vol, 0)                     AS daily_vol,
           coalesce(p.income_index, 0)                  AS income_index,
           coalesce(p.daytime_population_multiplier, 0) AS daytime_multiplier,
           coalesce(p.weighted_nearby_footfall, 0)      AS poi_footfall,
           coalesce(a.slot_days_sold, 0)                AS slot_days_sold,
           pi.price_index,
           coalesce(pi.price_n, 0)                      AS price_n
    FROM v_screen_profile p
    LEFT JOIN vol v      USING (screen_id)
    LEFT JOIN absorbed a USING (screen_id)
    LEFT JOIN price_idx pi ON pi.screen_id = p.screen_id
),
ranked AS (
    SELECT *,
        percent_rank() OVER (w ORDER BY daily_vol)          AS pr_volume,
        percent_rank() OVER (w ORDER BY income_index)       AS pr_income,
        percent_rank() OVER (w ORDER BY daytime_multiplier) AS pr_daytime,
        percent_rank() OVER (w ORDER BY poi_footfall)       AS pr_poi,
        percent_rank() OVER (w ORDER BY slot_days_sold)     AS pr_absorption,
        -- NOTE the extra partition key. `price_index` is NULL for every screen with no
        -- booking history, and ranking those in the SAME partition puts them at one end
        -- and squeezes the priced screens into a fraction of the 0-1 range: measured on
        -- bus_stop/ACS, 181 priced screens among 702 ranked 0.000-0.257 instead of
        -- 0.000-1.000. Every price rank came out depressed, so every residual came out
        -- inflated, and the model called screens underpriced that sell ABOVE their own
        -- comparables. Partitioning on the null-ness ranks the priced screens among
        -- themselves; the null group's own ranks are discarded by the CASE.
        CASE WHEN price_index IS NULL THEN NULL
             ELSE percent_rank() OVER (w_priced ORDER BY price_index)
        END AS pr_price
    FROM base
    WINDOW w        AS (PARTITION BY screen_type, city_id),
           w_priced AS (PARTITION BY screen_type, city_id, (price_index IS NULL))
)
SELECT *,
       {W_VOLUME} * pr_volume + {W_INCOME} * pr_income
     + {W_DAYTIME} * pr_daytime + {W_POI} * pr_poi AS merit
FROM ranked
"""


@dataclass(frozen=True)
class ScreenDemandValue:
    """One screen's merit, its realized price position, and what follows from the gap."""

    screen_id: str
    merit: float
    price_index: float | None
    price_rank: float | None
    absorption_rank: float
    residual: float | None
    premium: float  # multiplier, 1.0 when no premium applies
    reason: str

    @property
    def has_premium(self) -> bool:
        return self.premium > 1.0


_NO_VALUE = ScreenDemandValue(
    screen_id="",
    merit=0.0,
    price_index=None,
    price_rank=None,
    absorption_rank=0.0,
    residual=None,
    premium=1.0,
    reason="screen not in the demand-value model",
)


class DemandValueModel:
    """Per-screen merit, mispricing residual and the resulting capped premium."""

    def __init__(self, frame: pd.DataFrame):
        self._values: dict[str, ScreenDemandValue] = {
            row.screen_id: _evaluate(row) for row in frame.itertuples(index=False)
        }
        premiums = [v for v in self._values.values() if v.has_premium]
        self._n_premium = len(premiums)
        self._mean_premium = sum(v.premium for v in premiums) / len(premiums) if premiums else 1.0
        self._n_withheld = sum(1 for v in self._values.values() if "does not sell" in v.reason)

    @classmethod
    def build(cls) -> DemandValueModel:
        frame = query_df(DEMAND_VALUE_SQL)
        model = cls(frame)
        debug(
            f"demand value: scored {len(frame):,} screens, "
            f"{model.n_premium:,} carry a premium (mean x{model.mean_premium:.3f}, "
            f"cap x{1 + MAX_PREMIUM:.2f}), {model._n_withheld:,} withheld as high-merit "
            f"but not selling"
        )
        return model

    @property
    def n_premium(self) -> int:
        return self._n_premium

    @property
    def mean_premium(self) -> float:
        return self._mean_premium

    @property
    def screens(self) -> int:
        return len(self._values)

    def for_screen(self, screen_id: str) -> ScreenDemandValue:
        """Never raises. An unknown screen gets a neutral premium and says why."""
        return self._values.get(screen_id, _NO_VALUE)

    def describe(self) -> dict:
        """Model card, for `describe_pricing_model`."""
        return {
            "form": (
                "transparent weighted percentile index over audience volume, zone income, "
                "daytime population multiplier and nearby POI footfall — never trained on "
                "price, so it can detect a systematically underpriced category rather than "
                "only deviations from one"
            ),
            "weights": {
                "audience_volume": W_VOLUME,
                "zone_income": W_INCOME,
                "daytime_activity": W_DAYTIME,
                "poi_draw": W_POI,
            },
            "ranked_within": "screen_type x city, matching the price index's own segment",
            "screens_scored": self.screens,
            "screens_with_premium": self.n_premium,
            "mean_premium_applied": round(self.mean_premium, 4),
            "max_premium": 1.0 + MAX_PREMIUM,
            "gates": {
                "inventory_class": sorted(PREMIUM_ELIGIBLE_CLASSES),
                "min_bookings_for_a_price_index": MIN_PRICE_HISTORY,
                "min_absorption_rank": MIN_ABSORPTION_RANK,
                "min_merit": MIN_MERIT,
                "min_residual": MIN_RESIDUAL,
            },
            "excluded": (
                "Mobile inventory. Modelled audience correlates NEGATIVELY with slot-days "
                "sold for bus (-0.25) and metro_rail_coach (-0.21), a symptom of the "
                "corridor-divided-by-vehicles volume artifact, and mobile screens have no "
                "zone demographics so three of four merit components are structurally zero."
            ),
            "accuracy_metric": (
                "none — there is no ground truth for what a screen is worth. Validate "
                "forward: do premium-flagged screens keep their occupancy after the rise?"
            ),
        }


def _evaluate(row) -> ScreenDemandValue:
    """Apply the gates in order, and record which one stopped a premium.

    Order matters for the message, not the outcome: the reason a screen carries no premium
    is the most useful thing this returns for anything other than the price itself.
    """
    merit = float(row.merit)
    absorption = float(row.pr_absorption)
    price_index = None if pd.isnull(row.price_index) else float(row.price_index)
    price_rank = None if pd.isnull(row.pr_price) else float(row.pr_price)

    def no_premium(reason: str, residual: float | None = None) -> ScreenDemandValue:
        return ScreenDemandValue(
            screen_id=row.screen_id,
            merit=merit,
            price_index=price_index,
            price_rank=price_rank,
            absorption_rank=absorption,
            residual=residual,
            premium=1.0,
            reason=reason,
        )

    if row.inventory_class not in PREMIUM_ELIGIBLE_CLASSES:
        return no_premium(
            f"{row.inventory_class} inventory is excluded: its audience figure is a "
            f"corridor-divided-by-vehicles estimate that correlates negatively with what "
            f"actually sells, and it carries no zone demographics"
        )
    if int(row.price_n) < MIN_PRICE_HISTORY or price_rank is None:
        return no_premium(
            f"only {int(row.price_n)} historical booking(s) on this screen "
            f"(need {MIN_PRICE_HISTORY}) — no reliable read on what it sells for"
        )

    residual = merit - price_rank

    if merit < MIN_MERIT:
        return no_premium(
            f"merit {merit:.2f} is below the median for its screen type and city "
            f"({MIN_MERIT:.2f}) — cheap, but not underpriced: a weak screen priced weakly "
            f"is the market being right",
            residual,
        )
    if residual < MIN_RESIDUAL:
        return no_premium(
            f"merit {merit:.2f} and price rank {price_rank:.2f} agree within "
            f"{MIN_RESIDUAL:.2f} — priced about right for what it delivers",
            residual,
        )
    if absorption < MIN_ABSORPTION_RANK:
        # The gate that earns its place. High merit and a low price look like an
        # opportunity right up until you notice nobody is buying it.
        return no_premium(
            f"merit {merit:.2f} is well above its price rank {price_rank:.2f}, but this "
            f"screen does not sell (absorption rank {absorption:.2f} < "
            f"{MIN_ABSORPTION_RANK:.2f}) — the market disagrees with the merit score, so "
            f"no premium is applied",
            residual,
        )

    ramp = (residual - MIN_RESIDUAL) / (FULL_PREMIUM_RESIDUAL - MIN_RESIDUAL)
    premium = 1.0 + MAX_PREMIUM * min(max(ramp, 0.0), 1.0)
    return ScreenDemandValue(
        screen_id=row.screen_id,
        merit=merit,
        price_index=price_index,
        price_rank=price_rank,
        absorption_rank=absorption,
        residual=residual,
        premium=premium,
        reason=(
            # Rank against rank, because that is the comparison the residual makes. The
            # ratio is added as context and is NOT the claim -- a screen can transact
            # slightly above its own segment median and still sit well below its peers on
            # price while delivering far more audience than they do.
            f"underpriced for its audience: merit ranks {merit:.2f} among its peers but "
            f"price only {price_rank:.2f} (it has transacted at {price_index:.2f}x its "
            f"segment median), and it sells (absorption rank {absorption:.2f})"
        ),
    )


_model: DemandValueModel | None = None
_lock = threading.Lock()


def get_demand_value_model() -> DemandValueModel:
    """Process-wide singleton, built with the pricing engine."""
    global _model
    with _lock:
        if _model is None:
            _model = DemandValueModel.build()
        return _model


def reset_demand_value_model() -> None:
    """Drop the singleton. For tests only."""
    global _model
    with _lock:
        _model = None
        info("demand value model reset")
