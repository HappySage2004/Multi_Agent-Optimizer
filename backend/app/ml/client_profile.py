"""M8 -- Client Negotiation Profile. A FLAG FOR THE REP, NOT A PRICE INPUT.

Nothing in this module ever reaches a quote. It is wired to no engine, no lever default and
no artifact; it answers one question for the salesperson holding the conversation: *how has
this client behaved on price before, and what should I expect from them this time?*

WHY IT IS ADVISORY BY CONSTRUCTION, and not merely by convention. Price-driven loss rates
are FLAT across leverage tiers -- 34.2% high, 32.5% medium, 34.8% low -- so nothing about a
client's posture tells you whether you will lose them. What varies is what they settle at.
A profile is therefore worth knowing before opening a negotiation and worthless as an
automatic adjustment: applying it would be pricing off the half of the finding that does not
hold. If a rep wants to act on it, they say so and `master_tools.set_pricing_levers` applies
a `commercial_multiplier` they chose.

THE DECLARED `negotiation_leverage` LABEL IS NOT A RELIABLE PREDICTOR, and this module is
careful not to imply otherwise. It looks predictive only when the population is weighted by
line items:

    weighting                     high     medium    low
    per line item (median)      0.9651     1.0167   1.0188   <- monotone, looks clean
    per line item (mean)        1.0128     1.0640   1.0778   <- monotone
    per CLIENT (median)         1.0394     1.0129   1.0528   <- ordering breaks
    per CLIENT (mean)           1.0513     1.0823   1.0659   <- ordering breaks

The reason is account size, not leverage: high-leverage clients carry a median 328 priced
line items against 172 (low) and 165 (medium). Weighted by volume the big accounts dominate
and they do transact better; give every client one vote and the tier ordering disappears.
Since the question this module answers is about a CLIENT rather than a line item, the
per-client figure is the honest one -- and it says the label is context, not a forecast.

So the profile leads with the two things that ARE measured per client: their own realized
price index, judged against its own standard error, and their own recorded price objections.
The tier median is reported alongside as a population reference and explicitly not as a
prediction.

WHY THE RELATIONSHIP DATA IS WORTH MINING AT ALL. 96% of clients are repeat business
(499 of 520 have more than one deal; median 36 deals, mean 109), and every one of the 807
lost leads with a known `client_id` belongs to a client who ALSO has bookings. Every
identified loss is therefore a recorded price objection from a client still on the books.

THE PRECISION PROBLEM, STATED RATHER THAN HIDDEN. Across the 433 clients with >= 30 line
items, the per-client price index runs p10 0.956, p50 1.043, p90 1.239 -- a ~30% spread
between clients. But the spread WITHIN one client averages 0.214, which is about the same
size. A client's index is therefore a real central tendency and a poor per-deal predictor,
so this module reports a standard error and downgrades its own confidence rather than
quoting a point estimate that sounds sharper than it is. `suggested_commercial_multiplier`
stays at 1.0 unless the client's history is far enough from neutral to survive its own noise.

A KNOWN LIMITATION, not currently detected: the segment medians the index is measured
against include this client's own past bookings, so a client holding a large share of one
segment is partly measured against themselves, which pulls their index toward 1.0 and makes
this model CONSERVATIVE rather than wrong. Immaterial for most of 520 clients over 191,109
bookings; it would bite on a client dominating a thin segment. Excluding each client's own
rows from the median before dividing is the fix, and it is not implemented.

Lifecycle: its own lazy singleton, deliberately NOT part of `PricingEngine.build()`. The
pricing path never needs it, and paying for it on every engine build would buy nothing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pandas as pd

from app.data.db import query_df
from app.logging_utils import debug, info
from app.ml.levers import COMMERCIAL_MULTIPLIER_RANGE

# A client needs this many priced line items before their own index is used at all. Below
# it, the leverage tier's population figure is the honest fallback.
MIN_LINE_ITEMS = 30

# How many standard errors the index must sit from 1.0 before a departure is suggested.
# 2.0 is the ordinary "unlikely to be noise" bar, and with sd ~0.214 and n=36 (the median
# client) one SE is ~0.036 -- so a client has to be ~7% off neutral to move the suggestion.
# That is deliberately hard to clear: the cost of a wrong suggestion is a rep opening a
# negotiation at the wrong number.
CONFIDENCE_SE_THRESHOLD = 2.0

# Suggestions are additionally capped tighter than the lever's own bound. The measured
# between-client spread is ~0.956-1.239, and the honest reading of a noisy mean is a nudge,
# not a repricing.
SUGGESTION_RANGE = (0.90, 1.15)

PRICE_DRIVEN_LOSS_REASONS = ("price_too_high", "budget_mismatch")

# Per-client realized price, relative to each booking's own comparables. Same two-stage
# shape as `demand_value.py`: divide by the segment median FIRST so a $200 metro line and a
# $40 bus-stop line are on one scale, then aggregate per client.
CLIENT_PRICE_SQL = """
WITH seg AS (
    SELECT bk.client_id,
           bk.deal_id,
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
idx AS (
    SELECT seg.client_id, seg.deal_id, seg.price / med.m AS ratio
    FROM seg JOIN med USING (screen_size, screen_type, position, city_id, daypart)
)
SELECT client_id,
       count(*)                     AS line_items,
       count(DISTINCT deal_id)      AS deals,
       avg(ratio)                   AS price_index,
       -- stddev_samp is NULL at n=1; coalesced downstream, never silently zeroed.
       stddev_samp(ratio)           AS price_index_sd
FROM idx
GROUP BY 1
"""

CLIENT_DATES_SQL = """
SELECT client_id,
       min(booked_date) AS first_booking,
       max(booked_date) AS last_booking
FROM bookings
GROUP BY 1
"""

# Every lost lead with a known client. `price_gap_pct` is only populated on the price-driven
# reasons, so it is averaged over those rows alone rather than over all losses.
CLIENT_LOSSES_SQL = f"""
SELECT client_id,
       count(*)                                                        AS lost_leads,
       sum(CASE WHEN loss_reason IN {PRICE_DRIVEN_LOSS_REASONS} THEN 1 ELSE 0 END)
                                                                       AS price_driven_losses,
       avg(CASE WHEN loss_reason IN {PRICE_DRIVEN_LOSS_REASONS} THEN price_gap_pct END)
                                                                       AS avg_price_gap_asked,
       max(CASE WHEN loss_reason IN {PRICE_DRIVEN_LOSS_REASONS} THEN price_gap_pct END)
                                                                       AS max_price_gap_asked,
       avg(negotiation_rounds)                                         AS avg_negotiation_rounds,
       sum(CASE WHEN competitor_mentioned THEN 1 ELSE 0 END)           AS competitor_mentions,
       avg(indicated_budget)                                           AS avg_indicated_budget
FROM lost_leads
WHERE client_id IS NOT NULL
GROUP BY 1
"""

CLIENT_FACTS_SQL = """
SELECT client_id, company_name, industry, client_tier, home_city_id,
       typical_campaign_budget, budget_variance_pct, campaign_frequency,
       avg_campaign_duration_days, bundle_affinity, negotiation_leverage,
       relationship_start_date, account_status
FROM client_facts
"""


@dataclass
class ClientNegotiationProfile:
    """One client's price behaviour, with the evidence and the uncertainty attached."""

    client_id: str
    company_name: str
    industry: str
    client_tier: str
    negotiation_leverage: str
    account_status: str
    bundle_affinity: str
    campaign_frequency: str

    # relationship
    deals: int
    line_items: int
    is_repeat: bool
    first_booking: str | None
    last_booking: str | None
    typical_campaign_budget: float | None
    budget_variance_pct: float | None
    avg_campaign_duration_days: float | None

    # what they actually pay
    realized_price_index: float | None
    price_index_sd: float | None
    price_index_standard_error: float | None
    tier_price_index: float | None  # their leverage tier's population median

    # objection history
    lost_leads: int
    price_driven_losses: int
    avg_price_gap_asked: float | None
    max_price_gap_asked: float | None
    avg_negotiation_rounds: float | None
    competitor_mentions: int

    # guidance -- advisory, never applied
    posture: str
    confidence: str  # 'strong' | 'moderate' | 'weak' | 'none'
    suggested_commercial_multiplier: float
    talking_points: list[str] = field(default_factory=list)

    def as_context(self) -> dict:
        """Shape the Master reads. Flat, JSON-safe, and it says what it is."""
        return {
            "client_id": self.client_id,
            "company_name": self.company_name,
            "industry": self.industry,
            "relationship": {
                "client_tier": self.client_tier,
                "account_status": self.account_status,
                "is_repeat_client": self.is_repeat,
                "deals": self.deals,
                "line_items": self.line_items,
                "first_booking": self.first_booking,
                "last_booking": self.last_booking,
                "campaign_frequency": self.campaign_frequency,
                "bundle_affinity": self.bundle_affinity,
                "typical_campaign_budget": self.typical_campaign_budget,
                "budget_variance_pct": self.budget_variance_pct,
                "avg_campaign_duration_days": self.avg_campaign_duration_days,
            },
            "price_behaviour": {
                "declared_negotiation_leverage": self.negotiation_leverage,
                "realized_price_index": self.realized_price_index,
                "price_index_standard_error": self.price_index_standard_error,
                "within_client_spread_sd": self.price_index_sd,
                "leverage_tier_median_index": self.tier_price_index,
                "reading": (
                    "1.0 means this client has paid exactly what comparable inventory "
                    "goes for. Below 1.0 means they habitually settle under it. The "
                    "standard error is what matters for how far to trust the figure — "
                    "within-client spread is about as wide as the between-client spread, "
                    "so this is a central tendency, not a per-deal prediction."
                ),
                "tier_caveat": (
                    "The leverage tier median is a population reference, NOT a forecast. "
                    "Per client the tier ordering does not hold — the label tracks account "
                    "size (high-leverage clients carry a median 328 line items against "
                    "165-172) rather than price behaviour. Lead with this client's own "
                    "index and their own objections."
                ),
            },
            "objection_history": {
                "lost_leads": self.lost_leads,
                "price_driven_losses": self.price_driven_losses,
                "avg_discount_asked": self.avg_price_gap_asked,
                "max_discount_asked": self.max_price_gap_asked,
                "avg_negotiation_rounds": self.avg_negotiation_rounds,
                "competitor_mentions": self.competitor_mentions,
            },
            "guidance": {
                "posture": self.posture,
                "confidence": self.confidence,
                "suggested_commercial_multiplier": self.suggested_commercial_multiplier,
                "talking_points": self.talking_points,
                "how_to_use": (
                    "ADVISORY ONLY. Nothing here has touched the quote. If the rep wants "
                    "to act on it, call set_pricing_levers with a commercial_multiplier "
                    "they have agreed to — never apply the suggestion silently."
                ),
            },
        }


class ClientProfileModel:
    """Per-client negotiation profiles. Lookup by id or by company name."""

    def __init__(self, frame: pd.DataFrame, tier_index: dict[str, float]):
        self._frame = frame.set_index("client_id")
        self._tier_index = tier_index
        self._by_name = {
            str(row.company_name).strip().lower(): str(idx) for idx, row in self._frame.iterrows()
        }

    @classmethod
    def build(cls) -> ClientProfileModel:
        facts = query_df(CLIENT_FACTS_SQL)
        prices = query_df(CLIENT_PRICE_SQL)
        losses = query_df(CLIENT_LOSSES_SQL)
        dates = query_df(CLIENT_DATES_SQL)

        frame = (
            facts.merge(prices, on="client_id", how="left")
            .merge(losses, on="client_id", how="left")
            .merge(dates, on="client_id", how="left")
        )

        # Population median index per leverage tier -- the fallback for a client whose own
        # history is too thin to read, and the yardstick their own index is judged against.
        priced = frame[frame["line_items"].fillna(0) >= MIN_LINE_ITEMS]
        tier_index = (
            priced.groupby("negotiation_leverage")["price_index"].median().round(4).to_dict()
        )

        repeat = int((frame["deals"].fillna(0) > 1).sum())
        debug(
            f"client profile: {len(frame):,} clients, {repeat:,} repeat "
            f"({100 * repeat / max(len(frame), 1):.0f}%), "
            f"{int(frame['line_items'].fillna(0).ge(MIN_LINE_ITEMS).sum()):,} with "
            f">={MIN_LINE_ITEMS} priced line items, tier medians {tier_index}"
        )
        return cls(frame, tier_index)

    @property
    def clients(self) -> int:
        return len(self._frame)

    @property
    def tier_index(self) -> dict[str, float]:
        return dict(self._tier_index)

    def resolve(self, term: str) -> list[str]:
        """client_id or company name -> matching client_ids. Never guesses a single winner.

        Returns every match so the caller can ask which one rather than pricing a
        conversation against the wrong account.
        """
        needle = term.strip()
        if needle in self._frame.index:
            return [needle]

        lowered = needle.lower()
        if lowered in self._by_name:
            return [self._by_name[lowered]]
        return sorted(cid for name, cid in self._by_name.items() if lowered in name)

    def profile(self, client_id: str) -> ClientNegotiationProfile:
        if client_id not in self._frame.index:
            raise KeyError(f"Unknown client_id '{client_id}'")
        return _build_profile(self._frame.loc[client_id], client_id, self._tier_index)


def _f(value) -> float | None:
    return None if pd.isnull(value) else float(value)


def _i(value) -> int:
    return 0 if pd.isnull(value) else int(value)


def _d(value) -> str | None:
    return None if pd.isnull(value) else str(pd.Timestamp(value).date())


def _build_profile(row, client_id: str, tier_index: dict[str, float]):
    line_items = _i(row.get("line_items"))
    deals = _i(row.get("deals"))
    index = _f(row.get("price_index"))
    sd = _f(row.get("price_index_sd"))
    leverage = str(row.get("negotiation_leverage"))
    tier = tier_index.get(leverage)

    # Standard error of the client's mean index. This is the whole basis for how much the
    # figure is trusted: a 30% departure on 4 line items is noise, the same departure on
    # 300 is a pattern.
    se = None
    if index is not None and sd is not None and line_items > 1:
        se = sd / (line_items**0.5)

    confidence = _confidence(line_items, index, se)
    posture, points = _read(row, client_id, index, se, tier, leverage, line_items, deals)
    suggestion = _suggest(index, se, confidence)

    if suggestion != 1.0:
        points.append(
            f"Suggested opening adjustment x{suggestion:.3f} — ADVISORY. It is not applied "
            f"to any quote; call set_pricing_levers only if the rep agrees to it."
        )

    return ClientNegotiationProfile(
        client_id=client_id,
        company_name=str(row.get("company_name")),
        industry=str(row.get("industry")),
        client_tier=str(row.get("client_tier")),
        negotiation_leverage=leverage,
        account_status=str(row.get("account_status")),
        bundle_affinity=str(row.get("bundle_affinity")),
        campaign_frequency=str(row.get("campaign_frequency")),
        deals=deals,
        line_items=line_items,
        is_repeat=deals > 1,
        first_booking=_d(row.get("first_booking")),
        last_booking=_d(row.get("last_booking")),
        typical_campaign_budget=_f(row.get("typical_campaign_budget")),
        budget_variance_pct=_f(row.get("budget_variance_pct")),
        avg_campaign_duration_days=_f(row.get("avg_campaign_duration_days")),
        realized_price_index=None if index is None else round(index, 4),
        price_index_sd=None if sd is None else round(sd, 4),
        price_index_standard_error=None if se is None else round(se, 4),
        tier_price_index=tier,
        lost_leads=_i(row.get("lost_leads")),
        price_driven_losses=_i(row.get("price_driven_losses")),
        avg_price_gap_asked=_f(row.get("avg_price_gap_asked")),
        max_price_gap_asked=_f(row.get("max_price_gap_asked")),
        avg_negotiation_rounds=_f(row.get("avg_negotiation_rounds")),
        competitor_mentions=_i(row.get("competitor_mentions")),
        posture=posture,
        confidence=confidence,
        suggested_commercial_multiplier=suggestion,
        talking_points=points,
    )


def _confidence(line_items: int, index: float | None, se: float | None) -> str:
    """How much weight the client's OWN index can carry.

    Sample size alone is not the question -- an index of 1.001 on 500 line items is
    precisely measured and says nothing actionable. What matters is whether the departure
    from neutral is large relative to its own error.
    """
    if index is None or line_items == 0:
        return "none"
    if line_items < MIN_LINE_ITEMS or se is None or se == 0:
        return "weak"
    distance = abs(index - 1.0) / se
    if distance >= CONFIDENCE_SE_THRESHOLD * 2:
        return "strong"
    if distance >= CONFIDENCE_SE_THRESHOLD:
        return "moderate"
    return "weak"


def _suggest(index: float | None, se: float | None, confidence: str) -> float:
    """Advisory opening multiplier. 1.0 whenever the history cannot carry a departure.

    Suggests the client's own realized index rather than something derived from it: the
    most defensible opening posture is "what this client has actually paid, relative to
    comparable inventory".
    """
    if index is None or se is None or confidence not in {"moderate", "strong"}:
        return 1.0
    lo, hi = SUGGESTION_RANGE
    return round(min(max(index, lo), hi), 4)


def _read(row, client_id, index, se, tier, leverage, line_items, deals) -> tuple[str, list[str]]:
    """Posture label + talking points citing real figures. No generic sales copy."""
    points: list[str] = []

    if deals > 1:
        points.append(
            f"Repeat client: {deals:,} deals, {line_items:,} line items"
            + (f", last booked {_d(row.get('last_booking'))}" if row.get("last_booking") else "")
        )
    elif deals == 1:
        points.append("Single prior deal — treat price history as indicative only.")
    else:
        points.append("No booking history: nothing here is measured from their own deals.")

    if str(row.get("account_status")) == "lapsed":
        points.append("Account is LAPSED — a win-back, not a renewal.")

    # Posture from the client's own realized index, judged against its own error.
    if index is None:
        posture = "unknown — no priced history"
    elif line_items < MIN_LINE_ITEMS:
        posture = "unknown — too few line items to read"
        points.append(
            f"Only {line_items} priced line item(s) (need {MIN_LINE_ITEMS}); their "
            f"{leverage}-leverage tier settles around x{tier:.3f} as a population figure."
            if tier
            else f"Only {line_items} priced line item(s); too few to read."
        )
    else:
        delta = (index - 1.0) * 100
        if se and abs(index - 1.0) / se >= CONFIDENCE_SE_THRESHOLD:
            if index > 1.0:
                posture = "pays above comparable inventory"
                points.append(
                    f"Has paid {delta:+.1f}% against comparable inventory across "
                    f"{line_items:,} line items (SE {se:.3f}) — this client has not needed "
                    f"a discount to transact."
                )
            else:
                posture = "settles below comparable inventory"
                points.append(
                    f"Has settled {delta:+.1f}% against comparable inventory across "
                    f"{line_items:,} line items (SE {se:.3f}) — expect to concede "
                    f"something like it again."
                )
        else:
            posture = "pays about market"
            points.append(
                f"Realized price index {index:.3f} on {line_items:,} line items is within "
                f"noise of comparable inventory"
                + (f" (SE {se:.3f})" if se else "")
                + " — no evidence they price differently from the market."
            )

    if tier is not None and index is not None and line_items >= MIN_LINE_ITEMS:
        points.append(
            f"Declared negotiation_leverage is '{leverage}' (tier median x{tier:.3f}, this "
            f"client x{index:.3f}) — context only. Per client the tier ordering does not "
            f"hold, because the label tracks account size rather than price behaviour."
        )

    # The objection history. This is what the rep most wants and cannot get anywhere else.
    losses = _i(row.get("lost_leads"))
    price_losses = _i(row.get("price_driven_losses"))
    if price_losses:
        # `price_gap_pct` is populated on most price-driven leads but not all, so the gap
        # can be unknown even when the loss is definitely price-driven. Say "not recorded"
        # rather than crashing or implying a zero gap — a rep reading "asking for 0% off"
        # would draw exactly the wrong conclusion.
        gap = _f(row.get("avg_price_gap_asked"))
        worst = _f(row.get("max_price_gap_asked"))
        if gap is None:
            asked = ", discount asked not recorded on those leads"
        else:
            asked = f", asking for {gap:.0%} off on average"
            if worst is not None and worst > gap:
                asked += f" and up to {worst:.0%}"
        points.append(
            f"Has walked away over price {price_losses} time(s) out of {losses} lost "
            f"lead(s){asked}. Network-wide, price losses come from clients asking ~33% "
            f"off — not from quotes being a few percent high."
        )
    elif losses:
        points.append(
            f"{losses} lost lead(s), none of them price-driven — their objections have not "
            f"been about the quote."
        )

    rounds = _f(row.get("avg_negotiation_rounds"))
    if rounds:
        points.append(f"Averages {rounds:.1f} negotiation round(s) on lost deals.")
    if _i(row.get("competitor_mentions")):
        points.append(
            f"Mentioned a competitor on {_i(row.get('competitor_mentions'))} lost lead(s)."
        )

    budget = _f(row.get("typical_campaign_budget"))
    variance = _f(row.get("budget_variance_pct"))
    if budget:
        points.append(
            f"Typical campaign budget {budget:,.0f}"
            + (f" with {variance:.0%} variance" if variance is not None else "")
            + f"; bundle affinity '{row.get('bundle_affinity')}'."
        )

    return posture, points


_model: ClientProfileModel | None = None
_lock = threading.Lock()


def get_client_profile_model() -> ClientProfileModel:
    """Lazy process-wide singleton. Deliberately NOT built with the pricing engine — the
    pricing path never reads it, so building it there would cost time for nothing."""
    global _model
    with _lock:
        if _model is None:
            _model = ClientProfileModel.build()
        return _model


def reset_client_profile_model() -> None:
    """Drop the singleton. For tests only."""
    global _model
    with _lock:
        _model = None
        info("client profile model reset")


__all__ = [
    "COMMERCIAL_MULTIPLIER_RANGE",
    "ClientNegotiationProfile",
    "ClientProfileModel",
    "get_client_profile_model",
    "reset_client_profile_model",
]
