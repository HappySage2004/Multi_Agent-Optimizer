"""Pricing levers — the parameters an agent turn is allowed to move.

Every multiplier in the pricing engine used to be derived once and applied on every run,
which meant a second turn could not act on anything the sales rep said. A rep who knows
the client will not pay a peak premium, or that the seasonality haircut is wrong for this
flight, had no way to express it: the only lever was rewriting the brief.

This module is that surface. It is deliberately NOT a free-form price override:

* Each lever moves an EXISTING, named step of the price decision. Nothing here invents a
  new term, and nothing here can bypass the feasibility gate — a sold-out screen stays
  sold out no matter what the rep asks for.
* Every lever is CLAMPED IN CODE. `clamp()` never rejects, it bounds and reports, because
  a rejected tool call in an agent loop turns into a retry rather than a corrected price.
  The agent is told exactly what it asked for and exactly what it got.
* Defaults are all identity. A run with no levers set is byte-identical to the behaviour
  before this module existed, which is what makes the clamped path safe to trust.

The lever set is intentionally small and additive. When the client-elasticity and
screen-demand factors land, they arrive as further fields here with identity defaults, and
`effective_multiplier` stays the one place a weight becomes a number.

WHERE EACH LEVER LANDS IN THE PRICE DECISION
--------------------------------------------
    1. feasibility gate                             <- NO lever reaches this
    2. band = p25/p50/p90 of comparables
         x industry adjustment                      <- industry_weight
    3. x day-of-week / holiday multiplier           <- seasonality_weight
       x event boost                                <- event_weight
    4. position in band = occupancy ** gamma        <- occupancy_gamma
       or an explicit position                      <- band_position (overrides 4)
    5. price = floor + position x (cap - floor)
    6. x per-screen demand premium                  <- demand_premium_weight
    7. x commercial adjustment                      <- commercial_multiplier
    8. price = max(price, floor)                    <- respect_band_floor

A NOTE ON `demand_premium_weight`, because it is the one lever whose identity default does
something. All the others reduce to "no adjustment" at 1.0; this one applies the demand
model's computed premium in full. That is the same semantics, not an exception -- a weight
of 1.0 has always meant "use the model's derived value", exactly as `seasonality_weight=1.0`
applies the derived seasonality multiplier rather than nothing. Setting it to 0.0 is what
returns the engine to pricing purely off historical comparables.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Bounds are business judgements, and each is here rather than inline so the whole
# permitted range of the engine is readable in one place.

# A weight dials how much of a derived multiplier survives, not its magnitude: 0.0 turns
# the term off, 1.0 is the derived value, 2.0 doubles its deviation from neutral. Capped at
# 2.0 because past that a rep is no longer adjusting the model, they are replacing it.
WEIGHT_RANGE = (0.0, 2.0)

# Reshapes occupancy -> position in band while pinning both ends (empty screen quotes at
# floor, full screen quotes at cap). Below 1.0 is aggressive, above is conservative.
OCCUPANCY_GAMMA_RANGE = (0.25, 4.0)

# An explicit position inside the band. Cannot escape the band — that is what the
# commercial multiplier is for, and it is separately bounded.
BAND_POSITION_RANGE = (0.0, 1.0)

# The negotiation lever. +/-30% off a band already built from real contracted prices is a
# wide commercial range; wider than this is a different price list, not an adjustment.
COMMERCIAL_MULTIPLIER_RANGE = (0.70, 1.30)


class PricingLevers(BaseModel):
    """Run-scoped pricing parameters. All defaults are identity."""

    seasonality_weight: float = Field(
        default=1.0,
        description=(
            "How much of the day-of-week / holiday ridership multiplier to apply. 1.0 is "
            "the derived value, 0.0 disables it. Worth knowing before you move it: the "
            "multiplier averages 0.913 over a full week, so a whole-week flight takes a "
            "~9% haircut off a band already built from actual contracted prices. Setting "
            "this to 0.0 is the documented way to stop that double-count."
        ),
    )
    event_weight: float = Field(
        default=1.0,
        description=(
            "How much of the event-overlap boost to apply. 1.0 is the derived value "
            "(1.03/1.08/1.15 by attendance tier, halved for a zone-only match), 0.0 "
            "disables it, 2.0 doubles the premium."
        ),
    )
    industry_weight: float = Field(
        default=1.0,
        description=(
            "How much of the industry-vertical band adjustment to apply. The EFFECTIVE "
            "adjustment stays clamped to [0.85, 1.20] whatever this is set to, so the "
            "guarantee that industry never swings the band more than -15/+20% survives "
            "the lever."
        ),
    )
    occupancy_gamma: float = Field(
        default=1.0,
        description=(
            "Reshapes how occupancy positions the price inside the band: "
            "position = occupancy_rate ** gamma. 1.0 is linear (today). Below 1.0 quotes "
            "higher on partly-empty inventory (aggressive); above 1.0 quotes lower "
            "(defensive). An empty screen still quotes at floor and a full one at cap."
        ),
    )
    band_position: float | None = Field(
        default=None,
        description=(
            "Quote at a FIXED position in the band instead of an occupancy-driven one. "
            "0.0 = floor, 0.5 = midpoint, 1.0 = cap. None (the default) keeps the "
            "occupancy rule. Use when the rep wants a posture rather than a scarcity "
            "read — 'open at the cap', 'go in at floor to win the logo'."
        ),
    )
    demand_premium_weight: float = Field(
        default=1.0,
        description=(
            "How much of the per-screen demand premium to apply. This one is AUTO-APPLIED "
            "at 1.0: the model (app/ml/demand_value.py) finds screens that are underpriced "
            "for the audience they deliver and raises them by up to 15%, gated on the "
            "screen actually selling and restricted to fixed inventory. Set 0.0 to quote "
            "purely off historical comparables — which is what the engine did before this "
            "model existed, and undersells the screens it flags."
        ),
    )
    commercial_multiplier: float = Field(
        default=1.0,
        description=(
            "Blanket commercial adjustment applied last: 0.95 quotes 5% under, 1.10 quotes "
            "10% over. This is the negotiation lever. It does not change the band, so the "
            "quoted price stays traceable to its comparables."
        ),
    )
    respect_band_floor: bool = Field(
        default=True,
        description=(
            "Keep the final price at or above the band floor (p25 of comparables, after "
            "seasonality). Set False only to authorise a quote below the floor — the row "
            "then discloses that it went sub-floor."
        ),
    )
    note: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Why these levers were set — the rep's reason, in their words. Carried onto "
            "the artifact so a price is traceable to the conversation that produced it."
        ),
    )

    def clamp(self) -> tuple[PricingLevers, list[str]]:
        """Bound every lever to its permitted range. Returns the safe levers + what moved.

        Bounds rather than raises on purpose: in an agent loop a rejected call becomes a
        retry, and a retry costs a model call against a per-minute rate limit to arrive at
        the number this could have returned directly. The caller is told what was clamped
        so it can say so instead of quietly pricing off a different figure.
        """
        adjustments: list[str] = []

        def bound(name: str, value: float, low: float, high: float) -> float:
            capped = min(max(value, low), high)
            if capped != value:
                adjustments.append(f"{name} {value:g} clamped to {capped:g} (allowed {low}-{high})")
            return capped

        clamped = self.model_copy(
            update={
                "seasonality_weight": bound(
                    "seasonality_weight", self.seasonality_weight, *WEIGHT_RANGE
                ),
                "event_weight": bound("event_weight", self.event_weight, *WEIGHT_RANGE),
                "industry_weight": bound("industry_weight", self.industry_weight, *WEIGHT_RANGE),
                "demand_premium_weight": bound(
                    "demand_premium_weight", self.demand_premium_weight, *WEIGHT_RANGE
                ),
                "occupancy_gamma": bound(
                    "occupancy_gamma", self.occupancy_gamma, *OCCUPANCY_GAMMA_RANGE
                ),
                "band_position": (
                    None
                    if self.band_position is None
                    else bound("band_position", self.band_position, *BAND_POSITION_RANGE)
                ),
                "commercial_multiplier": bound(
                    "commercial_multiplier",
                    self.commercial_multiplier,
                    *COMMERCIAL_MULTIPLIER_RANGE,
                ),
            }
        )
        return clamped, adjustments

    def is_default(self) -> bool:
        """True when nothing was moved — the run prices exactly as it would with no levers."""
        return (
            self.seasonality_weight == 1.0
            and self.event_weight == 1.0
            and self.industry_weight == 1.0
            and self.demand_premium_weight == 1.0
            and self.occupancy_gamma == 1.0
            and self.band_position is None
            and self.commercial_multiplier == 1.0
            and self.respect_band_floor is True
        )

    def changes(self) -> list[str]:
        """Human-readable list of the levers that are not at their identity default."""
        moved: list[str] = []
        if self.seasonality_weight != 1.0:
            moved.append(f"seasonality_weight={self.seasonality_weight:g}")
        if self.event_weight != 1.0:
            moved.append(f"event_weight={self.event_weight:g}")
        if self.industry_weight != 1.0:
            moved.append(f"industry_weight={self.industry_weight:g}")
        if self.demand_premium_weight != 1.0:
            moved.append(f"demand_premium_weight={self.demand_premium_weight:g}")
        if self.occupancy_gamma != 1.0:
            moved.append(f"occupancy_gamma={self.occupancy_gamma:g}")
        if self.band_position is not None:
            moved.append(f"band_position={self.band_position:g}")
        if self.commercial_multiplier != 1.0:
            moved.append(f"commercial_multiplier={self.commercial_multiplier:g}")
        if not self.respect_band_floor:
            moved.append("respect_band_floor=false")
        return moved


DEFAULT_LEVERS = PricingLevers()


def effective_multiplier(multiplier: float, weight: float) -> float:
    """Dial a derived multiplier's INFLUENCE, not its magnitude.

        weight 0.0 -> 1.0 (term off)
        weight 1.0 -> multiplier (unchanged)
        weight 2.0 -> twice the deviation from neutral

    Weighting the deviation rather than scaling the multiplier is what makes 0.0 mean
    exactly "off" for a term that can sit on either side of 1.0. Scaling would turn a 0.88
    haircut into 0.0 at weight 0 and a 1.15 boost into 0.0 as well.
    """
    return 1.0 + weight * (multiplier - 1.0)
