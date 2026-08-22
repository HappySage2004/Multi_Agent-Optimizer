"""The ONE interface between the solver and the rest of the pipeline.

OR owns allocation decisions. The audience engine owns volume and relevance; the pricing
engine owns price and availability. If an input is missing or nonsensical this module fails
loudly, naming the component that owns the column, rather than letting the solver produce a
confident wrong plan.

Ported in spirit from `or_engine/contract.py`, but rewritten against the real contracts
(`ScreenEconomics`, `ScreenCandidate`) instead of the handoff's paraphrase of them. The
bundle's own adapters (`or_engine/adapters/`) are not used: they expect `min_free_slots`,
top-level `recommended_price`, twelve `impressions_block_*` columns and a numeric
`reason_context_fit`, none of which exist here under those names. Field-for-field it is the
same data, so an adapter that renames ours into theirs and back would only add a layer to
drift.
"""

from __future__ import annotations

import pandas as pd

from app.logging_utils import debug, error, info
from app.models.campaign import CampaignSpec
from app.models.economics import ScreenEconomics
from app.models.screens import ScreenCandidate
from app.optimize import config as C

# Column -> the component answerable for it, quoted back in the error message.
OWNERS: dict[str, str] = {
    "screen_id": "identity",
    "time_block_id": "identity (dim_slot, 1-6)",
    "price": "pricing engine  -- price per slot per day",
    "available": "pricing engine  -- free slots on the flight's tightest day",
    "exposures_per_slot_per_day": "audience engine x exposure model  -- viewed exposures",
    "pool_population": "audience engine  -- reachable people in the whole pool",
    "pool_key": "audience engine  -- screens sharing it share a crowd",
}


class ContractError(ValueError):
    """An input the solver cannot honestly optimize against."""


def build_candidate_frame(
    economics: list[ScreenEconomics],
    candidates: dict[str, ScreenCandidate],
    spec: CampaignSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """Priced lines + audience facts -> one row per purchasable (screen, time block) cell.

    Returns the frame and the notes worth reporting about how it was built (dropped rows,
    substituted defaults, pruning). Notes are surfaced in the solver log, never swallowed.
    """
    notes: list[str] = []

    purchasable = [
        e
        for e in economics
        if e.feasible
        and e.pricing is not None
        and e.pricing.recommended_price > 0
        and e.max_slots_per_day >= 1
    ]
    dropped = len(economics) - len(purchasable)
    if dropped:
        notes.append(f"dropped_unpurchasable_lines={dropped} (infeasible, unpriced, or sold out)")
        debug(
            f"contract: dropped {dropped} of {len(economics)} priced lines as "
            f"unpurchasable (infeasible, unpriced, or sold out)"
        )
    if not purchasable:
        error(
            f"contract: none of the {len(economics)} priced lines is purchasable — "
            f"nothing to hand the solver"
        )
        return pd.DataFrame(), notes

    # A hard constraint the validator enforces, so it must bind before the solve rather
    # than being discovered after it. The blocks actually priced come from the campaign's
    # preferred blocks, which may be a wider set than this.
    hc = spec.hard_constraints
    required_blocks = hc.get("required_time_blocks") or hc.get("time_blocks")
    if required_blocks:
        allowed = {str(b) for b in required_blocks}
        before = len(purchasable)
        purchasable = [e for e in purchasable if str(e.time_block_id) in allowed]
        if not purchasable:
            raise ContractError(
                f"hard constraint required_time_blocks={sorted(allowed)} excludes every "
                f"priced line. The pricing stage was asked for different blocks than the "
                f"constraint permits."
            )
        if before != len(purchasable):
            removed = before - len(purchasable)
            notes.append(f"required_time_blocks={sorted(allowed)} removed {removed} cell(s)")
            debug(
                f"contract: hard constraint required_time_blocks={sorted(allowed)} removed "
                f"{removed} of {before} cell(s)"
            )

    missing_candidate = 0
    rows: list[dict] = []
    for e in purchasable:
        candidate = candidates.get(e.screen_id)
        if candidate is None:
            missing_candidate += 1
            continue

        block = str(e.time_block_id)
        # Block-qualified: two screens on one platform share a crowd WITHIN a block, not
        # across blocks. The reported reach accounting groups on the same (pool, block)
        # pair, so the two definitions stay in step.
        pool_key = f"{candidate.pool_key or e.screen_id}|{block}"

        # The pool's whole crowd, not one vehicle's share of it. For stop-mounted screens
        # the partition count is 1 and this is the per-screen figure unchanged.
        pool_population = e.reachable_daily_audience * candidate.pool_partition_count

        row = {
            "screen_id": e.screen_id,
            "time_block_id": int(block),
            "pool_key": pool_key,
            "price": e.pricing.recommended_price,
            "available": e.max_slots_per_day,
            "exposures_per_slot_per_day": e.viewed_exposures_per_slot_per_day,
            "reachable_daily_audience": e.reachable_daily_audience,
            "pool_population": pool_population,
            "pool_partition_count": candidate.pool_partition_count,
            # Carried for reporting and for coverage groups; not in the objective.
            "relevance_score": candidate.relevance_score,
            # POI/context proximity, the only conversion-shaped signal in this system.
            # It is NOT a conversion model — see solver.PROFILES.
            "conv_fit": candidate.contextual_score,
            "screen_type": candidate.screen_type,
            "zone_id": candidate.zone_id,
        }
        row.update(_slot_price_factors(e))
        rows.append(row)

    if missing_candidate:
        notes.append(
            f"dropped_lines_without_audience_row={missing_candidate} "
            f"(priced but absent from screen_candidates)"
        )
        # A seam failure between stages 4 and 5, not a data property — worth [INFO].
        info(
            f"contract: dropped {missing_candidate} priced line(s) with no matching "
            f"screen_candidates row — the pricing and audience artifacts disagree"
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, notes

    frame, zero_audience = drop_zero_audience(frame)
    if zero_audience:
        notes.append(
            f"dropped_cells_with_no_modelled_audience={zero_audience} "
            f"(zero means not modelled, never 'nobody there')"
        )
        debug(
            f"contract: dropped {zero_audience} cell(s) with no modelled audience "
            f"(zero = not modelled, never 'nobody there')"
        )
    if frame.empty:
        error("contract: every candidate cell was dropped — no modelled audience anywhere")
        return frame, notes

    frame, pruned = _prune_saturated_pools(frame)
    if pruned:
        notes.append(
            f"pruned_cells_beyond_{C.MAX_CELLS_PER_POOL}_per_pool={pruned} "
            f"(reach saturates; extra cells in one pool add search symmetry, not audience)"
        )
        debug(
            f"contract: pruned {pruned} cell(s) beyond {C.MAX_CELLS_PER_POOL} per pool "
            f"(reach saturates; extras add search symmetry, not audience)"
        )

    frame = validate(frame)
    debug(f"contract: candidate frame ready {describe(frame)}")
    return frame, notes


def _slot_price_factors(line: ScreenEconomics) -> dict[str, float]:
    """disc1..disc6: price at k slots relative to the 1-slot price.

    Flat (1.0) by design today — the apparent volume discount in `bookings` is an
    inventory-composition confound, ~1.6% within a price-band segment. Reading the map
    rather than assuming flatness means a future real curve needs no change here.

    Beyond availability the map holds null; 1.0 is carried because the availability
    constraint already forbids that depth.
    """
    base = line.pricing.recommended_price
    curve = line.price_by_slot_count or {}
    factors: dict[str, float] = {}
    for k in range(1, C.SLOTS_PER_CELL + 1):
        raw = curve.get(k, curve.get(str(k)))
        try:
            factors[f"disc{k}"] = float(raw) / base if raw is not None and base > 0 else 1.0
        except (TypeError, ValueError):
            factors[f"disc{k}"] = 1.0
    return factors


def _prune_saturated_pools(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the best `MAX_CELLS_PER_POOL` cells per audience pool, by exposures per dollar.

    The handoff applied this in its demo scripts and never in the solver, while its
    `diagnose` tool reported "audience saturation within pools" as binding by comparing
    against the constant. Enforced here so the report and the model agree.
    """
    if frame.empty:
        return frame, 0
    ranked = frame.assign(
        _value_per_cost=frame["exposures_per_slot_per_day"] / frame["price"].clip(lower=1e-9)
    ).sort_values(["pool_key", "_value_per_cost"], ascending=[True, False])
    kept = ranked.groupby("pool_key", sort=False).head(C.MAX_CELLS_PER_POOL)
    pruned = len(frame) - len(kept)
    return kept.drop(columns="_value_per_cost").reset_index(drop=True), pruned


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on anything the solver cannot honestly optimize against."""
    missing = [c for c in OWNERS if c not in frame.columns]
    if missing:
        error(f"contract: candidate frame missing required columns {missing}")
        raise ContractError(
            "candidate frame missing required columns:\n"
            + "\n".join(f"  {c:28s} <- {OWNERS[c]}" for c in missing)
        )

    problems: list[str] = []
    if frame.empty:
        problems.append("candidate frame is empty")
    else:
        if not frame["time_block_id"].between(1, C.SLOTS_PER_CELL).all():
            problems.append("time_block_id outside 1-6")
        if not frame["available"].between(1, C.SLOTS_PER_CELL).all():
            problems.append("available outside 1-6 (unpurchasable rows must be dropped first)")
        if (frame["price"] <= 0).any():
            problems.append("non-positive price")
        if (frame["exposures_per_slot_per_day"] < 0).any():
            problems.append("negative exposures")
        if frame["pool_key"].isna().any():
            problems.append("null pool_key — reach cannot be deduplicated")
        if (frame["pool_population"] <= 0).any():
            # Real and expected: block 1 is zero everywhere, and any block with no
            # scheduled service reports zero. Those cells cannot contribute reach, so they
            # are dropped rather than being handed to the solver as a zero-ceiling pool.
            problems.append("non-positive pool_population")
        if frame.duplicated(["screen_id", "time_block_id"]).any():
            problems.append("duplicate (screen_id, time_block_id) cells")

    if problems:
        error(f"contract: candidate frame failed validation: {problems}")
        raise ContractError(
            "candidate frame failed validation:\n" + "\n".join("  - " + p for p in problems)
        )
    return frame.reset_index(drop=True)


def drop_zero_audience(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove cells whose pool has no modelled audience, BEFORE validation.

    A zero here means "not modelled", never "nobody there" — volume is schedule-derived
    with no ambient term, so block 1 (00:00-04:00) is zero for every screen in the network
    despite 8,544 real block-1 bookings. Such a cell can carry frequency but no reach, and
    a zero-population pool has no meaningful saturation curve, so it is excluded and
    counted rather than silently optimized against.
    """
    if frame.empty:
        return frame, 0
    keep = frame["pool_population"] > 0
    return frame[keep].reset_index(drop=True), int((~keep).sum())


def describe(frame: pd.DataFrame) -> dict:
    """Shape of the frame handed to the solver, for the solver log."""
    if frame.empty:
        return {"cells": 0}
    return {
        "cells": len(frame),
        "screens": int(frame["screen_id"].nunique()),
        "audience_pools": int(frame["pool_key"].nunique()),
        "time_blocks": sorted(frame["time_block_id"].unique().tolist()),
        "price_min": round(float(frame["price"].min()), 2),
        "price_max": round(float(frame["price"].max()), 2),
        "mean_free_slots": round(float(frame["available"].mean()), 2),
    }
