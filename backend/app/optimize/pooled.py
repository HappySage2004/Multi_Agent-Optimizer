"""Shared-audience reach: the piece that stops the optimizer double-counting.

Screens at one stop, or on one corridor, reach the same people. Exposures add; reach
saturates. Two saturation models appear in this system, and only one of them counts.

REPORTED, and what the solver maximizes:

    reach = SUM over (pool, block) of min(gross viewed exposures, reachable audience)

`min()` of two linear functions is concave, so bounding it needs exactly two linear
constraints and no parameter (`solver.py`, "reach: R <= min(E, P)"). It is computed by
`or_agent_tools._package_metrics` and recomputed independently by
`validation._reach_checks`.

DIAGNOSTIC ONLY:

    reach(E) = P * (1 - exp(-lambda * E / P))

This is the handoff bundle's model, bounded in its solver by tangent lines. It is retained
here so the two can be compared, and for nothing else. `REACH_LAMBDA` is ASSUMED with no
ground truth in the 14 CSVs, so it belongs in neither a validated figure nor a client-facing
one — and maximizing it does not maximize the reported reach, which cost measured audience
before the switch (141,501-157,869 against 261,329 on the canonical brief).

The tangent construction itself is deleted rather than kept unused: six constraints per pool
approximating a curve nothing reports.

`curve_reach` is guarded at its call site by `curve_reach <= min(sum E, sum P)` — an
assertion with no lambda dependency, which is why it is worth making: it catches pool
misalignment and index drift, the failure modes that produce a confidently wrong audience
number rather than an obviously broken one.

Ported from `or_engine/pooled.py`. One further deliberate change, at its site: a pool with
no resolvable population raises instead of silently defaulting to 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.logging_utils import debug, error
from app.optimize import config as C


class PoolPopulationError(ValueError):
    """A pool's audience ceiling could not be resolved. Never guessed."""


def pool_population(cand: pd.DataFrame, pool_keys: list[str]) -> np.ndarray:
    """Reachable people per pool, aligned to `pool_keys`.

    `pool_population` is supplied per row by `contract.build_candidate_frame`, which
    reconstructs the pool's whole crowd from the per-screen figure and
    `pool_partition_count`. Screens in a pool carry the same value, so `max` is the
    aggregate — it is a lookup, not an estimate.

    THAT PREMISE IS LOAD-BEARING AND HAS BEEN BROKEN ONCE. `max` here is taken over the
    whole candidate frame, while the reported reach (`or_agent_tools._package_metrics`) and
    the validator take it over the ALLOCATED lines. Those agree only while every screen in a
    pool reports the same audience. When `TERMINUS_WEIGHT` was 1.5 they did not — a site
    merging both sides of a road held a route's terminus and a mid-route stop, 1.5x apart —
    and the two ceilings diverged, so `curve_reach_bounded` failed on every package. Any
    change that makes a pool's screens disagree has to be reconciled at the SOURCE, in
    `app/data/db.py`, not papered over by picking a convention here.

    Port note: the handoff did `.reindex(keys).fillna(1.0)`. A pool silently assigned a
    population of ONE PERSON is not a conservative fallback — it makes that pool look
    instantly saturated, so the solver stops buying there and the shortfall is invisible.
    Raises instead.
    """
    if "pool_population" not in cand.columns:
        error("pool_population column absent from the candidate frame — contract violated")
        raise PoolPopulationError(
            "candidate frame has no pool_population column; "
            "contract.build_candidate_frame must supply it"
        )
    by_pool = cand.groupby("pool_key")["pool_population"].max()
    aligned = by_pool.reindex(pool_keys)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()[:5]
        error(
            f"{int(aligned.isna().sum())} pool(s) have no resolvable population, e.g. "
            f"{missing} — refusing to guess a ceiling"
        )
        raise PoolPopulationError(
            f"{int(aligned.isna().sum())} pool(s) have no resolvable population, e.g. "
            f"{missing}. A pool with no audience ceiling cannot be optimized against."
        )
    populations = aligned.to_numpy(dtype=float)
    debug(
        f"pooled: {len(pool_keys)} pool ceiling(s) resolved, reachable people "
        f"total {populations.sum():,.0f} "
        f"(min {populations.min():,.0f} / median {np.median(populations):,.0f} / "
        f"max {populations.max():,.0f})"
    )
    return populations


def curve_reach(exposures: np.ndarray, population: np.ndarray) -> np.ndarray:
    """Evaluate the saturation curve directly, per pool. Diagnostic only.

    Port note (handoff CHANGELOG section 5): a reach figure must NOT be read off a solver
    variable that is only bounded from ABOVE. When its objective weight is small the solver
    has no incentive to raise it and may leave it at zero on a plan that genuinely reaches
    people — their frequency profile reported zero reached on a plan reaching 633,000.
    Evaluating the curve from the exposures is exact and weight-independent. The same
    reasoning applies to the reported `R` variables here, which is why
    `_package_metrics` recomputes reach from the allocations rather than reading the solve.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        reach = np.where(
            population > 0,
            population * (1.0 - np.exp(-C.REACH_LAMBDA * exposures / np.maximum(population, 1e-9))),
            0.0,
        )
    return np.nan_to_num(reach, nan=0.0, posinf=0.0)
