"""Media-plan MILP. Ported from the OR handoff bundle's `or_engine/solver_v1.py`.

================================ FORMULATION ================================
For each candidate cell i = (screen, time block):

    y[i,k]  binary      "cell i gets at least k+1 rotation slots",  k = 0..5
    z[s]    binary      "screen s is used"
    E[p]    continuous  viewed exposures accumulated in audience pool p
    R[p]    continuous  reach in pool p, R <= min(E[p], P[p]) — two exact linear bounds
    c[g]    continuous  shortfall on elastic coverage group g
    w[p]    continuous  shortfall on the wear-out frequency cap in pool p

    max  w_reach * sum(R) + w_freq * sum(E) + w_conv * sum(conv_fit * E)
         - coverage_penalty * sum(c) - wear_out_penalty * sum(w)
         - tie_breaker * cost

Hard:     budget, per-day availability, slot ordering, screen count, required blocks,
          slots per screen per day.
Elastic:  coverage groups, wear-out cap.

Slots are LINEAR in value: all 6 slots loop continuously through the block, so the k-th
slot is worth exactly as much as the first (`exposure.py`). The only concavity is at the
pool, where a shared crowd saturates — and it is the `min(E, P)` bound above, which is the
definition `or_agent_tools._package_metrics` reports and `validation._reach_checks`
recomputes. Solver and report optimize the same number.

============================ CHANGES ON THE PORT ============================
Five, each at its site below:

0.  REACH IS BOUNDED BY min(E, P), NOT BY TANGENT LINES. The bundle bounded a concave
    exponential saturation curve by its tangents. That curve carries an assumed constant
    and is a different function from the reach this system reports, so maximizing it
    left reach on the table — measured 141,501-157,869 against the true optimum on the
    canonical brief. min() is concave too, needs two constraints instead of six, has no
    free parameter, and makes the solver optimize exactly what the validator checks.

1.  SCREEN COUNT COUNTS SCREENS. The bundle summed the level-1 binary over all cells
    (`solver_v1.py:108`), which counts (screen x block) CELLS: a screen bought in three
    blocks counted three times, so `min_screens=8` could return five screens while the
    summary reported `nunique(screen_id)`. Our validator checks distinct screen ids
    against `requested_num_screens`, so this had to become an explicit z[s] indicator.

2.  THE SPEND FLOOR RELAXES. `MIN_SPEND_FRAC = 0.90` was a hard constraint. It is not
    declared by any campaign spec, and combined with an exact screen count it can make a
    brief infeasible that has a perfectly good package. It is now a ladder
    (`MIN_SPEND_FRACTIONS`) tried in order, with the level used reported.

3.  WEAR-OUT CAP added, elastic. See WEAR-OUT below for why it cannot be hard.

4.  MARGINAL VALUE ARRIVES PRE-COMPUTED. The bundle applied `LOOP_PASSES_PER_TRIP / 6`
    and viewability inside `_coeffs`. Both now live in `exposure.py` and are applied once,
    upstream, so `ScreenEconomics` and the package are in the same unit.

=========================== SLOTS PER SCREEN PER DAY ===========================
`max_slots_per_screen_per_day` binds on the PHYSICAL SCREEN, summed over time blocks:

    for each screen s:   sum over its cells i, over k:  y[i,k]  <=  cap

The alternative reading — cap each (screen, block) cell — was what the pipeline effectively
did, and it is wrong for the constraint briefs actually write. A brief saying "1 rotating
slot on each screen" describes how much of that screen's airtime the client holds; under a
per-cell reading a screen bought in Block 3 and Block 5 takes 1 + 1 = 2 slots that day and
passes. Measured on a 45-day EV brief: a per-cell cap of 1 returned a plan whose busiest
screen carried 2 slots/day, which is an over-delivery against the contract however the
blocks are labelled.

The per-screen reading is also the stricter one, and for a compliance constraint the
stricter reading is the right default. The applied cap and this semantics string are
reported in the package payload, so the figure always says which reading produced it.

A per-cell clip cannot express this. `contract` still clips `available` to the cap, because
a per-screen limit of k implies no cell exceeds k and the tighter bound shrinks the search —
but the clip is an implication of the constraint, never the constraint.

================================ WEAR-OUT ================================
The cap is `E[p] <= F_max * R[p] + w[p]`, with `w` penalized above any exposure reward, so
it binds like a hard constraint but yields rather than manufacturing an infeasibility.

`F_max` is a MULTIPLE of the flight's unavoidable exposure floor, not an absolute exposure
count, and that is what makes it satisfiable. Duration is fixed by the brief, the minimum
purchase is one slot and daily allocation is constant, so exposures per reached person
cannot go below `LOOP_PASSES_PER_TRIP / SLOTS_PER_CELL * days` — about 40 on a 30-day
flight — before the optimizer chooses anything. An absolute cap below that is satisfiable by
no plan at all.

What the cap does is stop STACKING: piling extra slots and extra screens into a pool the
plan has already saturated, which buys frequency nobody asked for. It also removes the
distortion the spend floor was creating. With a 90% floor and no binding cap the solver had
to convert every leftover rupee into depth — measured on the canonical brief: 100% of budget
spent, 16 screens, 112 exposures per person, and not one of those last rupees added a single
person to reach. Now the cap binds first, the spend-floor ladder relaxes, and the unspent
budget is reported as a finding: reach saturated before the budget did.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

from app.logging_utils import debug, error, info
from app.optimize import config as C
from app.optimize import pooled

# Objective weights per campaign goal, from the bundle's PROFILES with one change.
#
# `reach` carries NO exposure weight. The bundle used 0.05 because pure reach maximisation
# gave its LP relaxation almost no guidance: under the tangent construction the binaries
# reached the objective only indirectly through upper-bounded R variables, and the search
# stalled (measured: w_freq=0.00 returning 485,896 reached against ~930,000 at 0.05). The
# exact `R <= min(E, P)` bound removes that problem at the source — reach now has a direct
# gradient from the slot binaries — so the crutch is not merely unnecessary but harmful: it
# made the reach profile spend its budget on depth. Measured on the canonical brief,
# w_freq=0.05 returned 227,119 reached at 76x frequency; w_freq=0.00 returns 261,329 at
# 40x, and solves in 0.1s rather than hitting the 30s limit.
#
# `conversion` weights `conv_fit`, the audience engine's industry->POI context score. That
# is a PROXY, not a conversion model: this system has no conversion data of any kind. The
# substitution is recorded in the solver log and the OR agent's prompt requires it to be
# reported as such.
PROFILES: dict[str, dict[str, float]] = {
    "reach": {"w_reach": 1.00, "w_freq": 0.00, "w_conv": 0.00},
    "awareness": {"w_reach": 0.70, "w_freq": 0.30, "w_conv": 0.00},
    "frequency": {"w_reach": 0.20, "w_freq": 0.80, "w_conv": 0.00},
    "conversion": {"w_reach": 0.40, "w_freq": 0.20, "w_conv": 0.40},
}

DEFAULT_PROFILE = "awareness"


def exposure_floor_per_person(days: int) -> float:
    """Exposures per reached person that no allocation can go below on this flight.

    One slot on a fully saturated pool already delivers `LOOP_PASSES_PER_TRIP / 6` viewed
    exposures per person per day. Duration is the brief's, and there is no flighting, so
    this is a property of the campaign length rather than of the selection.
    """
    return C.LOOP_PASSES_PER_TRIP / C.SLOTS_PER_CELL * days


def wear_out_frequency_cap(days: int) -> float:
    """The frequency above which extra exposure is the optimizer's choice, not the flight's."""
    return C.WEAR_OUT_STACKING_MULTIPLE * exposure_floor_per_person(days)


class ReachAccountingError(AssertionError):
    """The solver's pool bookkeeping disagrees with its own inputs."""


class ConstraintBreachError(AssertionError):
    """The returned plan violates a constraint the formulation was given.

    Distinct from `ReachAccountingError`, which is about arithmetic. This one means a hard
    row did not bind — a class of bug that yields a plausible-looking package breaching a
    client's written brief, so it fails loudly rather than being reported and validated
    later.
    """


@dataclass
class SolveOutcome:
    """What the solver found. `plan` is None exactly when nothing feasible exists."""

    status: str  # "optimal" | "feasible" | "infeasible"
    plan: pd.DataFrame | None
    objective: str
    detail: str = ""
    spend: float = 0.0
    exposures: float = 0.0
    curve_reach_diagnostic: float = 0.0
    pools_used: int = 0
    screens_used: int = 0
    min_spend_fraction: float = 0.0
    coverage_shortfall: dict[str, float] = field(default_factory=dict)
    wear_out_exposures_over_cap: float = 0.0
    wear_out_pools: int = 0
    max_slots_per_screen_per_day: int | None = None
    slots_on_busiest_screen: int = 0
    slots_in_busiest_cell: int = 0
    pool_table: pd.DataFrame | None = None
    log: list[str] = field(default_factory=list)


def solve(
    cand: pd.DataFrame,
    *,
    budget: float,
    days: int,
    objective: str = DEFAULT_PROFILE,
    min_screens: int = 0,
    max_screens: int | None = None,
    exact_screens: int | None = None,
    coverage: dict[str, dict] | None = None,
    unit_coverage: dict[str, dict] | None = None,
    time_limit: float = 30.0,
    wear_out_cap: float | None = None,
    min_spend_fraction: float | None = None,
    max_slots_per_screen_per_day: int | None = None,
) -> SolveOutcome:
    """Solve one media plan, relaxing the spend floor rather than reporting a false
    infeasibility.

    `cand` must come from `contract.build_candidate_frame`.

    `coverage` is ELASTIC: `{label: {"mask": bool array over rows, "min": int}}` — at least
    n cells from this group, with a penalized shortfall if it cannot be met.

    `unit_coverage` is HARD: `{label: {"masks": {unit: bool array}, "min_units": int}}` —
    at least n of these units must contain a purchased cell. This is what
    `min_zone_coverage` needs, and it is not expressible as a cell count: "cover 3 zones"
    is a cardinality over distinct groups, not 3 cells that might all sit in one zone.
    Hard because the validation layer fails a package that misses it.

    `max_slots_per_screen_per_day` caps slots on the PHYSICAL SCREEN across all its time
    blocks — see the module docstring for why that, and not the per-cell reading.
    """
    profile = PROFILES.get(objective, PROFILES[DEFAULT_PROFILE])
    log: list[str] = [
        f"objective={objective}",
        (
            f"profile=w_reach:{profile['w_reach']} w_freq:{profile['w_freq']} "
            f"w_conv:{profile['w_conv']}"
        ),
        (
            f"cells={len(cand)} screens={cand['screen_id'].nunique()} "
            f"pools={cand['pool_key'].nunique()}"
        ),
        f"budget={budget:,.2f} days={days}",
    ]
    if objective == "conversion":
        log.append(
            "no conversion model exists in this system: w_conv weights the audience "
            "engine's industry->POI context score as a proxy, which is reported rather "
            "than presented as a conversion optimum"
        )
        info(
            "MILP objective 'conversion' has no conversion model: w_conv weights the "
            "audience engine's POI context score as a PROXY"
        )

    if cand.empty:
        error("MILP received an empty candidate frame — nothing purchasable to optimize")
        return SolveOutcome(
            status="infeasible",
            plan=None,
            objective=objective,
            detail="No purchasable candidate cells.",
            log=log,
        )

    debug(
        f"MILP solve: objective={objective} "
        f"(w_reach={profile['w_reach']} w_freq={profile['w_freq']} w_conv={profile['w_conv']}), "
        f"cells={len(cand)} screens={cand['screen_id'].nunique()} "
        f"pools={cand['pool_key'].nunique()}, budget={budget:,.0f} days={days}, "
        f"screens_min={min_screens} max={max_screens} exact={exact_screens}, "
        f"time_limit={time_limit:g}s gap={C.MIP_REL_GAP:.0%}"
    )

    floor = C.MIN_SPEND_FRACTION_DEFAULT if min_spend_fraction is None else min_spend_fraction
    return _solve_once(
        cand,
        budget=budget,
        days=days,
        objective=objective,
        profile=profile,
        min_screens=min_screens,
        max_screens=max_screens,
        exact_screens=exact_screens,
        coverage=coverage,
        unit_coverage=unit_coverage,
        time_limit=time_limit,
        wear_out_cap=wear_out_cap,
        min_spend_fraction=floor,
        max_slots_per_screen_per_day=max_slots_per_screen_per_day,
        log=list(log),
    )


def _solve_once(
    cand: pd.DataFrame,
    *,
    budget: float,
    days: int,
    objective: str,
    profile: dict[str, float],
    min_screens: int,
    max_screens: int | None,
    exact_screens: int | None,
    coverage: dict[str, dict] | None,
    unit_coverage: dict[str, dict] | None,
    time_limit: float,
    wear_out_cap: float | None,
    min_spend_fraction: float,
    max_slots_per_screen_per_day: int | None,
    log: list[str],
) -> SolveOutcome:
    S = C.SLOTS_PER_CELL
    n = len(cand)

    value, marginal_cost, total_cost = _coefficients(cand, days)
    available = cand["available"].to_numpy()

    pool_keys = sorted(cand["pool_key"].unique())
    pool_index = {k: j for j, k in enumerate(pool_keys)}
    pool_of_cell = cand["pool_key"].map(pool_index).to_numpy()
    population = pooled.pool_population(cand, pool_keys)
    n_pools = len(pool_keys)

    screen_ids = sorted(cand["screen_id"].unique())
    screen_index = {s: j for j, s in enumerate(screen_ids)}
    screen_of_cell = cand["screen_id"].map(screen_index).to_numpy()
    n_screens = len(screen_ids)

    groups = list((coverage or {}).items())
    n_groups = len(groups)
    cap = wear_out_cap if wear_out_cap is not None else wear_out_frequency_cap(days)
    units = [
        (label, unit, mask)
        for label, spec in (unit_coverage or {}).items()
        for unit, mask in spec["masks"].items()
    ]
    n_units = len(units)

    # Variable layout. Everything before off_e is binary.
    n_y = n * S
    off_z = n_y
    off_u = off_z + n_screens
    off_e = off_u + n_units
    off_r = off_e + n_pools
    off_c = off_r + n_pools
    off_w = off_c + n_groups
    n_vars = off_w + n_pools

    rows: list[lil_matrix] = []
    lower: list[float] = []
    upper: list[float] = []

    def push(matrix: lil_matrix, lo, hi, count: int = 1) -> None:
        rows.append(matrix)
        lower.extend(list(np.atleast_1d(lo)) if np.ndim(lo) else [lo] * count)
        upper.extend(list(np.atleast_1d(hi)) if np.ndim(hi) else [hi] * count)

    # --- budget, floor and ceiling -------------------------------------------
    m = lil_matrix((1, n_vars))
    m[0, :n_y] = marginal_cost.reshape(-1)
    push(m, min_spend_fraction * budget, budget)

    # --- slot ordering: y[k+1] <= y[k] ---------------------------------------
    m = lil_matrix((n * (S - 1), n_vars))
    for k in range(S - 1):
        for i in range(n):
            m[k * n + i, (k + 1) * n + i] = 1
            m[k * n + i, k * n + i] = -1
    push(m, -np.inf, 0, n * (S - 1))

    # --- availability: y[i,k] = 0 for k >= available_i ------------------------
    m = lil_matrix((1, n_vars))
    for k in range(S):
        for i in np.where(available <= k)[0]:
            m[0, k * n + i] = 1
    push(m, 0, 0)

    # --- exposures per pool: E[p] = sum of value over its cells ---------------
    m = lil_matrix((n_pools, n_vars))
    for k in range(S):
        for i in range(n):
            m[pool_of_cell[i], k * n + i] -= value[k, i]
    for p in range(n_pools):
        m[p, off_e + p] = 1
    push(m, 0, 0, n_pools)

    # --- reach: R[p] <= min(E[p], P[p]), EXACTLY -------------------------------
    # min() of two linear functions is concave, so as an upper bound on a maximized
    # variable it is exactly two linear constraints. No tangent envelope, no lambda, no
    # approximation error — and, decisively, the quantity the solver maximizes is then the
    # SAME quantity the package reports and the validator recomputes.
    #
    # This replaces the handoff's tangent construction of P x (1 - exp(-lambda E / P)).
    # That curve is a different function from min(E, P), so maximizing it does not maximize
    # the reported reach: measured on the canonical brief, the tangent objective returned
    # 141,501-157,869 reached where this returns the true optimum for the same budget. The
    # curve survives as a reported diagnostic (pooled.curve_reach), which is all decision 1
    # asked of it.
    m = lil_matrix((n_pools, n_vars))
    for p in range(n_pools):
        m[p, off_r + p] = 1
        m[p, off_e + p] = -1
    push(m, -np.inf, 0, n_pools)  # R - E <= 0

    m = lil_matrix((n_pools, n_vars))
    for p in range(n_pools):
        m[p, off_r + p] = 1
    push(m, -np.inf, population.tolist(), n_pools)  # R <= P

    # --- screen indicators ----------------------------------------------------
    # CHANGE 1. z[s] = 1 iff at least one cell on screen s is bought. Both directions are
    # needed: y <= z stops a screen being used without being counted, z <= sum(y) stops the
    # solver setting z=1 for free to satisfy a minimum it did not actually meet.
    m = lil_matrix((n, n_vars))
    for i in range(n):
        m[i, i] = 1
        m[i, off_z + screen_of_cell[i]] = -1
    push(m, -np.inf, 0, n)

    m = lil_matrix((n_screens, n_vars))
    for i in range(n):
        m[screen_of_cell[i], i] = -1
    for s in range(n_screens):
        m[s, off_z + s] = 1
    push(m, -np.inf, 0, n_screens)

    if exact_screens is not None:
        m = lil_matrix((1, n_vars))
        m[0, off_z : off_z + n_screens] = 1
        push(m, exact_screens, exact_screens)
    elif min_screens or max_screens:
        m = lil_matrix((1, n_vars))
        m[0, off_z : off_z + n_screens] = 1
        push(m, min_screens or 0, max_screens or np.inf)

    # --- hard slots per SCREEN per day ----------------------------------------
    # Summed over the screen's cells and all slot levels, so a screen bought in two time
    # blocks spends its allowance across both. A per-cell bound cannot say this; see the
    # module docstring for the brief that this got wrong.
    if max_slots_per_screen_per_day is not None:
        m = lil_matrix((n_screens, n_vars))
        for k in range(S):
            for i in range(n):
                m[screen_of_cell[i], k * n + i] = 1
        push(m, -np.inf, max_slots_per_screen_per_day, n_screens)

    # --- hard distinct-unit coverage: u <= sum(y1 in unit), sum(u) >= min_units ------
    if n_units:
        m = lil_matrix((n_units, n_vars))
        for j, (_label, _unit, mask) in enumerate(units):
            m[j, off_u + j] = 1
            for i in np.where(mask)[0]:
                m[j, i] = -1
        push(m, -np.inf, 0, n_units)

        cursor = 0
        for _label, spec in (unit_coverage or {}).items():
            width = len(spec["masks"])
            m = lil_matrix((1, n_vars))
            m[0, off_u + cursor : off_u + cursor + width] = 1
            push(m, spec["min_units"], np.inf)
            cursor += width

    # --- elastic coverage: sum(y1 in group) + slack >= k ----------------------
    for j, (_label, spec) in enumerate(groups):
        m = lil_matrix((1, n_vars))
        for i in np.where(spec["mask"])[0]:
            m[0, i] = 1
        m[0, off_c + j] = 1
        push(m, spec["min"], np.inf)

    # --- elastic wear-out cap: E[p] - cap*R[p] - w[p] <= 0 --------------------
    m = lil_matrix((n_pools, n_vars))
    for p in range(n_pools):
        m[p, off_e + p] = 1
        m[p, off_r + p] = -cap
        m[p, off_w + p] = -1
    push(m, -np.inf, 0, n_pools)

    constraints = LinearConstraint(vstack(rows).tocsc(), np.array(lower), np.array(upper))

    # --- objective ------------------------------------------------------------
    # Normalised by natural ceilings so the profile weights are true shares. Raw reach is
    # O(1e5) and exposures O(1e6), so unnormalised a nominal 70/30 split delivered 28/72.
    total_population = max(float(np.nansum(population)), 1.0)
    per_reach = 1.0 / total_population
    per_exposure = 1.0 / (total_population * C.EFFECTIVE_FREQUENCY_FLOOR)

    c = np.zeros(n_vars)
    c[:n_y] = -profile["w_freq"] * value.reshape(-1) * per_exposure
    if profile["w_conv"]:
        conv = cand["conv_fit"].fillna(0.0).to_numpy()
        c[:n_y] -= profile["w_conv"] * (value * conv[None, :]).reshape(-1) * per_exposure
    c[off_r : off_r + n_pools] = -profile["w_reach"] * per_reach
    # Cost as a tie-breaker only: at equal audience prefer the cheaper package, and stop
    # buying once there is nobody left to reach. Without it the solver is indifferent to
    # spending the balance on repetition, because cost otherwise appears only as a
    # constraint and never in the objective.
    c[:n_y] += C.COST_TIE_BREAKER * marginal_cost.reshape(-1) / max(budget, 1.0)
    c[off_c : off_c + n_groups] = C.COVERAGE_PENALTY
    c[off_w : off_w + n_pools] = C.WEAR_OUT_PENALTY * per_exposure

    integrality = np.zeros(n_vars)
    integrality[:off_e] = 1  # y, z and the unit-coverage indicators are binary
    ub = np.full(n_vars, np.inf)
    ub[:off_e] = 1

    debug(
        f"MILP problem: {n_vars:,} vars ({n_y:,} slot binaries + {n_screens:,} screen "
        f"binaries + {n_units} coverage units + {3 * n_pools:,} pool continuous + "
        f"{n_groups} elastic groups), {len(lower):,} constraint rows, "
        f"spend_floor={min_spend_fraction:.0%} wear_out_cap={cap:.0f} "
        f"slots_per_screen_per_day<="
        f"{'unbounded' if max_slots_per_screen_per_day is None else max_slots_per_screen_per_day}"
    )

    t0 = time.perf_counter()
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), ub),
        options={"time_limit": time_limit, "mip_rel_gap": C.MIP_REL_GAP},
    )
    elapsed = time.perf_counter() - t0

    if result.x is None:
        # Not an error on its own — the caller re-solves without the spend floor or the
        # exact screen count to work out which constraint actually binds.
        debug(
            f"MILP found nothing in {elapsed:.2f}s at spend_floor="
            f"{min_spend_fraction:.0%}: {result.message}"
        )
        return SolveOutcome(
            status="infeasible",
            plan=None,
            objective=objective,
            detail=str(result.message),
            min_spend_fraction=min_spend_fraction,
            log=log + [f"min_spend_fraction={min_spend_fraction:.2f}: {result.message}"],
        )

    debug(f"MILP solved in {elapsed:.2f}s: success={result.success} ({result.message})")

    return _read_solution(
        result,
        cand=cand,
        objective=objective,
        value=value,
        total_cost=total_cost,
        days=days,
        pool_keys=pool_keys,
        population=population,
        groups=groups,
        offsets=(n_y, off_e, off_c, off_w),
        counts=(n, S, n_pools, n_groups),
        min_spend_fraction=min_spend_fraction,
        cap=cap,
        slot_cap=max_slots_per_screen_per_day,
        log=log,
    )


def _coefficients(cand: pd.DataFrame, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginal value and marginal cost of the k-th slot, shape (S, n).

    CHANGE 4: `exposures_per_slot_per_day` already carries the loop-pass and viewability
    factors from `exposure.py`, so value is simply that figure times the flight length,
    identical for every slot. The bundle applied those factors here, which put
    `ScreenEconomics` and the package in different units.
    """
    S = C.SLOTS_PER_CELL
    per_slot = cand["exposures_per_slot_per_day"].to_numpy(dtype=float) * days
    value = np.tile(per_slot, (S, 1))

    price = cand["price"].to_numpy(dtype=float)
    discounts = np.array([cand[f"disc{k + 1}"].to_numpy(dtype=float) for k in range(S)])
    total = np.array([(k + 1) * price * discounts[k] for k in range(S)]) * days
    marginal = np.vstack([total[0]] + [total[k] - total[k - 1] for k in range(1, S)])
    return value, marginal, total


def _read_solution(
    result,
    *,
    cand: pd.DataFrame,
    objective: str,
    value: np.ndarray,
    total_cost: np.ndarray,
    days: int,
    pool_keys: list[str],
    population: np.ndarray,
    groups: list,
    offsets: tuple[int, ...],
    counts: tuple[int, ...],
    min_spend_fraction: float,
    cap: float,
    slot_cap: int | None,
    log: list[str],
) -> SolveOutcome:
    n_y, off_e, off_c, off_w = offsets
    n, S, n_pools, n_groups = counts

    y = result.x[:n_y].round().astype(int).reshape(S, n)
    slots = y.sum(axis=0)

    plan = cand.copy()
    plan["slots"] = slots
    plan = plan[plan.slots > 0].copy()
    if plan.empty:
        error("MILP returned a solution that buys nothing — every cell got zero slots")
        return SolveOutcome(
            status="infeasible",
            plan=None,
            objective=objective,
            detail="Solver returned an empty allocation.",
            min_spend_fraction=min_spend_fraction,
            log=log,
        )

    positions = plan.index.to_numpy()
    plan["line_cost"] = [total_cost[s - 1, i] for i, s in zip(positions, plan.slots, strict=True)]
    plan["viewed_exposures"] = [value[0, i] * s for i, s in zip(positions, plan.slots, strict=True)]
    plan = plan.reset_index(drop=True)

    # Observed depth, re-derived from the plan rather than assumed from the bound. The
    # per-screen figure is the one the constraint is about and the one the package reports.
    busiest_screen = int(plan.groupby("screen_id")["slots"].sum().max())
    busiest_cell = int(plan.slots.max())
    if slot_cap is not None and busiest_screen > slot_cap:
        error(
            f"MILP slot cap breached: busiest screen carries {busiest_screen} slots/day "
            f"against a cap of {slot_cap} — the per-screen constraint did not bind"
        )
        raise ConstraintBreachError(
            f"busiest screen carries {busiest_screen} slots/day against a hard cap of "
            f"{slot_cap}. Refusing to report a package that breaches a declared constraint."
        )

    exposures = result.x[off_e : off_e + n_pools]
    curve = pooled.curve_reach(exposures, population)

    # The guard. No lambda dependency: reach can exceed neither the exposures bought nor
    # the people available, whatever the saturation constant is. A violation means the pool
    # indexing disagrees with the coefficients — the one failure mode that yields a
    # confident wrong audience number rather than an obviously broken one.
    ceiling = min(float(exposures.sum()), float(population.sum()))
    if float(curve.sum()) > ceiling + max(1.0, abs(ceiling) * 1e-6):
        error(
            f"MILP reach accounting broken: curve reach {curve.sum():,.0f} > ceiling "
            f"{ceiling:,.0f} (exposures {exposures.sum():,.0f}, population "
            f"{population.sum():,.0f}) — pool indexing disagrees with the coefficients"
        )
        raise ReachAccountingError(
            f"curve reach {curve.sum():,.0f} exceeds min(exposures {exposures.sum():,.0f}, "
            f"population {population.sum():,.0f}). Pool bookkeeping is wrong; refusing to "
            f"report a package."
        )

    shortfall = {
        groups[j][0]: float(result.x[off_c + j])
        for j in range(n_groups)
        if result.x[off_c + j] > 0.01
    }
    wear_out = result.x[off_w : off_w + n_pools]
    status = "optimal" if result.success else "feasible"

    pool_table = pd.DataFrame(
        {
            "pool_key": pool_keys,
            "population": population,
            "exposures": exposures,
            "curve_reach": curve,
        }
    )
    pool_table = pool_table[pool_table.exposures > 1].sort_values("curve_reach", ascending=False)

    spend = float(plan.line_cost.sum())
    log = log + [
        f"solver=HiGHS via scipy.optimize.milp, mip_rel_gap={C.MIP_REL_GAP}",
        f"status={status} ({result.message})",
        f"min_spend_fraction={min_spend_fraction:.2f}",
        (
            f"wear_out_frequency_cap={cap:.0f} exposures per person "
            f"({C.WEAR_OUT_STACKING_MULTIPLE}x the {days}-day floor of "
            f"{exposure_floor_per_person(days):.0f})"
        ),
        f"allocated_screens={int(plan.screen_id.nunique())}",
        f"allocated_cells={len(plan)}",
        (
            f"slots_per_screen_per_day: cap="
            f"{'none' if slot_cap is None else slot_cap} busiest_screen={busiest_screen} "
            f"busiest_cell={busiest_cell} (cap binds per SCREEN across time blocks)"
        ),
        (
            f"audience_pool_blocks_touched={int((exposures > 1).sum())} "
            f"(pool x time block; physical pools are counted separately below)"
        ),
        f"spend={spend:,.2f}",
        f"gross_impressions_viewed={plan.viewed_exposures.sum():,.0f}",
        (
            f"curve_reach_diagnostic={curve.sum():,.0f} "
            f"(lambda={C.REACH_LAMBDA}, ASSUMED — not the reported reach)"
        ),
    ]
    info(
        f"MILP {status}: {int(plan.screen_id.nunique())} screens / {len(plan)} cells, "
        f"spend {spend:,.2f}, exposures {plan.viewed_exposures.sum():,.0f}, "
        f"pool-blocks touched {int((exposures > 1).sum())}, "
        f"curve_reach_diagnostic {curve.sum():,.0f} (lambda={C.REACH_LAMBDA}, ASSUMED)"
    )
    if shortfall:
        info(f"MILP coverage shortfall: {shortfall}")

    if wear_out.sum() > 1:
        debug(
            f"MILP wear-out cap breached by {wear_out.sum():,.0f} exposures across "
            f"{int((wear_out > 1).sum())} pool(s) at cap {cap:.0f} — a hard constraint "
            f"forced the depth"
        )
        log.append(
            f"wear_out_exposures_over_cap={wear_out.sum():,.0f} across "
            f"{int((wear_out > 1).sum())} pool(s) at a cap of {cap:.0f} exposures per "
            f"reached person ({C.WEAR_OUT_STACKING_MULTIPLE}x the {days}-day flight's "
            f"unavoidable floor of {exposure_floor_per_person(days):.0f}). A hard constraint "
            f"forced the breach — it is not a free choice at this penalty."
        )

    return SolveOutcome(
        status=status,
        plan=plan,
        objective=objective,
        detail=str(result.message),
        spend=spend,
        exposures=float(plan.viewed_exposures.sum()),
        curve_reach_diagnostic=float(curve.sum()),
        pools_used=int((exposures > 1).sum()),
        screens_used=int(plan.screen_id.nunique()),
        min_spend_fraction=min_spend_fraction,
        coverage_shortfall=shortfall,
        wear_out_exposures_over_cap=float(wear_out.sum()),
        wear_out_pools=int((wear_out > 1).sum()),
        max_slots_per_screen_per_day=slot_cap,
        slots_on_busiest_screen=busiest_screen,
        slots_in_busiest_cell=busiest_cell,
        pool_table=pool_table,
        log=log,
    )
