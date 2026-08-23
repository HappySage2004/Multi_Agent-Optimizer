"""Tools for the OR / OPTIMIZATION AGENT.

A thin wrapper. The MILP lives in `app/optimize/` (formulation in `solver.py`, pooled reach
in `pooled.py`, input validation in `contract.py`, the exposure model in `exposure.py`).
This module maps run state onto that solver and its result back onto the Pydantic
contracts. No decision logic here.

================================ REACH ACCOUNTING ================================
This is the definition of reach the system reports, and it is NOT the solver's internal
one. Keep it that way.

Exposures and reach are different quantities and the gap is large. A time block is a 4-hour
window in which all 6 rotation slots cycle continuously, so holding k slots puts the
creative on k of every 6 loop passes: viewed exposures are LINEAR in slots and scale with
slots x days.

Reach does not. `reachable_daily_audience` is the distinct people who LOOK at a screen's
POOL during that block on a typical day — and screens sharing a `pool_key` (one stop, or
one corridor) see the SAME people. Buying more slots, more days, or more screens at the
same stop raises frequency against that same audience.

    reach = SUM over (pool_key, time_block) of
                min( gross viewed exposures bought in that group,
                     that group's reachable daily audience )

The min() is what makes reach saturate, and it guarantees reach <= exposures for any flight
length. Both sides are in VIEWED units: capping viewed exposures at the undiscounted crowd
would let a saturated plan claim every passer-by when only ~35% of them look.

Why this and not the solver's saturation curve: the curve needs `REACH_LAMBDA`, which is
ASSUMED with no ground truth in the 14 CSVs. A client-facing number that moves with an
unmeasurable constant is worse than a lambda-free bound, and a second implementation of the
same lambda-dependent formula inside the validator would validate nothing about lambda —
the only real unknown. The curve's value rides along as `curve_reach_diagnostic`.

`validation._reach_checks` recomputes the min() independently. Two agreeing implementations
is the point, so do not import one into the other.
===============================================================================
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.tools import tool

from app.config import get_settings
from app.logging_utils import debug, error, info
from app.models.campaign import CampaignSpec
from app.models.economics import ScreenEconomics
from app.models.optimization import (
    Allocation,
    InfeasibilityReport,
    OptimizationResult,
    OptimizedPackage,
)
from app.models.screens import ScreenCandidate
from app.optimize import config as C
from app.optimize import contract, solver
from app.services import run_state
from app.services.artifact_store import read_models
from app.tools import coerce
from app.tools.coerce import ArgumentError

ECONOMICS_KIND = "screen_economics"
CANDIDATES_KIND = "screen_candidates"

# Which quantity each campaign goal is scored on for `objective_value`. The solver's own
# objective is a weighted blend (`solver.PROFILES`); this is the single headline number.
_OBJECTIVE_QUANTITY: dict[str, str] = {
    "reach": "deduplicated_reach",
    "awareness": "gross_impressions_viewed",
    "frequency": "gross_impressions_viewed",
    "conversion": "deduplicated_reach",
}

SOLVER_NOTICE = (
    "Allocation is a MILP (HiGHS via scipy). Pooled reach enters as the exact bound "
    "R <= min(exposures, reachable audience) — the same definition reported below — solved "
    f"to a {C.MIP_REL_GAP:.0%} relative gap. A status of 'feasible' rather than 'optimal' "
    "means a valid plan within that gap, not an error."
)


@tool
def optimize_package(run_id: str, slots_per_day_cap: int | str | None = None) -> dict:
    """Select the inventory package that best serves the campaign objective.

    Consumes `screen_economics` and stores an OptimizationResult on the run. Returns
    either a package summary or an infeasibility report with reason codes and
    relaxation options — never a fabricated package.

    The slot ceiling comes off the RUN, from the brief's own
    `hard_constraints["max_slots_per_day"]`, and is reported back in `slot_structure`.
    Do not pass `slots_per_day_cap` to satisfy a brief — a constraint routed through a
    tool argument is a constraint that eventually arrives wrong, and one that arrives
    wrong here ships as an over-delivery the client is contractually owed.

    Args:
        run_id: Handle for the campaign run.
        slots_per_day_cap: Exploration override only. May TIGHTEN the run's cap, never
            widen it. Leave unset in normal operation.
    """
    if not run_state.exists(run_id):
        error(f"STAGE 5 optimize_package called with unknown run_id={run_id!r}")
        return run_state.unknown_run(run_id, tool="optimize_package")

    if (blocked := run_state.missing_prerequisite(run_id, ECONOMICS_KIND)) is not None:
        error(f"STAGE 5 blocked: {blocked['detail']}")
        return blocked

    # Bounded before `resolve_slot_cap` sees it: the override may only ever tighten the
    # brief's own cap, and a value outside 1-6 is not a slot count at all. Clamping an
    # exploration knob is right where rejecting a brief-declared cap is right — one is our
    # search choice, the other is a client commitment.
    argument_notes: list[str] = []
    try:
        slots_per_day_cap, cap_note = coerce.clamp_int(
            slots_per_day_cap, field="slots_per_day_cap", low=1, high=C.SLOTS_PER_CELL
        )
    except ArgumentError as exc:
        error(f"STAGE 5 optimize_package rejected an argument: {exc}")
        return {"status": "invalid", "errors": str(exc)}
    if cap_note:
        argument_notes.append(cap_note)

    spec = run_state.get_spec(run_id)
    economics, candidates = _load_inputs(run_id)

    try:
        slot_cap = contract.resolve_slot_cap(spec, slots_per_day_cap)
    except contract.ContractError as exc:
        error(f"STAGE 5 slot cap unusable run_id={run_id}: {exc}")
        return _fail(
            run_id,
            ["CONFLICTING_HARD_CONSTRAINTS"],
            str(exc),
            [f"Record max_slots_per_day as a whole number of slots, 1-{C.SLOTS_PER_CELL}"],
            log=[],
        )

    if not economics:
        error(f"STAGE 5 screen_economics artifact is empty on run_id={run_id}")
        return _fail(
            run_id,
            ["NO_CANDIDATES"],
            "No screen economics available to optimize over.",
            ["Broaden the requested geography", "Relax hard constraints"],
            log=[],
        )

    debug(
        f"STAGE 5 optimizing {len(economics)} priced lines, goal={spec.optimization_goal}, "
        f"budget={spec.budget:,.0f}, slots_per_screen_per_day<={slot_cap.limit} "
        f"({slot_cap.source})"
    )
    if slot_cap.declared:
        info(
            f"STAGE 5 brief declares max_slots_per_day={slot_cap.limit} — enforced as a hard "
            f"per-SCREEN-per-day constraint across time blocks, and re-derived by the "
            f"validation layer"
        )

    try:
        frame, notes = contract.build_candidate_frame(economics, candidates, spec, slot_cap)
    except contract.ContractError as exc:
        error(f"STAGE 5 contract violation run_id={run_id}: {exc}")
        return _fail(
            run_id,
            ["CONFLICTING_HARD_CONSTRAINTS"],
            str(exc),
            ["Relax the conflicting hard constraint", "Re-price the required time blocks"],
            log=[],
        )

    if frame.empty:
        error(
            f"STAGE 5 no purchasable cell survived contract construction on "
            f"run_id={run_id} ({len(economics)} priced lines in)"
        )
        return _fail(
            run_id,
            ["INSUFFICIENT_INVENTORY"],
            f"None of the {len(economics)} priced screen/time-block lines is purchasable "
            f"for the requested window with a modelled audience.",
            [
                "Shift or shorten the campaign window",
                "Reduce slots per day",
                "Add time blocks",
                "Broaden the requested geography",
            ],
            log=notes,
        )

    # A minimum budget utilisation is honoured only when the BRIEF declares one. There is
    # no default floor: inventing one makes the solver spend leftover budget on repetition,
    # which on a reach brief buys no additional people at all.
    declared_utilization = spec.hard_constraints.get("min_budget_utilization")
    outcome = _solve(
        spec,
        frame,
        objective=spec.optimization_goal,
        notes=notes,
        slot_cap=slot_cap,
        min_spend_fraction=float(declared_utilization) if declared_utilization else None,
    )

    # A declared utilisation floor is HARD, so if it is what blocks the solve, say so with
    # the figure that is actually achievable rather than relaxing it silently.
    if outcome.plan is None and declared_utilization:
        debug(
            f"STAGE 5 no plan under min_budget_utilization="
            f"{float(declared_utilization):.0%}; re-solving without the floor to find out "
            f"whether it is the binding constraint"
        )
        unfloored = _solve(
            spec, frame, objective=spec.optimization_goal, notes=notes, slot_cap=slot_cap
        )
        if unfloored.plan is not None:
            return _fail(
                run_id,
                ["CONFLICTING_HARD_CONSTRAINTS", "BUDGET_CONSTRAINT"],
                f"min_budget_utilization={float(declared_utilization):.0%} cannot be met: "
                f"the best plan under the other constraints spends "
                f"{unfloored.spend / spec.budget:.0%} of budget "
                f"({unfloored.spend:,.2f}). Audience saturation, not price, is what stops "
                f"it spending more.",
                [
                    f"Reduce min_budget_utilization to {unfloored.spend / spec.budget:.2f}",
                    "Widen the geography so there is more distinct audience to buy",
                    "Extend the flight",
                ],
                log=outcome.log,
            )

    # Diagnose an exact-screen-count failure by re-solving without it: if a package exists
    # at a lower count, the constraint is what binds, not the budget.
    if outcome.plan is None and spec.requested_num_screens is not None:
        debug(
            f"STAGE 5 no plan at exactly {spec.requested_num_screens} screens; re-solving "
            f"without the count to find out whether it is the binding constraint"
        )
        relaxed = _solve(
            spec,
            frame,
            objective=spec.optimization_goal,
            notes=notes,
            slot_cap=slot_cap,
            exact=None,
        )
        if relaxed.plan is not None:
            return _fail(
                run_id,
                ["TOO_MANY_SCREENS_REQUESTED", "BUDGET_CONSTRAINT"],
                f"Requested exactly {spec.requested_num_screens} screens, but only "
                f"{relaxed.screens_used} fit within the {spec.budget:,.2f} budget over "
                f"{spec.duration_days} days.",
                [
                    f"Reduce requested screens to {relaxed.screens_used}",
                    "Increase budget",
                    "Shorten the campaign duration",
                    "Allow fewer slots per day",
                ],
                log=outcome.log,
            )

    if outcome.plan is None:
        return _fail(run_id, *_diagnose(spec, frame, outcome, slot_cap, notes), log=outcome.log)

    result = _to_result(spec, outcome, economics, slot_cap)
    run_state.set_optimization(run_id, result)
    pkg = result.package

    info(
        f"STAGE 5 package ready [{pkg.optimization_method}]: "
        f"{len(pkg.screen_ids)} screens, cost {pkg.total_cost:,.2f} "
        f"({pkg.budget_utilization:.1%} of budget), reach {pkg.expected_reach:,.0f}, "
        f"viewed exposures {pkg.gross_impressions_viewed:,.0f}, "
        f"frequency {pkg.expected_frequency:.2f}"
    )

    payload = {
        "status": result.status,
        "optimization_method": pkg.optimization_method,
        "objective": spec.optimization_goal,
        "objective_quantity": _OBJECTIVE_QUANTITY.get(spec.optimization_goal, "deduplicated_reach"),
        "objective_value": round(pkg.objective_value, 2),
        "screens_selected": len(pkg.screen_ids),
        "allocations": len(pkg.allocations),
        "total_cost": round(pkg.total_cost, 2),
        "budget_utilization": round(pkg.budget_utilization, 4),
        "gross_impressions_viewed": round(pkg.gross_impressions_viewed, 0),
        "expected_reach": round(pkg.expected_reach, 0),
        "expected_frequency": round(pkg.expected_frequency, 3),
        "cost_per_person_reached": (
            round(pkg.total_cost / pkg.expected_reach, 4) if pkg.expected_reach else None
        ),
        "audience_pools": _physical_pools(pkg, economics),
        "audience_pool_blocks": outcome.pools_used,
        "budget_unspent": round(spec.budget - pkg.total_cost, 2),
        "curve_reach_diagnostic": round(pkg.curve_reach_diagnostic or 0.0, 0),
        "reach_basis": (
            "deduplicated by pool_key x time block, capped at each pool's reachable daily "
            "audience — never the sum of exposures"
        ),
        "slot_structure": {
            **slot_cap.describe(),
            "slots_on_busiest_screen": outcome.slots_on_busiest_screen,
            "slots_in_busiest_cell": outcome.slots_in_busiest_cell,
        },
        "constraint_status": pkg.constraint_status,
        "unmet_coverage": pkg.unmet_coverage,
        "screen_type_mix": _mix_report(spec, pkg, candidates, economics, frame, notes),
        "solver_log": result.solver_log,
        # Non-empty means an argument you passed was bounded. Report the applied figure.
        "argument_notes": argument_notes,
        "caveat": SOLVER_NOTICE,
    }
    if (warning := _wear_out_warning(pkg, spec)) is not None:
        payload["wear_out_warning"] = warning
        info(
            f"STAGE 5 wear-out: {pkg.expected_frequency:.0f} viewed exposures per person "
            f"reached, against a {spec.duration_days}-day floor of "
            f"{solver.exposure_floor_per_person(spec.duration_days):.0f} and a cap of "
            f"{solver.wear_out_frequency_cap(spec.duration_days):.0f}"
        )
    if (
        finding := _unspent_budget_finding(pkg, spec, outcome, slot_cap, economics, frame, notes)
    ) is not None:
        payload["budget_finding"] = finding
        info(
            f"STAGE 5 budget unspent: {spec.budget - pkg.total_cost:,.0f} of "
            f"{spec.budget:,.0f} ({1 - pkg.budget_utilization:.0%}) — "
            + (
                f"the brief's {slot_cap.limit}-slot cap binds on depth"
                if slot_cap.declared and outcome.slots_on_busiest_screen >= slot_cap.limit
                else "audience saturation is binding, not money"
            )
        )
    if pkg.unmet_coverage:
        info(f"STAGE 5 unmet coverage: {pkg.unmet_coverage}")

    # The brief's screen-type mix. Reported on every run; a FINDING only when a requested
    # type is missing, because that is the case a rep must not be allowed to miss.
    mix = payload["screen_type_mix"]
    pool_by_type = (
        frame.groupby("screen_type")["screen_id"].nunique().to_dict()
        if "screen_type" in frame.columns
        else {}
    )
    if (
        mix_finding := _mix_finding(spec, mix["delivered_screens_by_type"], pool_by_type)
    ) is not None:
        payload["mix_finding"] = mix_finding
        error(
            f"STAGE 5 requested screen_type_mix {spec.screen_type_mix} NOT fully delivered: "
            f"package holds {mix['delivered_screens_by_type']}, pool held {pool_by_type}"
        )
    elif spec.screen_type_mix:
        info(
            f"STAGE 5 screen_type_mix {spec.screen_type_mix} honoured: "
            f"{mix['delivered_screens_by_type']}"
            + (
                f" (cost {mix['reach_cost_of_the_mix']:,.0f} reach against an unmixed plan)"
                if mix.get("reach_cost_of_the_mix")
                else ""
            )
        )
    return payload


@tool
def compare_objectives(run_id: str, objectives: list[str] | str | None = None) -> dict:
    """Solve the same brief under different objectives and return the plans side by side.

    Use this when the brief blends goals ("launch awareness and drive footfall") so the
    trade-off is presented to the client as a choice rather than resolved silently. Does
    not change the stored package — `optimize_package` owns that.

    Args:
        run_id: Handle for the campaign run.
        objectives: Which objectives to compare, as a LIST — e.g. ["reach", "awareness"].
            Valid values are reach, frequency, awareness and conversion. Defaults to reach
            and awareness.
    """
    if not run_state.exists(run_id):
        error(f"STAGE 5 compare_objectives called with unknown run_id={run_id!r}")
        return run_state.unknown_run(run_id, tool="compare_objectives")

    try:
        objectives = coerce.as_str_list(
            objectives,
            field="objectives",
            vocabulary=tuple(sorted(solver.PROFILES)),
            example='["reach", "awareness"]',
        )
    except ArgumentError as exc:
        error(f"STAGE 5 compare_objectives rejected an argument: {exc}")
        return {"status": "invalid", "errors": str(exc)}

    if (blocked := run_state.missing_prerequisite(run_id, ECONOMICS_KIND)) is not None:
        error(f"STAGE 5 objective comparison blocked: {blocked['detail']}")
        return blocked

    spec = run_state.get_spec(run_id)
    economics, candidates = _load_inputs(run_id)
    if not economics:
        error(f"STAGE 5 objective comparison: nothing priced on run_id={run_id}")
        return {"status": "no_candidates", "detail": "Nothing priced to compare."}

    try:
        slot_cap = contract.resolve_slot_cap(spec)
        frame, notes = contract.build_candidate_frame(economics, candidates, spec, slot_cap)
    except contract.ContractError as exc:
        error(f"STAGE 5 objective comparison contract violation run_id={run_id}: {exc}")
        return {"status": "error", "detail": str(exc)}
    if frame.empty:
        error(f"STAGE 5 objective comparison: no purchasable cells on run_id={run_id}")
        return {"status": "infeasible", "detail": "No purchasable cells."}

    # Already validated against `solver.PROFILES` by the coercion above, so an unknown
    # objective never reaches here — it comes back as a recoverable `invalid` result
    # instead of being dropped from a comparison the agent then reports as complete.
    requested = objectives or ["reach", "awareness"]
    debug(
        f"STAGE 5 comparing objectives {requested} on {len(frame)} cells "
        f"(one MILP solve per objective)"
    )
    comparison: list[dict] = []
    withheld: list[dict] = []

    for objective in requested:
        outcome = _solve(spec, frame, objective=objective, notes=notes, slot_cap=slot_cap)
        if outcome.plan is None:
            info(f"STAGE 5 objective '{objective}' infeasible: {outcome.detail}")
            withheld.append({"objective": objective, "reason": f"infeasible: {outcome.detail}"})
            continue

        allocations = _allocations(spec, outcome)
        exposures, reach, frequency, pools = _package_metrics(allocations, economics)
        row = {
            "objective": objective,
            "spend": round(outcome.spend, 2),
            "budget_unspent": round(spec.budget - outcome.spend, 2),
            "screens": outcome.screens_used,
            "audience_pools": pools,
            "gross_impressions_viewed": round(exposures, 0),
            "expected_reach": round(reach, 0),
            "expected_frequency": round(frequency, 2),
            "exposures_per_person_floor": round(
                solver.exposure_floor_per_person(spec.duration_days), 1
            ),
            "cost_per_person_reached": round(outcome.spend / reach, 4) if reach else None,
        }

        # The wear-out gate. A plan whose exposures per reached person exceed the cap is not
        # offered as an option, because there is no mechanism in this system to bring it
        # down: flight duration is fixed by the brief and daily allocation is constant, so
        # the floor is LOOP_PASSES_PER_TRIP / 6 x days regardless of allocation. It is
        # reported with its measured frequency rather than dropped, so the reader can see
        # what was excluded and why.
        cap = solver.wear_out_frequency_cap(spec.duration_days)
        if frequency > cap:
            info(
                f"STAGE 5 objective '{objective}' withheld: frequency {frequency:.0f} "
                f"exceeds the wear-out cap of {cap:.0f} for a {spec.duration_days}-day "
                f"flight — reported with its figures, not offered as an option"
            )
            withheld.append(
                {
                    "objective": objective,
                    "reason": (
                        f"withheld: {frequency:.0f} exposures per person reached exceeds the "
                        f"wear-out cap of {cap:.0f} for a {spec.duration_days}-day flight "
                        f"({C.WEAR_OUT_STACKING_MULTIPLE}x its unavoidable floor of "
                        f"{solver.exposure_floor_per_person(spec.duration_days):.0f}). That "
                        f"much repetition is stacking, not flight length, so it is not "
                        f"offered as an option."
                    ),
                    "measured": row,
                }
            )
            continue
        debug(
            f"STAGE 5 objective '{objective}': {outcome.screens_used} screens, "
            f"spend {outcome.spend:,.2f}, reach {reach:,.0f}, "
            f"exposures {exposures:,.0f}, frequency {frequency:.2f}"
        )
        comparison.append(row)

    info(
        f"STAGE 5 objective comparison ready: {len(comparison)} plan(s) offered, "
        f"{len(withheld)} withheld"
    )
    return {
        "status": "ok",
        "slot_structure": slot_cap.describe(),
        "comparison": comparison,
        "withheld": withheld,
        "note": (
            "expected_reach is distinct people; gross_impressions_viewed is exposures. The "
            "trade-off between breadth and repetition is a media planner's judgement — "
            "present it, do not resolve it silently."
        ),
        "caveat": SOLVER_NOTICE,
    }


# =============================================================================
# RUN STATE -> SOLVER
# =============================================================================


def _load_inputs(run_id: str) -> tuple[list[ScreenEconomics], dict[str, ScreenCandidate]]:
    """Both artifacts. Price and availability live on one, audience facts on the other."""
    economics = read_models(run_state.require_artifact(run_id, ECONOMICS_KIND), ScreenEconomics)
    candidates: dict[str, ScreenCandidate] = {}
    try:
        rows = read_models(run_state.require_artifact(run_id, CANDIDATES_KIND), ScreenCandidate)
        candidates = {c.screen_id: c for c in rows}
    except KeyError:
        # The economics artifact cannot exist without candidates, so this is unreachable in
        # practice — but an empty dict would silently drop every line, so it is not defaulted.
        error(f"STAGE 5 screen_candidates missing on run_id={run_id}")
    return economics, candidates


def _solve(
    spec: CampaignSpec,
    frame,
    *,
    objective: str,
    notes: list[str],
    slot_cap: contract.SlotCap,
    exact: int | None | object = ...,
    min_spend_fraction: float | None = None,
    slot_cap_limit: int | None | object = ...,
    mix: list[str] | None | object = ...,
) -> solver.SolveOutcome:
    """Translate spec constraints into solver arguments and solve.

    `slot_cap_limit` overrides the cap for a diagnostic re-solve only — passing None asks
    "would a package exist without this cap?", which is how the report can name the cap as
    the binding constraint instead of blaming the budget.
    """
    settings = get_settings()
    hc = spec.hard_constraints

    exact_screens = spec.requested_num_screens if exact is ... else exact
    max_screens = hc.get("max_screens") or settings.max_screens_in_package

    # The brief's screen-type mix, as ELASTIC coverage: one group per requested type, at
    # least one cell each. Elastic on purpose — see `_mix_finding`. `COVERAGE_PENALTY` is a
    # tenth of the total reachable population per unit of shortfall, far above what one
    # screen's marginal reach can earn, so the solver honours the mix unless a HARD
    # constraint leaves it no choice, and then yields and reports instead of failing.
    # `mix=[]` drops the coverage rows for a diagnostic re-solve — "what would this brief
    # reach if we ignored the requested mix?" — which is how the cost of honouring it is
    # reported rather than asserted.
    requested_mix = spec.screen_type_mix if mix is ... else (mix or [])
    coverage = None
    if requested_mix:
        masks = {f"screen_type:{t}": (frame["screen_type"] == t).to_numpy() for t in requested_mix}
        # A type with nothing in the pool cannot be covered by any plan. Asking anyway
        # would spend the penalty on an unsatisfiable row and bias the whole solve; the
        # absence is reported from the pool composition instead, which is where it happened.
        coverage = {label: {"mask": m, "min": 1} for label, m in masks.items() if m.any()}
        if absent := [label for label, m in masks.items() if not m.any()]:
            info(
                f"STAGE 5 screen_type_mix {[a.split(':')[1] for a in absent]} has no priced "
                f"cell in the candidate pool — no plan can include it, so it is reported "
                f"rather than penalized"
            )
        coverage = coverage or None

    unit_coverage = None
    if (min_zones := hc.get("min_zone_coverage")) is not None:
        zones = [z for z in frame["zone_id"].dropna().unique()]
        unit_coverage = {
            "zones": {
                "masks": {z: (frame["zone_id"] == z).to_numpy() for z in zones},
                "min_units": int(min_zones),
            }
        }

    outcome = solver.solve(
        frame,
        budget=spec.budget,
        days=spec.duration_days,
        objective=objective,
        min_screens=int(hc.get("min_screens") or 0),
        max_screens=int(max_screens),
        exact_screens=exact_screens,
        coverage=coverage,
        unit_coverage=unit_coverage,
        time_limit=float(settings.solver_time_limit_seconds),
        min_spend_fraction=min_spend_fraction,
        max_slots_per_screen_per_day=(
            slot_cap.limit if slot_cap_limit is ... else slot_cap_limit  # type: ignore[arg-type]
        ),
    )
    outcome.log = notes + outcome.log
    return outcome


# =============================================================================
# SOLVER -> CONTRACTS
# =============================================================================


def _allocations(spec: CampaignSpec, outcome: solver.SolveOutcome) -> list[Allocation]:
    """Solver plan rows -> Allocation contracts. `expected_revenue` is recomputed from the
    line's own price so it cannot drift from `price_per_slot_per_day x slots x days`."""
    return [
        Allocation(
            screen_id=row.screen_id,
            time_block_id=str(row.time_block_id),
            slots_per_day=int(row.slots),
            duration_days=spec.duration_days,
            price_per_slot_per_day=float(row.price),
            viewed_exposures=float(row.viewed_exposures),
            expected_revenue=float(row.price) * int(row.slots) * spec.duration_days,
        )
        for row in outcome.plan.itertuples(index=False)
    ]


def _to_result(
    spec: CampaignSpec,
    outcome: solver.SolveOutcome,
    economics: list[ScreenEconomics],
    slot_cap: contract.SlotCap,
) -> OptimizationResult:
    allocations = _allocations(spec, outcome)
    exposures, reach, frequency, pools = _package_metrics(allocations, economics)

    quantity = _OBJECTIVE_QUANTITY.get(spec.optimization_goal, "deduplicated_reach")
    objective_value = reach if quantity == "deduplicated_reach" else exposures

    # Cost is recomputed from the allocations rather than taken from the solver, so the
    # figure reported is the one the validator will independently re-derive.
    total_cost = sum(a.line_cost for a in allocations)

    package = OptimizedPackage(
        allocations=allocations,
        total_cost=round(total_cost, 2),
        gross_impressions_viewed=exposures,
        expected_reach=reach,
        expected_frequency=frequency,
        budget_utilization=total_cost / spec.budget if spec.budget else 0.0,
        constraint_status={
            "budget": total_cost <= spec.budget,
            "inventory": True,
            "geography": True,
            "campaign_dates": all(a.duration_days <= spec.duration_days for a in allocations),
            "requested_num_screens": (
                spec.requested_num_screens is None
                or len({a.screen_id for a in allocations}) == spec.requested_num_screens
            ),
            "coverage": not outcome.coverage_shortfall,
            # Only reported when the BRIEF declared it — a default cap is our choice, not a
            # commitment, and listing it here would imply the client asked for it.
            **(
                {"max_slots_per_day": outcome.slots_on_busiest_screen <= slot_cap.limit}
                if slot_cap.declared
                else {}
            ),
        },
        objective_value=objective_value,
        optimization_method=(
            f"milp_highs_pooled_reach_min[{outcome.objective},gap<={C.MIP_REL_GAP:.0%},"
            f"{outcome.status}]"
        ),
        curve_reach_diagnostic=outcome.curve_reach_diagnostic,
        unmet_coverage={k: round(v, 2) for k, v in outcome.coverage_shortfall.items()},
        wear_out_exposures_over_cap=outcome.wear_out_exposures_over_cap,
    )

    return OptimizationResult(
        status=outcome.status,
        package=package,
        solver_log=outcome.log
        + [
            (
                "reported_reach_definition=min(gross viewed exposures, reachable "
                "audience) per (pool, block) — lambda-free"
            ),
            f"deduplicated_reach={reach:,.0f}",
            f"distinct_audience_pools={pools}",
            f"frequency={frequency:.2f}",
            f"objective_value={objective_value:,.0f} ({quantity})",
            (
                f"slots_per_screen_per_day<={slot_cap.limit} ({slot_cap.source}); "
                f"busiest screen carries {outcome.slots_on_busiest_screen}"
            ),
        ],
    )


def _pool_of(line: ScreenEconomics) -> str:
    """The audience group a line competes within.

    Falls back to the screen's own id when no pool_key came through, which makes the screen
    its own pool — conservative in the right direction: it never merges audiences that were
    not proven to overlap.
    """
    return line.pool_key or line.screen_id


def _pool_ceiling(line: ScreenEconomics) -> float:
    """The reach ceiling for the line's POOL, not for the line.

    A vehicle's `reachable_daily_audience` is its share of the corridor, so capping a
    corridor's reach against it understates the corridor by the vehicle count — measured up
    to ~9x, and it made `curve_reach_bounded` fail on any package containing mobile
    inventory (132,724 against 14,682 on one brief). The pool figure is published on the
    contract; the fallback is only for artifacts written before that field existed.
    """
    return line.pool_reachable_daily_audience or line.reachable_daily_audience


def _package_metrics(
    allocations: list[Allocation], economics: list[ScreenEconomics]
) -> tuple[float, float, float, int]:
    """Gross viewed exposures, deduplicated reach, implied frequency, distinct pools.

    Reach sums each (pool, block) group's bought exposure capped at that group's reachable
    daily audience. See this module's REACH ACCOUNTING note. The validator recomputes this
    independently — keep the two definitions in step, and do not import one into the other.

    This is NOT part of the solver. Replacing the formulation does not touch it.
    """
    lookup = {(e.screen_id, str(e.time_block_id)): e for e in economics}

    exposures = sum(a.viewed_exposures for a in allocations)
    grouped: dict[tuple[str, str], float] = defaultdict(float)
    caps: dict[tuple[str, str], float] = {}
    for a in allocations:
        line = lookup.get((a.screen_id, str(a.time_block_id)))
        if line is None:
            continue
        key = (_pool_of(line), str(a.time_block_id))
        grouped[key] += a.viewed_exposures
        caps[key] = max(caps.get(key, 0.0), _pool_ceiling(line))

    reach = sum(min(gross, caps.get(key, 0.0)) for key, gross in grouped.items())
    frequency = (exposures / reach) if reach > 0 else 0.0
    pools = len({key[0] for key in grouped})
    return exposures, reach, frequency, pools


# =============================================================================
# INFEASIBILITY
# =============================================================================


def _diagnose(
    spec: CampaignSpec,
    frame,
    outcome: solver.SolveOutcome,
    slot_cap: contract.SlotCap,
    notes: list[str],
) -> tuple[list[str], str, list[str]]:
    """Reason codes, explanation and relaxations for a solve that found nothing."""
    # A declared slot cap is checked FIRST, by re-solving without it. A 1-slot cap removes
    # up to five sixths of a screen's purchasable depth, so it can make a brief infeasible
    # that the budget alone would have covered — and blaming the budget for it sends the rep
    # to ask for money that would not help. The cap is never widened to manufacture a
    # package; the relaxation is offered to the human with the figure attached.
    if slot_cap.declared:
        debug(
            f"STAGE 5 no plan at max_slots_per_day={slot_cap.limit}; re-solving uncapped to "
            f"find out whether the declared cap is the binding constraint"
        )
        uncapped = _solve(
            spec,
            frame,
            objective=spec.optimization_goal,
            notes=notes,
            slot_cap=slot_cap,
            slot_cap_limit=None,
        )
        if uncapped.plan is not None:
            return (
                ["CONFLICTING_HARD_CONSTRAINTS", "INSUFFICIENT_INVENTORY"],
                (
                    f"The brief's max_slots_per_day={slot_cap.limit} (per screen per day, "
                    f"across time blocks) cannot be met alongside the other constraints. "
                    f"Without it a package exists using {uncapped.screens_used} screens at "
                    f"{uncapped.spend:,.2f}, with up to "
                    f"{uncapped.slots_on_busiest_screen} slots/day on its busiest screen. "
                    f"Budget is not the binding constraint here."
                ),
                [
                    f"Raise max_slots_per_day to {uncapped.slots_on_busiest_screen}",
                    "Broaden the geography so the reach can come from more screens",
                    "Relax the screen-count or coverage constraint instead",
                ],
            )

    cheapest = float((frame["price"] * spec.duration_days).min())
    if cheapest > spec.budget:
        return (
            ["BUDGET_CONSTRAINT"],
            (
                f"Budget {spec.budget:,.2f} does not cover a single screen for "
                f"{spec.duration_days} days — the cheapest line costs {cheapest:,.2f}."
            ),
            [
                f"Increase budget to at least {cheapest:,.0f}",
                "Shorten the campaign duration",
                "Allow lower-cost screen types or off-peak time blocks",
            ],
        )

    hc = spec.hard_constraints
    if hc.get("min_zone_coverage") is not None:
        zones = int(frame["zone_id"].nunique())
        if zones < int(hc["min_zone_coverage"]):
            return (
                ["CONFLICTING_HARD_CONSTRAINTS", "GEOGRAPHY_UNAVAILABLE"],
                (
                    f"min_zone_coverage={hc['min_zone_coverage']} cannot be met: the "
                    f"candidate pool spans only {zones} zone(s)."
                ),
                [
                    f"Reduce min_zone_coverage to {zones}",
                    "Broaden the requested geography",
                ],
            )

    if (min_screens := hc.get("min_screens")) is not None:
        return (
            ["CONFLICTING_HARD_CONSTRAINTS", "BUDGET_CONSTRAINT"],
            (
                f"No package satisfies min_screens={min_screens} within the "
                f"{spec.budget:,.2f} budget over {spec.duration_days} days."
            ),
            ["Reduce min_screens", "Increase budget", "Shorten the campaign duration"],
        )

    return (
        ["BUDGET_CONSTRAINT"],
        (
            f"No feasible package within the {spec.budget:,.2f} budget over "
            f"{spec.duration_days} days. Solver: {outcome.detail}"
        ),
        [
            "Increase budget",
            "Shorten the campaign duration",
            "Relax hard constraints",
            "Broaden the requested geography",
        ],
    )


def _fail(
    run_id: str,
    reason_codes: list[str],
    explanation: str,
    relaxation_options: list[str],
    *,
    log: list[str],
) -> dict:
    """Store and return an InfeasibilityReport. Never a partial or speculative package."""
    result = OptimizationResult(
        status="infeasible",
        infeasibility=InfeasibilityReport(
            reason_codes=reason_codes,
            explanation=explanation,
            relaxation_options=relaxation_options,
        ),
        solver_log=log,
    )
    run_state.set_optimization(run_id, result)
    error(f"STAGE 5 INFEASIBLE run_id={run_id}: {reason_codes} — {explanation}")
    return {"status": "infeasible", **result.infeasibility.model_dump(), "solver_log": log}


def _physical_pools(package: OptimizedPackage, economics: list[ScreenEconomics]) -> int:
    """Distinct PHYSICAL audience pools, not pool x block cells. The two differ by ~2x on a
    two-block campaign and quoting the larger one overstates how spread the plan is."""
    lookup = {(e.screen_id, str(e.time_block_id)): e for e in economics}
    pools = {
        _pool_of(line)
        for a in package.allocations
        if (line := lookup.get((a.screen_id, str(a.time_block_id)))) is not None
    }
    return len(pools)


def _mix_report(
    spec: CampaignSpec,
    package: OptimizedPackage,
    candidates: dict[str, ScreenCandidate],
    economics: list[ScreenEconomics],
    frame=None,
    notes: list[str] | None = None,
) -> dict:
    """What the brief asked for, what it got, and what honouring it cost.

    `requested` empty means the brief named no mix — then this reports the delivered
    composition and nothing else. A package that is 100% one screen type is the single most
    consequential fact about it, so the composition is reported on EVERY run whether a mix
    was asked for or not.

    When a mix WAS requested, the plan is re-solved once with the coverage rows dropped, to
    report what the SAME pool would have reached without them. It costs one extra solve, only
    on briefs that asked for a mix. See `reach_cost_of_the_coverage_rule` for why that is
    deliberately not the same quantity as "what a single-type campaign would have reached".
    """
    delivered = _mix_delivered(package, candidates)
    report: dict = {
        "requested": spec.screen_type_mix,
        "delivered_screens_by_type": delivered,
    }
    if not spec.screen_type_mix:
        # No mix asked for. `honoured` is deliberately absent rather than False — there was
        # nothing to honour, and False reads as a failure to a model paraphrasing this.
        report["note"] = (
            "The brief named no screen-type mix, so the composition above is whatever the "
            "objective selected. Report it anyway: a package that is 100% one screen type "
            "is the most consequential fact about it."
        )
        return report

    report["honoured"] = all(delivered.get(t) for t in spec.screen_type_mix)
    report["enforcement"] = (
        "best effort, penalized and disclosed — never silently dropped, and never allowed "
        "to make the brief infeasible"
    )
    if frame is None:
        return report

    unmixed = _solve(
        spec,
        frame,
        objective=spec.optimization_goal,
        notes=notes or [],
        slot_cap=contract.resolve_slot_cap(spec),
        mix=[],
    )
    if unmixed.plan is not None:
        alt_reach = _package_metrics(_allocations(spec, unmixed), economics)[1]
        cost = alt_reach - package.expected_reach
        report["reach_if_coverage_dropped"] = round(alt_reach, 0)
        report["reach_cost_of_the_coverage_rule"] = round(cost, 0)
        # Named precisely, because there are TWO different costs here and conflating them
        # overstates this one. This is the cost of the SOLVER rule: same candidate pool,
        # coverage rows dropped. Measured at ~0 on the briefs tried, because the optimizer
        # buys the cheaper type anyway once it is in the pool — reach saturates per pool and
        # a cheap bus stop is a FRESH pool, so it is good value rather than a concession.
        #
        # The larger cost is upstream and is NOT this number: requesting a mix also
        # stratifies the stage-2 candidate cut, so the pool itself changes. A stratified
        # 120-screen pool reached 8,190 where a metro-only pool of the same size reached
        # 15,940. That belongs to candidate selection, not to the optimizer, and the OR
        # agent must not present it as the price of this rule.
        report["mix_cost_note"] = (
            f"Honouring the mix reaches {package.expected_reach:,.0f}; dropping the coverage "
            f"rule on the SAME candidate pool reaches {alt_reach:,.0f} "
            f"({cost:+,.0f}). Any larger gap against a single-type campaign comes from the "
            f"candidate pool being stratified upstream, not from this rule — do not "
            f"attribute it here."
        )
    return report


def _mix_delivered(
    package: OptimizedPackage, candidates: dict[str, ScreenCandidate]
) -> dict[str, int]:
    """Screens bought, by screen type. Counted from the package, never from the pool."""
    counts: dict[str, int] = {}
    for sid in package.screen_ids:
        if (c := candidates.get(sid)) is not None:
            counts[c.screen_type] = counts.get(c.screen_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _mix_finding(
    spec: CampaignSpec,
    delivered: dict[str, int],
    pool_by_type: dict[str, int],
) -> str | None:
    """State plainly when a screen type the brief asked for is not in the package.

    BEST EFFORT, DISCLOSED — the deliberate choice between two options. A hard requirement
    would guarantee the type or return an infeasibility; this one always ships a package and
    never lets a missing type pass in silence. The reason is that a mix costs measured reach
    (8,773 against 16,637 on one 60k brief, because bus stops draw smaller crowds), so
    whether to pay it is a media planner's call, not the solver's. What is NOT optional is
    saying so: `validation.screen_type_mix_disclosed` fails a package that drops a requested
    type without reporting it.

    The two causes read differently to a rep and are separated here. Nothing of that type in
    the candidate pool is an upstream fact — stage 2 ranks on relevance alone and a scarcer
    type can lose every slot — and no solver setting fixes it. A type that WAS in the pool
    and still went unbought means a hard constraint crowded it out.
    """
    if not spec.screen_type_mix:
        return None
    missing = [t for t in spec.screen_type_mix if not delivered.get(t)]
    if not missing:
        return None

    unpriced = [t for t in missing if not pool_by_type.get(t)]
    crowded = [t for t in missing if pool_by_type.get(t)]
    parts = [
        (
            f"The brief asked for {spec.screen_type_mix} and the package contains "
            f"{delivered or 'none of them'}."
        )
    ]
    if unpriced:
        parts.append(
            f"{unpriced} reached the optimizer with no priced inventory at all, so no plan "
            f"could have included them — that is a candidate-selection outcome upstream of "
            f"this stage, not a choice made here. Report it as a gap in the pool and say "
            f"what the pool did contain."
        )
    if crowded:
        parts.append(
            f"{crowded} WAS available and was still not bought, which means another hard "
            f"constraint (budget, screen count, zone coverage or the slot cap) left no room "
            f"for it. Name that trade-off rather than the screen type."
        )
    parts.append("Do not present this package as matching the requested mix.")
    return " ".join(parts)


def _wear_out_warning(package: OptimizedPackage, spec: CampaignSpec) -> str | None:
    """Disclose the exposure frequency whenever it is past useful, cap breach or not.

    Reported for the campaign's own stated objective — never used to substitute a different
    one. The client asked for this goal; the number comes with it.
    """
    floor = solver.exposure_floor_per_person(spec.duration_days)
    cap = solver.wear_out_frequency_cap(spec.duration_days)
    if package.expected_frequency <= C.EFFECTIVE_FREQUENCY_FLOOR:
        return None
    breached = package.expected_frequency > cap
    # The floor holds for a SATURATED pool: one slot on a pool the plan has filled delivers
    # LOOP_PASSES/6 per person per day. A corridor bought at partial vehicle coverage is not
    # saturated in that sense — its ceiling is the whole corridor while the exposures come
    # from the vehicles actually bought — so mobile-heavy packages legitimately sit BELOW the
    # floor. Asserting the floor unconditionally printed "cannot go below 40" next to a
    # measured 3.96, which is the kind of contradiction that destroys trust in every other
    # figure on the page.
    if package.expected_frequency < floor:
        return (
            f"This package delivers {package.expected_frequency:.0f} viewed exposures per "
            f"person reached, which is BELOW the {floor:.0f} a saturated pool would force on "
            f"a {spec.duration_days}-day flight. That means the plan is not saturating its "
            f"pools — characteristic of vehicle inventory, where a corridor's whole ridership "
            f"is reachable but only the vehicles bought carry the creative. Report the figure "
            f"as measured; do not describe it as a floor breach."
        )
    return (
        f"This package delivers {package.expected_frequency:.0f} viewed exposures per "
        f"person reached. A {spec.duration_days}-day flight cannot go below "
        f"~{floor:.0f} on a saturated pool — duration is the brief's, the minimum purchase "
        f"is one slot, and there is no flighting — so most of that is flight length rather "
        f"than selection. "
        + (
            f"It also exceeds the stacking cap of {cap:.0f}, which means a hard constraint "
            f"forced extra depth; report that."
            if breached
            else "It is within the stacking cap, so no depth was bought gratuitously."
        )
        + " There is no wear-out model here: state the figure, do not imply it was tuned."
    )


def _unspent_budget_finding(
    package: OptimizedPackage,
    spec: CampaignSpec,
    outcome: solver.SolveOutcome,
    slot_cap: contract.SlotCap,
    economics: list[ScreenEconomics],
    frame=None,
    notes: list[str] | None = None,
) -> str | None:
    """Say so when the plan could not absorb the budget, and name what stopped it.

    Two different causes, and they lead the rep to opposite actions. Audience saturation
    means the money genuinely cannot buy more people and should be returned or spent
    elsewhere. A declared slot cap that costs reach means relaxing it WOULD buy people,
    which is the client's decision and has to be put to them with the figure attached.

    Distinguished by re-solving without the cap, the same way the utilisation floor and the
    screen count are diagnosed — and the comparison is on REACH, not on spend. "Without the
    cap we would spend more" is not a finding: at equal reach the extra spend buys exactly
    the repetition the cap was asked for. Measured across three briefs and four cap levels,
    an uncapped solve never reached more people than a 1-slot one, so this is expected to
    report saturation AND say the cap is not the cause — which is why the two branches read
    so differently.

    Only runs when a cap is declared, is actually binding, and material budget is left, so
    the extra solve is off the normal path.
    """
    unspent = spec.budget - package.total_cost
    if unspent <= max(1.0, spec.budget * 0.02):
        return None

    preamble = (
        f"{unspent:,.0f} of the {spec.budget:,.0f} budget is unspent "
        f"({unspent / spec.budget:.0%}). "
    )
    saturation = (
        f"The binding constraint is audience saturation, not money: the plan already "
        f"reaches {package.expected_reach:,.0f} of the people available in this geography, "
        f"and further spend would buy repetition rather than reach. Options are to widen "
        f"the geography, extend the flight, or return the balance — not to pad the package."
    )

    cap_binds = slot_cap.declared and outcome.slots_on_busiest_screen >= slot_cap.limit
    if not cap_binds or frame is None:
        return preamble + saturation

    uncapped = _solve(
        spec,
        frame,
        objective=spec.optimization_goal,
        notes=notes or [],
        slot_cap=slot_cap,
        slot_cap_limit=None,
    )
    if uncapped.plan is None:
        return preamble + saturation

    uncapped_reach = _package_metrics(_allocations(spec, uncapped), economics)[1]
    gain = uncapped_reach - package.expected_reach
    if gain <= max(1.0, package.expected_reach * 0.01):
        # The cap binds on depth but costs no audience: the same people are reached either
        # way, so the balance is saturation and the cap is not the story. Saying otherwise
        # sends the rep to renegotiate a written constraint for nothing.
        return (
            preamble
            + saturation
            + f" The brief's {slot_cap.limit}-slot-per-screen cap is NOT what leaves the "
            f"balance: solved without it the same brief spends {uncapped.spend:,.2f} and "
            f"reaches {uncapped_reach:,.0f} against this plan's "
            f"{package.expected_reach:,.0f} — the extra spend buys repetition, which is "
            f"the thing the cap was asked for. Do not offer to relax it to absorb budget."
        )
    return (
        preamble + f"The brief's max_slots_per_day={slot_cap.limit} (per screen per day) is "
        f"what leaves it, and here it does cost audience: solved without the cap the same "
        f"brief spends {uncapped.spend:,.2f} across {uncapped.screens_used} screens and "
        f"reaches {uncapped_reach:,.0f}, against {package.expected_reach:,.0f} — "
        f"{gain:,.0f} more people ({gain / max(package.expected_reach, 1):.0%}). The cap was "
        f"honoured. Put that trade-off to the client with both figures; relaxing a "
        f"constraint they wrote is their decision, not ours."
    )


TOOLS = [optimize_package, compare_objectives]
