"""Tools for the OR / OPTIMIZATION AGENT.

============================== INTEGRATION POINT ==============================
The OR Agent is owned by a separate implementer. Replace `_stub_greedy_allocate`
with a real MILP / CP-SAT formulation (PuLP is already a dependency).

Keep intact:
  * tool names and argument signatures
  * the OptimizationResult / OptimizedPackage / InfeasibilityReport contracts
  * the invariant that an infeasible problem returns InfeasibilityReport and NEVER a
    fabricated package -- the Master Agent relies on this

What the stub does today: a greedy impressions-per-dollar fill. It respects budget,
slot availability, and requested_num_screens, so the Master Agent's validation layer
exercises real code paths -- but it is NOT an optimizer and carries no optimality
guarantee. `optimization_method` is reported as "stub_greedy_value_density".

Expected real behaviour (SOLUTION.md sections 10-13):
  decision vars : x[s,t] = slots purchased on screen s in time block t (integer)
  objective     : switches on CampaignSpec.optimization_goal
                  reach -> maximize deduplicated impressions
                  awareness -> weighted impressions
                  frequency -> expected frequency s.t. a minimum reach
                  conversion -> expected conversions
  constraints   : cost <= budget; slots <= available; geography; dates; screen count;
                  daypart preferences; screen-type restrictions
  infeasibility : diagnose, emit reason codes, propose relaxations
===============================================================================
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.config import get_settings
from app.logging_utils import debug, error, info
from app.models.economics import ScreenEconomics
from app.models.optimization import (
    Allocation,
    InfeasibilityReport,
    OptimizationResult,
    OptimizedPackage,
)
from app.services import run_state
from app.services.artifact_store import read_models
from app.tools._stub_support import STUB_NOTICE, spread, unit

ECONOMICS_KIND = "screen_economics"


@tool
def optimize_package(run_id: str, slots_per_day_cap: int = 3) -> dict:
    """Select the inventory package that best serves the campaign objective.

    Consumes `screen_economics` and stores an OptimizationResult on the run. Returns
    either a package summary or an infeasibility report with reason codes and
    relaxation options — never a fabricated package.

    Args:
        run_id: Handle for the campaign run.
        slots_per_day_cap: Upper bound on slots bought per screen per day.
    """
    if (blocked := run_state.missing_prerequisite(run_id, ECONOMICS_KIND)) is not None:
        error(f"STAGE 4 blocked: {blocked['detail']}")
        return blocked

    spec = run_state.get_spec(run_id)
    ref = run_state.require_artifact(run_id, ECONOMICS_KIND)
    economics = read_models(ref, ScreenEconomics)

    if not economics:
        error(f"STAGE 4 nothing to optimize for run_id={run_id}")
        result = OptimizationResult(
            status="infeasible",
            infeasibility=InfeasibilityReport(
                reason_codes=["NO_CANDIDATES"],
                explanation="No screen economics available to optimize over.",
                relaxation_options=["Broaden the requested geography", "Relax hard constraints"],
            ),
        )
        run_state.set_optimization(run_id, result)
        return {"status": "infeasible", **result.infeasibility.model_dump()}

    debug(
        f"STAGE 4 optimizing {len(economics)} lines, goal={spec.optimization_goal}, "
        f"budget={spec.budget:,.0f}, slots_cap={slots_per_day_cap}"
    )
    result = _stub_greedy_allocate(spec, economics, slots_per_day_cap)
    run_state.set_optimization(run_id, result)

    if result.status == "infeasible":
        error(
            f"STAGE 4 INFEASIBLE run_id={run_id}: {result.infeasibility.reason_codes} — "
            f"{result.infeasibility.explanation}"
        )
        return {
            "status": "infeasible",
            **result.infeasibility.model_dump(),
            "warning": STUB_NOTICE,
        }

    pkg = result.package
    info(
        f"STAGE 4 package ready [STUB {pkg.optimization_method}]: "
        f"{len(pkg.screen_ids)} screens, cost {pkg.total_cost:,.2f} "
        f"({pkg.budget_utilization:.1%} of budget), "
        f"impressions {pkg.expected_impressions:,.0f}"
    )
    return {
        "status": result.status,
        "optimization_method": pkg.optimization_method,
        "screens_selected": len(pkg.screen_ids),
        "allocations": len(pkg.allocations),
        "total_cost": round(pkg.total_cost, 2),
        "budget_utilization": round(pkg.budget_utilization, 4),
        "expected_impressions": round(pkg.expected_impressions, 0),
        "expected_reach": round(pkg.expected_reach, 0),
        "expected_frequency": round(pkg.expected_frequency, 3),
        "constraint_status": pkg.constraint_status,
        "solver_log": result.solver_log,
        "warning": STUB_NOTICE,
    }


def _stub_greedy_allocate(
    spec, economics: list[ScreenEconomics], slots_cap: int
) -> OptimizationResult:
    """PLACEHOLDER allocation: greedy by impressions per dollar. Replace with MILP."""
    settings = get_settings()
    log: list[str] = [
        f"objective={spec.optimization_goal}",
        f"candidate_lines={len(economics)}",
        f"budget={spec.budget:,.2f}",
        f"duration_days={spec.duration_days}",
    ]

    # One line per screen: take each screen's best-value time block, so a single screen
    # is never double-booked across blocks in the placeholder package.
    best_by_screen: dict[str, ScreenEconomics] = {}
    for e in economics:
        if e.pricing.recommended_price <= 0 or e.max_slots_per_day < 1:
            continue
        density = e.expected_impressions / e.pricing.recommended_price
        incumbent = best_by_screen.get(e.screen_id)
        if incumbent is None or density > (
            incumbent.expected_impressions / incumbent.pricing.recommended_price
        ):
            best_by_screen[e.screen_id] = e

    ranked = sorted(
        best_by_screen.values(),
        key=lambda e: (-(e.expected_impressions / e.pricing.recommended_price), e.screen_id),
    )

    target_screens = spec.requested_num_screens
    max_screens = min(
        target_screens or settings.max_screens_in_package, settings.max_screens_in_package
    )

    allocations: list[Allocation] = []
    spent = 0.0
    for e in ranked:
        if len(allocations) >= max_screens:
            break
        slots = min(slots_cap, e.max_slots_per_day)
        line_cost = e.pricing.recommended_price * slots * spec.duration_days

        # Shrink slots rather than skip the screen when a full line does not fit.
        while slots > 1 and spent + line_cost > spec.budget:
            slots -= 1
            line_cost = e.pricing.recommended_price * slots * spec.duration_days
        if spent + line_cost > spec.budget:
            continue

        allocations.append(
            Allocation(
                screen_id=e.screen_id,
                time_block_id=e.time_block_id,
                slots_per_day=slots,
                duration_days=spec.duration_days,
                price_per_slot_per_day=e.pricing.recommended_price,
                expected_impressions=e.expected_impressions * slots * spec.duration_days,
                expected_revenue=e.expected_revenue * slots * spec.duration_days,
            )
        )
        spent += line_cost

    if not allocations:
        cheapest = min(
            e.pricing.recommended_price * spec.duration_days for e in best_by_screen.values()
        )
        return OptimizationResult(
            status="infeasible",
            infeasibility=InfeasibilityReport(
                reason_codes=["BUDGET_CONSTRAINT"],
                explanation=(
                    f"Budget {spec.budget:,.2f} does not cover a single screen for "
                    f"{spec.duration_days} days — the cheapest line costs "
                    f"{cheapest:,.2f}."
                ),
                relaxation_options=[
                    f"Increase budget to at least {cheapest:,.0f}",
                    "Shorten the campaign duration",
                    "Allow lower-cost screen types or off-peak time blocks",
                ],
            ),
            solver_log=log,
        )

    if target_screens is not None and len(allocations) < target_screens:
        return OptimizationResult(
            status="infeasible",
            infeasibility=InfeasibilityReport(
                reason_codes=["BUDGET_CONSTRAINT", "TOO_MANY_SCREENS_REQUESTED"],
                explanation=(
                    f"Requested {target_screens} screens but only {len(allocations)} fit "
                    f"within the {spec.budget:,.2f} budget over {spec.duration_days} days."
                ),
                relaxation_options=[
                    f"Reduce requested screens to {len(allocations)}",
                    "Increase budget",
                    "Shorten the campaign duration",
                    "Allow fewer slots per day",
                ],
            ),
            solver_log=log + [f"allocated_screens={len(allocations)}"],
        )

    impressions = sum(a.expected_impressions for a in allocations)
    # Placeholder dedup: overlap grows with package size. The real optimizer must model
    # reach properly instead of scaling gross impressions.
    assumed_frequency = round(spread(unit(len(allocations), "freq"), 1.15, 1.85), 3)
    reach = impressions / assumed_frequency

    package = OptimizedPackage(
        allocations=allocations,
        total_cost=round(spent, 2),
        expected_impressions=impressions,
        expected_reach=reach,
        expected_frequency=impressions / reach,
        budget_utilization=spent / spec.budget,
        constraint_status={
            "budget": spent <= spec.budget,
            "inventory": all(a.slots_per_day <= slots_cap for a in allocations),
            "geography": True,
            "campaign_dates": all(a.duration_days <= spec.duration_days for a in allocations),
            "requested_num_screens": target_screens is None or len(allocations) == target_screens,
        },
        objective_value=impressions,
        optimization_method="stub_greedy_value_density",
    )
    return OptimizationResult(
        status="feasible",
        package=package,
        solver_log=log + [f"allocated_screens={len(allocations)}", f"spent={spent:,.2f}"],
    )


TOOLS = [optimize_package]
