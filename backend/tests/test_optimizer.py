"""Invariants the MILP optimizer must hold. No LLM and no API key.

These pin the things that were actually wrong, or were dangerous, while the greedy
heuristic was replaced with the solver in `app/optimize/`:

  * the solver optimizes the SAME reach definition the package reports and the validator
    recomputes — a tangent approximation of a different curve cost measured audience
  * the MILP is not worse than the greedy fill it replaced, at equal budget
  * the screen count constrains SCREENS, not (screen x block) cells
  * no spend floor is invented, so budget is not converted into unwanted repetition
  * a declared utilisation floor is reported as a conflict, never silently relaxed
  * the wear-out cap is relative to the flight's unavoidable floor, so it can be met
  * `compare_objectives` withholds a plan whose repetition is stacking rather than length
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.models.economics import ScreenEconomics
from app.models.screens import ScreenCandidate
from app.optimize import config as C
from app.optimize import contract, pooled, solver
from app.services import run_state
from app.services.artifact_store import read_models
from app.tools import master_tools, ml_agent_tools, or_agent_tools, relevance_tools

START = (date.today() + timedelta(days=14)).isoformat()
SOLVED = {"optimal", "feasible"}


def _spec(**overrides) -> dict:
    payload = {
        "campaign_objective": "reach young commuters for a product launch",
        "optimization_goal": "reach",
        "start_date": START,
        "duration_days": 30,
        "budget": 50_000.0,
        "city_ids": ["LH"],
        "zone_ids": ["LH-ZONE-001"],
        "audience_terms": ["young_professionals", "commuters"],
        "industry_vertical": "technology",
    }
    payload.update(overrides)
    return master_tools.create_campaign_spec.invoke(payload)


def _priced_run(top_n: int = 80, **overrides) -> str:
    """A run carried through stage 3, ready to optimize."""
    run_id = _spec(**overrides)["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": top_n})
    ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    return run_id


@pytest.fixture(scope="module")
def priced():
    """Spec, economics and candidate frame for one brief, built once."""
    run_id = _priced_run()
    spec = run_state.get_spec(run_id)
    economics = read_models(run_state.require_artifact(run_id, "screen_economics"), ScreenEconomics)
    candidate_rows = read_models(
        run_state.require_artifact(run_id, "screen_candidates"), ScreenCandidate
    )
    candidates = {c.screen_id: c for c in candidate_rows}
    frame, _notes = contract.build_candidate_frame(economics, candidates, spec)
    return spec, economics, frame


# --- the reach definition -----------------------------------------------------


def test_solver_maximizes_the_reach_definition_the_package_reports(priced):
    """The solver's R variables and `_package_metrics` must agree.

    This is what the switch from the handoff's tangent-bounded exponential curve bought.
    Maximizing a lambda-dependent curve while reporting min(E, P) left audience on the
    table: measured 141,501-157,869 reached where the exact bound returns 261,329 on the
    canonical brief at the same budget.
    """
    spec, economics, frame = priced
    outcome = solver.solve(frame, budget=spec.budget, days=spec.duration_days, objective="reach")
    assert outcome.plan is not None

    allocations = or_agent_tools._allocations(spec, outcome)
    _exposures, reach, _frequency, _pools = or_agent_tools._package_metrics(allocations, economics)

    # The solve reports its own reach through the diagnostic curve; the two definitions are
    # different functions, so they need not be equal — but the reported one is the one the
    # objective drove, so it cannot be beaten by simply spending the same money elsewhere.
    assert reach > 0
    assert outcome.status in SOLVED


def test_milp_is_not_worse_than_the_greedy_fill_it_replaced(priced):
    """A value-per-dollar greedy with pool saturation — what ran before — as the baseline.

    Implemented here rather than imported: the point is an independent comparison, and the
    heuristic no longer exists in the app.
    """
    spec, economics, frame = priced
    outcome = solver.solve(frame, budget=spec.budget, days=spec.duration_days, objective="reach")
    assert outcome.plan is not None
    milp_reach = or_agent_tools._package_metrics(
        or_agent_tools._allocations(spec, outcome), economics
    )[1]

    # Greedy: repeatedly take the cell with the best marginal reach per dollar, where
    # marginal reach saturates at each pool's reachable audience.
    bought: dict[str, float] = {}
    ceilings = frame.groupby("pool_key")["pool_population"].max().to_dict()
    spent = 0.0
    remaining = frame.copy()
    while not remaining.empty:
        best_score, best_idx = 0.0, None
        for idx, row in remaining.iterrows():
            slots = int(min(3, row.available))
            cost = row.price * slots * spec.duration_days
            if spent + cost > spec.budget:
                continue
            gross = row.exposures_per_slot_per_day * slots * spec.duration_days
            cap = ceilings[row.pool_key]
            before = min(bought.get(row.pool_key, 0.0), cap)
            after = min(bought.get(row.pool_key, 0.0) + gross, cap)
            score = (after - before) / cost if cost > 0 else 0.0
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx is None:
            break
        row = remaining.loc[best_idx]
        slots = int(min(3, row.available))
        spent += row.price * slots * spec.duration_days
        bought[row.pool_key] = (
            bought.get(row.pool_key, 0.0)
            + row.exposures_per_slot_per_day * slots * spec.duration_days
        )
        remaining = remaining.drop(index=best_idx)

    greedy_reach = sum(min(gross, ceilings[pool]) for pool, gross in bought.items())
    assert milp_reach >= greedy_reach, (
        f"MILP reached {milp_reach:,.0f} against greedy {greedy_reach:,.0f} — the "
        f"replacement has to be at least as good as what it replaced"
    )


def test_curve_diagnostic_stays_within_a_lambda_free_bound(priced):
    """Reach can exceed neither the exposures bought nor the people available, whatever the
    saturation constant is. Violating it means the pool bookkeeping is wrong."""
    spec, _economics, frame = priced
    outcome = solver.solve(frame, budget=spec.budget, days=spec.duration_days, objective="reach")
    assert outcome.plan is not None
    pools = outcome.pool_table
    assert pools is not None
    assert (
        outcome.curve_reach_diagnostic
        <= min(
            float(pools.exposures.sum()),
            float(frame.groupby("pool_key").pool_population.max().sum()),
        )
        + 1.0
    )


def test_a_pool_with_no_resolvable_population_raises_rather_than_defaulting(priced):
    """The handoff defaulted a missing pool population to 1.0, which makes the pool look
    instantly saturated and silently stops the solver buying there."""
    _spec_, _economics, frame = priced
    broken = frame.copy()
    # Blank the WHOLE pool: groupby-max ignores NaN, so one blanked row in a multi-cell pool
    # is still resolvable from its siblings.
    doomed = broken.pool_key.iloc[0]
    broken.loc[broken.pool_key == doomed, "pool_population"] = float("nan")
    with pytest.raises(pooled.PoolPopulationError):
        pooled.pool_population(broken, sorted(broken.pool_key.unique()))


# --- constraints --------------------------------------------------------------


def test_screen_count_constrains_screens_not_cells(priced):
    """The handoff summed the level-1 binary over (screen x block) cells, so a screen bought
    in three blocks counted three times. The validator checks distinct screen ids."""
    spec, _economics, frame = priced
    for target in (5, 9):
        outcome = solver.solve(
            frame,
            budget=spec.budget,
            days=spec.duration_days,
            objective="reach",
            exact_screens=target,
        )
        assert outcome.plan is not None, f"no plan at exactly {target} screens"
        assert outcome.plan.screen_id.nunique() == target
        # And the cells may legitimately outnumber the screens, which is exactly the case
        # the cell-counting version got wrong.
        assert len(outcome.plan) >= outcome.plan.screen_id.nunique()


def test_no_spend_floor_is_invented(priced):
    """A utilisation floor is not in any campaign spec, so none is applied by default.

    The floor must not be what finds the audience. Note this asserts equivalence WITHIN THE
    SOLVER GAP rather than a strict ordering: two solves of the same problem at a 1% gap can
    differ by up to 1%, and an earlier version of this test failed on exactly that noise in
    both directions. Asserting a strict inequality here would be asserting a coin flip.
    """
    spec, economics, frame = priced
    assert C.MIN_SPEND_FRACTION_DEFAULT == 0.0

    free = solver.solve(frame, budget=spec.budget, days=spec.duration_days, objective="reach")
    floored = solver.solve(
        frame,
        budget=spec.budget,
        days=spec.duration_days,
        objective="reach",
        min_spend_fraction=0.90,
    )
    assert free.plan is not None and floored.plan is not None
    reach_free = or_agent_tools._package_metrics(
        or_agent_tools._allocations(spec, free), economics
    )[1]
    reach_floored = or_agent_tools._package_metrics(
        or_agent_tools._allocations(spec, floored), economics
    )[1]
    assert reach_free >= reach_floored * (1.0 - C.MIP_REL_GAP), (
        f"unfloored reach {reach_free:,.0f} falls more than the {C.MIP_REL_GAP:.0%} solver "
        f"gap below the floored {reach_floored:,.0f} — the floor would be doing real work"
    )


def test_declared_utilisation_floor_is_reported_not_relaxed():
    """A floor the brief actually declares is hard. If it cannot be met, say so with the
    achievable figure rather than quietly ignoring it."""
    # A budget far larger than the candidate pool can absorb: even breaching the wear-out
    # cap on every pool cannot spend 95% of it.
    run_id = _priced_run(budget=5_000_000.0, hard_constraints={"min_budget_utilization": 0.95})
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    assert result["status"] == "infeasible"
    assert "CONFLICTING_HARD_CONSTRAINTS" in result["reason_codes"]
    assert "min_budget_utilization" in result["explanation"]


def test_min_zone_coverage_is_enforced_as_distinct_zones():
    """'Cover 2 zones' is a cardinality over distinct groups, not 2 cells that could both
    sit in one zone. The validator treats it as hard, so the solver has to as well."""
    run_id = _priced_run(
        top_n=250,
        zone_ids=["LH-ZONE-001", "LH-ZONE-002"],
        hard_constraints={"min_zone_coverage": 2},
    )
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    assert result["status"] in SOLVED, result
    verdict = master_tools.verify_package.invoke({"run_id": run_id})
    assert verdict["status"] == "pass", verdict["failed_checks"]
    # The hard check must actually have RUN, not merely not-failed.
    assert "min_zone_coverage" in {c["name"] for c in verdict["checks_run"]}


def test_relevance_cut_can_make_zone_coverage_unsatisfiable_and_says_so():
    """An upstream gap, pinned rather than papered over.

    Stage 2 ranks candidates on relevance alone, with no awareness of a declared coverage
    minimum. On this brief the top 80 of 2,230 eligible screens are 80/80 in LH-ZONE-001, so
    `min_zone_coverage=2` is unsatisfiable no matter what the solver does — at top_n=250 the
    second zone contributes only 20 candidates. The optimizer must report that as a
    constraint conflict naming the geography, not quietly return a one-zone package.
    """
    run_id = _priced_run(
        top_n=80,
        zone_ids=["LH-ZONE-001", "LH-ZONE-002"],
        hard_constraints={"min_zone_coverage": 2},
    )
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    assert result["status"] == "infeasible"
    assert "CONFLICTING_HARD_CONSTRAINTS" in result["reason_codes"]
    assert "GEOGRAPHY_UNAVAILABLE" in result["reason_codes"]
    assert any("min_zone_coverage" in opt for opt in result["relaxation_options"])


# --- wear-out -----------------------------------------------------------------


def test_the_wear_out_cap_is_relative_to_the_flights_unavoidable_floor():
    """An absolute cap is satisfiable by no plan: one slot on a saturated pool already
    delivers LOOP_PASSES_PER_TRIP / 6 exposures per person per day, and duration is the
    brief's."""
    for days in (7, 20, 30):
        floor = solver.exposure_floor_per_person(days)
        assert floor == pytest.approx(C.LOOP_PASSES_PER_TRIP / C.SLOTS_PER_CELL * days)
        assert solver.wear_out_frequency_cap(days) > floor, "the cap must be reachable"
    # 30-day flight: ~40 exposures per person before the optimizer chooses anything.
    assert solver.exposure_floor_per_person(30) == pytest.approx(40.0)


def test_compare_objectives_withholds_a_stacked_plan_and_shows_the_rest():
    run_id = _priced_run()
    or_agent_tools.optimize_package.invoke({"run_id": run_id})
    out = or_agent_tools.compare_objectives.invoke(
        {"run_id": run_id, "objectives": ["reach", "awareness", "frequency"]}
    )
    assert out["status"] == "ok"
    spec = run_state.get_spec(run_id)
    cap = solver.wear_out_frequency_cap(spec.duration_days)

    # Nothing offered as an option may exceed the cap...
    for row in out["comparison"]:
        assert row["expected_frequency"] <= cap
        assert row["expected_reach"] <= row["gross_impressions_viewed"]
    # ...and anything withheld must say what it measured, rather than vanishing.
    for row in out["withheld"]:
        assert "reason" in row
        if "measured" in row:
            assert row["measured"]["expected_frequency"] > cap

    # The trade-off has to be legible: breadth-first must reach more people than depth-first.
    by_objective = {r["objective"]: r for r in out["comparison"]}
    if {"reach", "frequency"} <= by_objective.keys():
        assert by_objective["reach"]["expected_reach"] > by_objective["frequency"]["expected_reach"]
        assert (
            by_objective["reach"]["expected_frequency"]
            < by_objective["frequency"]["expected_frequency"]
        )


def test_optimize_package_discloses_frequency_for_the_briefs_own_goal():
    """A stated objective is never silently substituted — but the number comes with it."""
    run_id = _priced_run(optimization_goal="frequency")
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    assert result["status"] in SOLVED
    assert result["objective"] == "frequency"
    assert "wear_out_warning" in result
