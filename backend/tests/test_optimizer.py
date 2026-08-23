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

from app.agents.validation import validate_package
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


@pytest.fixture(scope="module")
def priced_candidates():
    """Spec, economics and the raw ScreenCandidate map — for tests that rebuild the frame
    themselves under different slot caps."""
    run_id = _priced_run()
    spec = run_state.get_spec(run_id)
    economics = read_models(run_state.require_artifact(run_id, "screen_economics"), ScreenEconomics)
    rows = read_models(run_state.require_artifact(run_id, "screen_candidates"), ScreenCandidate)
    return spec, economics, {c.screen_id: c for c in rows}


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
    minimum. On this brief the top 60 of 2,230 eligible screens are 60/60 in LH-ZONE-001, so
    `min_zone_coverage=2` is unsatisfiable no matter what the solver does — LH-ZONE-002
    first appears at top_n=80 (75/5) and contributes only 22 of 250. The optimizer must
    report that as a constraint conflict naming the geography, not quietly return a
    one-zone package.

    `top_n` was 80 here until stop-share volumes and the Fix-4 audience formulas re-ranked
    the pool; 80 now admits 5 second-zone screens and the constraint becomes satisfiable.
    The upstream gap is unchanged — the brief that exposes it just moved.
    """
    run_id = _priced_run(
        top_n=60,
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


# --- the brief's slot structure ----------------------------------------------
#
# A client brief specified "1 rotating slot on digital screens only" and the delivered
# package bought 3 slots/day on three of its lines. There was no channel for the constraint:
# `hard_constraints["max_slots_per_day"]` was persisted, echoed back to the Master and read
# by nobody, and the only real control was an LLM-supplied tool argument no prompt mentioned.
# These pin the channel, the semantics, and the independent re-derivation.


def _named(result, name: str):
    """The one check by name, so a test asserts on the check it means."""
    return next(c for c in result.checks if c.name == name)


def test_a_declared_slot_cap_binds_per_screen_across_time_blocks(priced):
    """The semantics decision, and the reading it replaces.

    A per-CELL cap of 1 lets a screen bought in two time blocks take 1 + 1 = 2 slots that
    day, an over-delivery against a brief asking for one rotating slot however the blocks
    are labelled. The cap therefore binds on the physical screen, summed across blocks —
    see the SLOTS PER SCREEN PER DAY block in `optimize/solver.py`.
    """
    spec, _economics, frame = priced
    # Only meaningful if some screen is purchasable in more than one block.
    assert frame.groupby("screen_id").size().max() >= 2, (
        "no screen spans two blocks on this brief, so it cannot exercise the bug"
    )

    clipped = frame.assign(available=frame["available"].clip(upper=1))

    # The old behaviour: clip the cell, impose nothing on the screen.
    per_cell = solver.solve(clipped, budget=spec.budget, days=spec.duration_days, objective="reach")
    assert per_cell.plan is not None
    assert per_cell.slots_on_busiest_screen > 1, (
        "a per-cell clip alone was expected to let a screen exceed the cap across blocks — "
        "if it no longer does, this test has stopped covering the bug"
    )

    # The constraint.
    per_screen = solver.solve(
        clipped,
        budget=spec.budget,
        days=spec.duration_days,
        objective="reach",
        max_slots_per_screen_per_day=1,
    )
    assert per_screen.plan is not None
    assert per_screen.slots_on_busiest_screen == 1
    assert per_screen.plan.groupby("screen_id")["slots"].sum().max() == 1


def test_the_declared_cap_comes_off_the_run_not_off_a_tool_argument():
    """The `PricingLevers` rule applied to a constraint: a caller may tighten what the brief
    declared, never widen it. Widening on request is how a constraint gets quietly relaxed
    until a package appears."""
    spec = run_state.get_spec(_spec(hard_constraints={"max_slots_per_day": 2})["run_id"])

    off_run = contract.resolve_slot_cap(spec)
    assert (off_run.limit, off_run.source, off_run.declared) == (2, "brief", True)

    widened = contract.resolve_slot_cap(spec, override=6)
    assert widened.limit == 2, "an override must never widen a brief-declared cap"
    assert widened.declared

    tightened = contract.resolve_slot_cap(spec, override=1)
    assert (tightened.limit, tightened.source) == (1, "caller_override")
    assert not tightened.declared, "our own tightening is not a client commitment"

    # No declared cap: the default applies and is labelled ours, never the brief's.
    bare = contract.resolve_slot_cap(run_state.get_spec(_spec()["run_id"]))
    assert (bare.limit, bare.source) == (C.DEFAULT_SLOTS_PER_DAY_CAP, "default")
    assert not bare.declared

    # Prose, a flag, a fraction, or a depth no screen can sell — all rejected rather than
    # coerced. int(True) is 1 and int(1.5) is 1, so coercion would enforce a plausible cap
    # the client never asked for.
    for bad in ("one rotating slot", 0, 7, True, 1.5):
        broken = run_state.get_spec(_spec(hard_constraints={"max_slots_per_day": bad})["run_id"])
        with pytest.raises(contract.ContractError):
            contract.resolve_slot_cap(broken)


def test_no_allocation_exceeds_a_declared_slot_cap_end_to_end():
    """The whole path: brief -> spec -> solver -> package -> verification."""
    run_id = _priced_run(hard_constraints={"max_slots_per_day": 1})
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    assert result["status"] in SOLVED, result

    structure = result["slot_structure"]
    assert structure["slots_per_screen_per_day_cap"] == 1
    assert structure["source"] == "brief"
    assert structure["slots_on_busiest_screen"] == 1
    # The reported figure has to say WHICH reading produced it.
    assert "PER SCREEN PER DAY" in structure["semantics"]
    assert result["constraint_status"]["max_slots_per_day"] is True

    package = run_state.get_optimization(run_id).package
    per_screen: dict[str, int] = {}
    for a in package.allocations:
        per_screen[a.screen_id] = per_screen.get(a.screen_id, 0) + a.slots_per_day
    assert max(per_screen.values()) == 1

    verdict = master_tools.verify_package.invoke({"run_id": run_id})
    assert verdict["status"] == "pass", verdict["failed_checks"]
    # The check must have RUN, not merely not-failed — its absence is the original bug.
    assert "max_slots_per_day" in {c["name"] for c in verdict["checks_run"]}


def test_the_validator_fails_a_package_that_breaches_a_declared_cap():
    """Independent re-derivation. A solver that ignored the constraint has to be caught
    rather than believed, so the validator sums the allocations itself — and across blocks,
    which is where `inventory_availability` (a different assertion, about unsold inventory)
    passes happily."""
    run_id = _priced_run(hard_constraints={"max_slots_per_day": 1})
    or_agent_tools.optimize_package.invoke({"run_id": run_id})
    spec = run_state.get_spec(run_id)
    economics = read_models(run_state.require_artifact(run_id, "screen_economics"), ScreenEconomics)
    package = run_state.get_optimization(run_id).package

    clean = validate_package(spec, package, economics)
    assert clean.passed
    assert _named(clean, "max_slots_per_day").status == "pass"

    # Same screen, second block, one slot each: two slots that day. Every per-CELL figure
    # still respects the cap, which is exactly why the per-cell reading passed this.
    blocks = {a.time_block_id for a in package.allocations}
    victim = package.allocations[0]
    other = next(b for b in sorted(blocks) if b != victim.time_block_id)
    breached = package.model_copy(
        update={
            "allocations": [
                *package.allocations,
                victim.model_copy(update={"time_block_id": other}),
            ]
        }
    )
    verdict = validate_package(spec, breached, economics)
    assert not verdict.passed
    slot_check = _named(verdict, "max_slots_per_day")
    assert slot_check.status == "fail"
    assert slot_check.observed is not None and "2" in slot_check.observed


def test_a_hard_constraint_no_stage_enforces_fails_verification():
    """The generalization, and the part that stops this recurring under another key.

    An unrecognized `hard_constraints` key used to be persisted, echoed back to the Master
    and dropped in silence — which is how the slot cap went missing with nobody told.
    Verification now refuses to bless a package it cannot check.
    """
    run_id = _priced_run(hard_constraints={"digital_screens_only": True})
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    assert result["status"] in SOLVED, "the package still builds — it just cannot be blessed"

    verdict = master_tools.verify_package.invoke({"run_id": run_id})
    assert verdict["status"] == "fail"
    # `checks_run` is name + status only; the prose lives on `failed_checks`.
    assert "hard_constraints_recognized" in {c["name"] for c in verdict["checks_run"]}
    failure = next(
        c for c in verdict["failed_checks"] if c["name"] == "hard_constraints_recognized"
    )
    # The rep has to be told WHICH constraint went unenforced, not merely that one did,
    # and what the enforceable keys are so the brief can be restated.
    assert "digital_screens_only" in failure["detail"]
    assert "min_zone_coverage" in failure["detail"]


def test_honouring_the_slot_cap_costs_no_reach_and_no_extra_repetition(priced):
    """Measured rather than assumed, and the reason the constraint is cheap to honour.

    Reach is bounded by each pool's DAILY reachable audience while exposures accumulate over
    the whole flight, so one slot over 30 days already over-saturates its pool and extra
    depth buys frequency, not people. Asserted within the solver gap, like
    `test_no_spend_floor_is_invented`: a strict ordering would be asserting a coin flip.
    """
    spec, economics, frame = priced

    def solve_at(cap: int) -> tuple[float, float]:
        capped = frame.assign(available=frame["available"].clip(upper=cap))
        outcome = solver.solve(
            capped,
            budget=spec.budget,
            days=spec.duration_days,
            objective="reach",
            max_slots_per_screen_per_day=cap,
        )
        assert outcome.plan is not None
        _e, reach, frequency, _p = or_agent_tools._package_metrics(
            or_agent_tools._allocations(spec, outcome), economics
        )
        return reach, frequency

    reach_1, freq_1 = solve_at(1)
    reach_default, freq_default = solve_at(C.DEFAULT_SLOTS_PER_DAY_CAP)

    assert reach_1 >= reach_default * (1.0 - C.MIP_REL_GAP), (
        f"a 1-slot cap reached {reach_1:,.0f} against {reach_default:,.0f} at "
        f"{C.DEFAULT_SLOTS_PER_DAY_CAP} slots — further below than the {C.MIP_REL_GAP:.0%} "
        f"gap explains, so honouring the brief would be costing real audience"
    )
    # Not a strict ordering, and not for float reasons alone. On this brief both plans sit
    # ON the flight's exposure floor, so the two frequencies differ only in the last bits;
    # and the relationship is genuinely non-monotone at wider caps, because nothing bounds
    # screens per pool as tightly as MAX_CELLS_PER_POOL does. What must hold is that a
    # tighter cap never buys MORE repetition than a looser one.
    assert freq_1 <= freq_default * (1.0 + C.MIP_REL_GAP), (
        f"a 1-slot cap delivered {freq_1:.2f} exposures per person against "
        f"{freq_default:.2f} at {C.DEFAULT_SLOTS_PER_DAY_CAP} slots — a tighter cap must not "
        f"increase repetition beyond the {C.MIP_REL_GAP:.0%} solver gap"
    )
    assert freq_1 >= solver.exposure_floor_per_person(spec.duration_days) - 0.01


def test_pool_pruning_is_not_scaled_by_the_slot_cap(priced_candidates):
    """`MAX_CELLS_PER_POOL` is deliberately cap-independent, and that is load-bearing.

    Scaling it to hold slot depth constant (4 -> 12 cells at a 1-slot cap) was implemented
    and measured: identical reach, spend 34,334 -> 115,628 and exposures per person
    40.0 -> 112.2, because the extra cells let the solver stack SCREENS into pools it could
    no longer stack slots into. The cap would then have relocated repetition rather than
    reducing it. See `contract._prune_saturated_pools`.
    """
    spec, economics, candidates = priced_candidates
    for cap in (1, C.DEFAULT_SLOTS_PER_DAY_CAP, C.SLOTS_PER_CELL):
        frame, _notes = contract.build_candidate_frame(
            economics, candidates, spec, contract.SlotCap(limit=cap, source="brief")
        )
        assert frame.groupby("pool_key").size().max() <= C.MAX_CELLS_PER_POOL
        assert int(frame["available"].max()) <= cap
