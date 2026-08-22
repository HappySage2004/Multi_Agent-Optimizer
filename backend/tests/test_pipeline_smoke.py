"""End-to-end pipeline exercise with no LLM in the loop.

Invokes each stage's tool directly in order, then asserts the Master Agent's validation
layer accepts the result. This is the regression net that had to keep passing while the OR
stage's greedy heuristic was replaced with a real MILP.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.services import run_state
from app.tools import master_tools, ml_agent_tools, or_agent_tools, relevance_tools

START = (date.today() + timedelta(days=14)).isoformat()

# A MILP solved to a relative gap reports "optimal" when it proved the bound and "feasible"
# when it stopped inside the gap. Both are real answers; the distinction is reported, not
# asserted away.
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
        "audience_age_min": 18,
        "audience_age_max": 34,
        "audience_commuter": True,
        "audience_terms": ["young_professionals", "commuters"],
        "industry_vertical": "technology",
        "original_query": "I have $50K for 30 days targeting young commuters downtown.",
    }
    payload.update(overrides)
    return master_tools.create_campaign_spec.invoke(payload)


def test_geography_resolution_is_conservative():
    out = master_tools.resolve_geography_terms.invoke(
        {"terms": ["Las Hackland", "Downtown Core", "LH-RT-B001", "eastern metro corridor"]}
    )
    assert out["city_ids"] == ["LH"]
    assert out["zone_ids"] == ["LH-ZONE-001"]
    assert out["corridor_ids"] == ["LH-RT-B001"]
    # A vague directional phrase must NOT be silently resolved to some plausible ID.
    assert out["unresolved"] == ["eastern metro corridor"]


def test_spec_rejects_invalid_budget_and_unknown_geography():
    assert _spec(budget=0)["status"] == "invalid"
    assert _spec(duration_days=0)["status"] == "invalid"
    assert _spec(zone_ids=["LH-ZONE-999"])["status"] == "invalid"


def test_full_pipeline_produces_a_package_that_passes_validation():
    created = _spec()
    assert created["status"] == "ok", created
    run_id = created["run_id"]

    inventory = relevance_tools.describe_inventory.invoke({"run_id": run_id})
    assert inventory["eligible_screens"] > 0

    candidates = relevance_tools.build_screen_candidates.invoke({"run_id": run_id})
    assert candidates["status"] == "ok"
    assert candidates["candidates_selected"] > 0

    economics = ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    assert economics["status"] == "ok"
    # Pricing is real: a computed band, real occupancy, and some lines sold out.
    assert economics["price_band"]["min"] > 0
    assert economics["lines_feasible"] > 0
    assert 0.0 <= economics["occupancy_mean"] <= 1.0

    optimized = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    # "optimal" is reachable now that this is a real solve; "feasible" means within the
    # solver's 5% gap. Both are valid; anything else is not.
    assert optimized["status"] in SOLVED, optimized
    assert optimized["total_cost"] <= 50_000.0

    verdict = master_tools.verify_package.invoke({"run_id": run_id})
    assert verdict["status"] == "pass", verdict["failed_checks"]

    # Nothing in the pipeline is a stub any more: relevance, pricing and optimization all
    # write provenance="computed". A stub reappearing here is a regression.
    assert verdict["stub_stages"] == []


def test_infeasible_budget_is_reported_not_papered_over():
    created = _spec(budget=25.0)
    run_id = created["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 20})
    ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})

    assert result["status"] == "infeasible"
    assert "BUDGET_CONSTRAINT" in result["reason_codes"]
    assert result["relaxation_options"]

    # And the Master Agent must refuse to validate a package that does not exist.
    verdict = master_tools.verify_package.invoke({"run_id": run_id})
    assert verdict["status"] == "infeasible"


def test_requested_screen_count_shortfall_is_infeasible():
    created = _spec(budget=6_000.0, requested_num_screens=40)
    run_id = created["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 60})
    ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    result = or_agent_tools.optimize_package.invoke({"run_id": run_id})

    assert result["status"] == "infeasible"
    assert "TOO_MANY_SCREENS_REQUESTED" in result["reason_codes"]


def test_validation_catches_a_tampered_package():
    """The validator must not trust the optimizer's own arithmetic."""
    created = _spec()
    run_id = created["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 40})
    ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    or_agent_tools.optimize_package.invoke({"run_id": run_id})

    result = run_state.get_optimization(run_id)
    result.package.total_cost = 1.0  # understated cost
    result.package.gross_impressions_viewed *= 10  # inflated exposures
    run_state.set_optimization(run_id, result)

    verdict = master_tools.verify_package.invoke({"run_id": run_id})
    assert verdict["status"] == "fail"
    failed = {c["name"] for c in verdict["failed_checks"]}
    assert "cost_reconciles" in failed
    assert "impressions_reconcile" in failed


def test_validation_catches_an_over_counted_reach():
    """Reach is the easiest number here to inflate — summing impressions over-counts ~23x.

    The validator must recompute it from the pool_key groups rather than trusting the
    optimizer, so passing gross impressions off as reach has to fail.
    """
    run_id = _spec()["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 40})
    ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    or_agent_tools.optimize_package.invoke({"run_id": run_id})

    result = run_state.get_optimization(run_id)
    honest_reach = result.package.expected_reach
    # Claim every gross viewed exposure was a distinct person.
    result.package.expected_reach = result.package.gross_impressions_viewed
    result.package.expected_frequency = 1.0
    run_state.set_optimization(run_id, result)

    verdict = master_tools.verify_package.invoke({"run_id": run_id})
    assert verdict["status"] == "fail"
    assert "reach_reconciles" in {c["name"] for c in verdict["failed_checks"]}
    # The honest figure really is much smaller — this is not a rounding-level check.
    assert honest_reach < result.package.gross_impressions_viewed


@pytest.mark.parametrize("goal", ["reach", "frequency", "awareness", "conversion"])
def test_every_optimization_goal_runs(goal):
    created = _spec(optimization_goal=goal)
    assert created["status"] == "ok"
    run_id = created["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 30})
    ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    assert or_agent_tools.optimize_package.invoke({"run_id": run_id})["status"] in SOLVED


def test_out_of_order_stages_return_a_recoverable_error():
    """A supervisor may delegate stages concurrently; dependent tools must not crash.

    Regression: the Master Agent once issued two stage delegations in the same turn, and
    the ML stage died on a KeyError instead of reporting what was missing.
    """
    run_id = _spec()["run_id"]

    # Stage 3 before stage 2.
    blocked = ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    assert blocked["status"] == "prerequisite_missing"
    assert blocked["missing_artifact"] == "screen_candidates"
    assert "build_screen_candidates" in blocked["detail"]

    # Stage 4 before stage 3.
    blocked = or_agent_tools.optimize_package.invoke({"run_id": run_id})
    assert blocked["status"] == "prerequisite_missing"
    assert blocked["missing_artifact"] == "screen_economics"
    assert "ml_agent" in blocked["detail"]

    # Running them in order still works.
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id, "top_n": 30})
    assert ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})["status"] == "ok"
    assert or_agent_tools.optimize_package.invoke({"run_id": run_id})["status"] in SOLVED
