"""Tools owned by the MASTER AGENT.

These are deliberately NOT stubs. Brief intake, geography resolution and package
verification are the Master Agent's own responsibility and must stay deterministic: the
LLM decides when to call them, the code decides what the answer is.
"""

from __future__ import annotations

from datetime import date

from langchain_core.tools import tool
from pydantic import ValidationError

from app.agents.validation import validate_explanations, validate_package
from app.data.reference import resolve_geography, screen_facts
from app.logging_utils import debug, error, info
from app.models.campaign import AudienceTarget, CampaignSpec
from app.models.economics import ScreenEconomics
from app.services import run_state
from app.services.artifact_store import read_models


@tool
def resolve_geography_terms(terms: list[str]) -> dict:
    """Map natural-language places or ID-like strings onto real city/zone/corridor IDs.

    Call this before create_campaign_spec whenever the brief names places in prose.
    Unmatched terms come back in `unresolved` — never invent an ID for one; ask the user
    or record it in missing_information.

    Args:
        terms: Place names or IDs from the brief, e.g. ["Las Hackland", "Downtown Core"].
    """
    resolved, unresolved = resolve_geography(terms)
    debug(f"geography: {terms} -> {resolved}, unresolved={unresolved}")
    if unresolved:
        info(f"geography terms could not be resolved: {unresolved}")
    return {
        **resolved,
        "unresolved": unresolved,
        "hint": (
            "Zone names are city-scoped labels like 'Downtown Core'; corridor IDs look "
            "like 'LH-RT-B001'. Vague directional phrases ('eastern corridor') do not "
            "resolve on their own and need user clarification."
        )
        if unresolved
        else None,
    }


@tool
def create_campaign_spec(
    campaign_objective: str,
    optimization_goal: str,
    start_date: str,
    duration_days: int,
    budget: float,
    city_ids: list[str] | None = None,
    zone_ids: list[str] | None = None,
    corridor_ids: list[str] | None = None,
    industry_vertical: str | None = None,
    ad_type: str | None = None,
    audience_age_min: int | None = None,
    audience_age_max: int | None = None,
    audience_commuter: bool | None = None,
    requested_num_screens: int | None = None,
    preferred_time_blocks: list[str] | None = None,
    preferred_dayparts: list[str] | None = None,
    hard_constraints: dict | None = None,
    soft_preferences: dict | None = None,
    original_query: str | None = None,
    missing_information: list[str] | None = None,
    session_id: str | None = None,
) -> dict:
    """Validate and persist the normalized campaign brief, opening a new run.

    Every downstream tool takes the returned `run_id`. Validation is deterministic:
    budget and duration must be positive, geography IDs must exist, and at least one of
    city/zone/corridor must resolve. Do not invent values to satisfy the validator —
    list what the brief omitted in `missing_information`.

    Args:
        campaign_objective: Business objective in the brief's own words.
        optimization_goal: One of reach, frequency, awareness, conversion.
        start_date: Campaign start as YYYY-MM-DD.
        duration_days: Flight length in days, > 0.
        budget: Total campaign budget, > 0.
        city_ids: Resolved city IDs, e.g. ["LH"].
        zone_ids: Resolved zone IDs, e.g. ["LH-ZONE-001"].
        corridor_ids: Resolved corridor IDs, e.g. ["LH-RT-B001"].
        industry_vertical: Advertiser vertical, if stated.
        ad_type: Creative/ad type, if stated.
        audience_age_min: Lower bound of the target age range.
        audience_age_max: Upper bound of the target age range.
        audience_commuter: True when the brief targets commuters specifically.
        requested_num_screens: Exact screen count, only if the brief demands one.
        preferred_time_blocks: dim_slot time_block_ids, "1".."6".
        preferred_dayparts: Named dayparts, e.g. ["morning", "evening"].
        hard_constraints: Non-negotiable limits, e.g. {"max_screens": 40}.
        soft_preferences: Nice-to-haves that must not be enforced as hard limits.
        original_query: The user's verbatim brief, for traceability.
        missing_information: Fields the brief did not specify.
        session_id: Chat session this run belongs to, if any.
    """
    age_range = None
    if audience_age_min is not None and audience_age_max is not None:
        age_range = (audience_age_min, audience_age_max)

    try:
        spec = CampaignSpec(
            campaign_objective=campaign_objective,
            optimization_goal=optimization_goal,  # type: ignore[arg-type]
            start_date=date.fromisoformat(start_date),
            duration_days=duration_days,
            budget=budget,
            city_ids=city_ids or [],
            zone_ids=zone_ids or [],
            corridor_ids=corridor_ids or [],
            industry_vertical=industry_vertical,
            ad_type=ad_type,
            target_audience=AudienceTarget(age_range=age_range, commuter=audience_commuter),
            requested_num_screens=requested_num_screens,
            preferred_time_blocks=[str(b) for b in (preferred_time_blocks or [])],
            preferred_dayparts=preferred_dayparts or [],
            hard_constraints=hard_constraints or {},
            soft_preferences=soft_preferences or {},
            original_query=original_query,
            missing_information=missing_information or [],
        )
    except (ValidationError, ValueError) as exc:
        error(f"campaign spec rejected: {str(exc).splitlines()[0]}")
        return {
            "status": "invalid",
            "errors": str(exc),
            "detail": "Fix the brief or ask the user — do not retry with invented values.",
        }

    unknown = _unknown_geography_ids(spec)
    if unknown:
        error(f"campaign spec rejected: unknown geography IDs {unknown}")
        return {
            "status": "invalid",
            "errors": f"Unknown geography IDs: {unknown}",
            "detail": "Call resolve_geography_terms first; do not guess IDs.",
        }

    run_id = run_state.create_run(spec, session_id=session_id)
    info(
        f"STAGE 1 intake ok: run_id={run_id} goal={spec.optimization_goal} "
        f"budget={spec.budget:,.0f} days={spec.duration_days} "
        f"geo=cities{spec.city_ids}/zones{spec.zone_ids}/corridors{spec.corridor_ids}"
    )
    if spec.missing_information:
        info(f"brief omitted: {spec.missing_information}")
    return {
        "status": "ok",
        "run_id": run_id,
        "normalized_spec": {
            "campaign_objective": spec.campaign_objective,
            "optimization_goal": spec.optimization_goal,
            "geography": {
                "city_ids": spec.city_ids,
                "zone_ids": spec.zone_ids,
                "corridor_ids": spec.corridor_ids,
            },
            "start_date": spec.start_date.isoformat(),
            "end_date": spec.end_date.isoformat(),
            "duration_days": spec.duration_days,
            "budget": spec.budget,
            "requested_num_screens": spec.requested_num_screens,
            "preferred_time_blocks": spec.preferred_time_blocks,
            "hard_constraints": spec.hard_constraints,
        },
        "missing_information": spec.missing_information,
    }


def _unknown_geography_ids(spec: CampaignSpec) -> list[str]:
    from app.data.reference import geography_index

    idx = geography_index()
    unknown = [c for c in spec.city_ids if c not in idx.city_ids]
    unknown += [z for z in spec.zone_ids if z not in idx.zone_ids]
    unknown += [c for c in spec.corridor_ids if c not in idx.corridor_ids]
    return unknown


@tool
def get_run_state(run_id: str) -> dict:
    """Report which pipeline stages have completed for a run, and which used stub output.

    Use this to decide what to delegate next, and before writing the final
    recommendation.
    """
    try:
        return run_state.snapshot(run_id)
    except KeyError as exc:
        return {"status": "error", "detail": str(exc)}


@tool
def verify_package(run_id: str) -> dict:
    """Verify the optimizer's package against the spec, real inventory, and its own math.

    This is the Master Agent's gate. Run it before writing any recommendation. It
    independently recomputes cost, impressions, reach/frequency, geography eligibility,
    slot availability and every declared hard constraint. A failing check must be
    reported to the user — it may never be explained away.
    """
    try:
        spec = run_state.get_spec(run_id)
        result = run_state.get_optimization(run_id)
    except KeyError as exc:
        return {"status": "error", "detail": str(exc)}

    if result is None:
        return {
            "status": "error",
            "detail": "No optimization result on this run yet — delegate to the OR agent first.",
        }
    if result.status == "infeasible" or result.package is None:
        return {
            "status": "infeasible",
            "detail": "Nothing to verify: the optimizer reported the problem infeasible.",
            "infeasibility": result.infeasibility.model_dump() if result.infeasibility else None,
        }

    economics = None
    if (ref := run_state.get_artifact(run_id, "screen_economics")) is not None:
        economics = read_models(ref, ScreenEconomics)

    validation = validate_package(spec, result.package, economics)
    payload = validation.model_dump(mode="json")
    run_state.set_validation(run_id, payload)

    if validation.passed:
        info(f"STAGE 5 verification PASSED ({len(validation.checks)} checks) run_id={run_id}")
    else:
        error(
            f"STAGE 5 verification FAILED run_id={run_id}: "
            f"{[c.name for c in validation.failures]}"
        )
        for check in validation.failures:
            error(f"  {check.name}: expected {check.expected}, observed {check.observed}")

    return {
        "status": "pass" if validation.passed else "fail",
        "summary": validation.summary(),
        "failed_checks": [c.model_dump() for c in validation.failures],
        "checks_run": [{"name": c.name, "status": c.status} for c in validation.checks],
        "stub_stages": run_state.stub_stages(run_id),
    }


@tool
def inspect_package(run_id: str, limit: int = 10) -> dict:
    """Return the package's headline numbers and its highest-value lines, with real
    screen attributes.

    Use this to ground screen-level explanations in actual zones, screen types, prices
    and impression figures. Never restate analytical numbers you have not read here.

    Args:
        run_id: Handle for the campaign run.
        limit: How many allocation lines to return, highest impressions first.
    """
    try:
        result = run_state.get_optimization(run_id)
    except KeyError as exc:
        return {"status": "error", "detail": str(exc)}
    if result is None or result.package is None:
        return {"status": "error", "detail": "No package on this run."}

    pkg = result.package
    facts = screen_facts()
    top = sorted(pkg.allocations, key=lambda a: -a.expected_impressions)[:limit]

    zones = {facts[s].zone_id for s in pkg.screen_ids if s in facts and facts[s].zone_id}
    types: dict[str, int] = {}
    for s in pkg.screen_ids:
        if s in facts:
            types[facts[s].screen_type] = types.get(facts[s].screen_type, 0) + 1

    return {
        "status": "ok",
        "totals": {
            "screens": len(pkg.screen_ids),
            "allocations": len(pkg.allocations),
            "total_cost": round(pkg.total_cost, 2),
            "budget_utilization": round(pkg.budget_utilization, 4),
            "expected_impressions": round(pkg.expected_impressions, 0),
            "expected_reach": round(pkg.expected_reach, 0),
            "expected_frequency": round(pkg.expected_frequency, 3),
            "optimization_method": pkg.optimization_method,
        },
        "composition": {"zones_covered": len(zones), "by_screen_type": types},
        "top_lines": [
            {
                "screen_id": a.screen_id,
                "zone_id": facts[a.screen_id].zone_id if a.screen_id in facts else None,
                "screen_type": facts[a.screen_id].screen_type if a.screen_id in facts else None,
                "time_block_id": a.time_block_id,
                "slots_per_day": a.slots_per_day,
                "duration_days": a.duration_days,
                "price_per_slot_per_day": a.price_per_slot_per_day,
                "line_cost": round(a.line_cost, 2),
                "expected_impressions": round(a.expected_impressions, 0),
            }
            for a in top
        ],
    }


@tool
def check_explanations(run_id: str, explained_screen_ids: list[str]) -> dict:
    """Confirm every screen you are about to explain is actually in the package.

    Run this on the screen IDs in your draft recommendation before returning it.
    """
    try:
        result = run_state.get_optimization(run_id)
    except KeyError as exc:
        return {"status": "error", "detail": str(exc)}
    if result is None or result.package is None:
        return {"status": "error", "detail": "No package on this run."}
    check = validate_explanations(result.package, explained_screen_ids)
    return check.model_dump()


TOOLS = [
    resolve_geography_terms,
    create_campaign_spec,
    get_run_state,
    verify_package,
    inspect_package,
    check_explanations,
]
