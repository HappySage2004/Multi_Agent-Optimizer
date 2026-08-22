"""Deterministic verification of specialist output.

This is the Master Agent's guard rail and the reason it can be told not to trust its
subagents. Every check here runs in Python against reference data — an LLM never decides
whether a constraint was met, and cannot reason a violation away.

Covers the checklist in SOLUTION.md section 18.

Two checks deliberately recompute rather than trust: `cost_reconciles` re-derives
sum(price x slots x days) from the allocations, and `reach_reconciles` re-derives
deduplicated reach from the pool_key groups. Both exist because those are the two numbers
an optimizer can most plausibly overstate.
"""

from __future__ import annotations

from datetime import date

from app.data.reference import eligible_screen_ids, screen_facts, time_block_ids
from app.models.campaign import CampaignSpec
from app.models.economics import ScreenEconomics
from app.models.optimization import OptimizedPackage
from app.models.recommendation import ValidationCheck, ValidationResult

MIN_ACCEPTABLE_CONFIDENCE = 0.30


def _tol(value: float) -> float:
    """Absolute tolerance that scales with magnitude, for float reconciliation."""
    return max(0.01, abs(value) * 1e-6)


def _check(name: str, ok: bool, detail: str, expected=None, observed=None) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status="pass" if ok else "fail",
        detail=detail,
        expected=None if expected is None else str(expected),
        observed=None if observed is None else str(observed),
    )


def _skip(name: str, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, status="skipped", detail=detail)


def validate_package(
    spec: CampaignSpec,
    package: OptimizedPackage,
    economics: list[ScreenEconomics] | None = None,
    *,
    today: date | None = None,
) -> ValidationResult:
    """Verify an OptimizedPackage against the spec, reference data, and its own arithmetic."""
    checks: list[ValidationCheck] = []
    allocations = package.allocations

    if not allocations:
        checks.append(
            _check("package_non_empty", False, "Package contains no allocations.", ">=1", 0)
        )
        return ValidationResult(passed=False, checks=checks)
    checks.append(
        _check("package_non_empty", True, f"Package contains {len(allocations)} allocations.")
    )

    checks.extend(_cost_checks(spec, package))
    checks.extend(_reconciliation_checks(package))
    checks.extend(_reach_checks(package, economics))
    checks.extend(_inventory_checks(package))
    checks.extend(_geography_checks(spec, package))
    checks.extend(_date_checks(spec, package, today=today))
    checks.extend(_hard_constraint_checks(spec, package))
    checks.extend(_availability_checks(package, economics))
    checks.extend(_confidence_checks(economics))

    return ValidationResult(passed=all(c.status != "fail" for c in checks), checks=checks)


# --- budget and arithmetic -----------------------------------------------------


def _cost_checks(spec: CampaignSpec, package: OptimizedPackage) -> list[ValidationCheck]:
    computed = sum(a.line_cost for a in package.allocations)
    checks = [
        _check(
            "budget_respected",
            computed <= spec.budget + _tol(spec.budget),
            f"Recomputed package cost {computed:,.2f} against budget {spec.budget:,.2f}.",
            f"<= {spec.budget:,.2f}",
            f"{computed:,.2f}",
        ),
        _check(
            "cost_reconciles",
            abs(computed - package.total_cost) <= _tol(computed),
            "Reported total_cost matches the sum of price x slots x days over allocations.",
            f"{computed:,.2f}",
            f"{package.total_cost:,.2f}",
        ),
    ]
    if spec.budget > 0:
        expected_util = computed / spec.budget
        checks.append(
            _check(
                "budget_utilization_reconciles",
                abs(expected_util - package.budget_utilization) <= 1e-4,
                "Reported budget_utilization matches recomputed cost / budget.",
                f"{expected_util:.4f}",
                f"{package.budget_utilization:.4f}",
            )
        )
    return checks


def _reconciliation_checks(package: OptimizedPackage) -> list[ValidationCheck]:
    exposures = sum(a.viewed_exposures for a in package.allocations)
    checks = [
        _check(
            "impressions_reconcile",
            abs(exposures - package.gross_impressions_viewed) <= _tol(exposures),
            "Reported gross_impressions_viewed matches the sum over allocations.",
            f"{exposures:,.0f}",
            f"{package.gross_impressions_viewed:,.0f}",
        )
    ]

    # Reach is deduplicated exposure, so it can never exceed gross exposures.
    if package.expected_reach:
        checks.append(
            _check(
                "reach_not_above_impressions",
                package.expected_reach <= package.gross_impressions_viewed + _tol(exposures),
                "Deduplicated reach does not exceed gross viewed exposures.",
                f"<= {package.gross_impressions_viewed:,.0f}",
                f"{package.expected_reach:,.0f}",
            )
        )
        implied_frequency = package.gross_impressions_viewed / package.expected_reach
        checks.append(
            _check(
                "frequency_reconciles",
                abs(implied_frequency - package.expected_frequency) <= 0.01,
                "Reported expected_frequency matches viewed exposures / reach.",
                f"{implied_frequency:.3f}",
                f"{package.expected_frequency:.3f}",
            )
        )
    else:
        checks.append(_skip("frequency_reconciles", "No expected_reach reported."))
    return checks


def _reach_checks(
    package: OptimizedPackage, economics: list[ScreenEconomics] | None
) -> list[ValidationCheck]:
    """Recompute deduplicated reach from the economics and compare it to the optimizer's.

    This is the check that catches an over-counted audience — the single easiest number in
    this system to inflate, because summing per-screen impressions looks reasonable and
    over-states reach by ~23x on a realistic pool. Screens sharing a `pool_key` see the
    same people, so each (pool, time block) group's bought exposure is capped at that
    group's REACHABLE daily audience — the share who look at the screen, since the exposures
    being capped are viewed exposures.

    Keep this definition in step with `or_agent_tools._package_metrics`. The point is that
    two independent implementations agree, so do not import one into the other.
    """
    if not economics:
        return [
            _skip(
                "reach_reconciles",
                "No ScreenEconomics supplied — deduplicated reach could not be "
                "independently recomputed.",
            )
        ]
    if not package.expected_reach:
        return [_skip("reach_reconciles", "Package reports no expected_reach.")]

    lookup = {(e.screen_id, str(e.time_block_id)): e for e in economics}
    grouped: dict[tuple[str, str], float] = {}
    caps: dict[tuple[str, str], float] = {}
    unmatched = 0
    for a in package.allocations:
        line = lookup.get((a.screen_id, str(a.time_block_id)))
        if line is None:
            unmatched += 1
            continue
        key = (line.pool_key or line.screen_id, str(a.time_block_id))
        grouped[key] = grouped.get(key, 0.0) + a.viewed_exposures
        caps[key] = max(caps.get(key, 0.0), line.reachable_daily_audience)

    if unmatched:
        return [
            _check(
                "reach_reconciles",
                False,
                f"{unmatched} allocation(s) have no matching screen_economics line, so "
                f"reach cannot be verified.",
                "0 unmatched",
                f"{unmatched} unmatched",
            )
        ]

    recomputed = sum(min(gross, caps.get(key, 0.0)) for key, gross in grouped.items())
    checks = [
        _check(
            "reach_reconciles",
            abs(recomputed - package.expected_reach) <= max(1.0, _tol(recomputed)),
            "Reported expected_reach matches reach recomputed from pool_key groups, each "
            "capped at its reachable daily audience.",
            f"{recomputed:,.0f}",
            f"{package.expected_reach:,.0f}",
        )
    ]
    checks.extend(_curve_reach_guard(package, caps))
    return checks


def _curve_reach_guard(
    package: OptimizedPackage, caps: dict[tuple[str, str], float]
) -> list[ValidationCheck]:
    """Bound the solver's saturation-curve diagnostic without depending on its constant.

    `curve_reach_diagnostic` comes from `P x (1 - exp(-lambda x E / P))`, and lambda is
    ASSUMED. Re-deriving that formula here would validate nothing about the only real
    unknown in it, so this checks the two things that hold for ANY lambda: reach can exceed
    neither the exposures bought nor the people available to be reached. It catches the
    over-count class — pool misalignment inflating a diagnostic that then gets quoted.
    """
    if package.curve_reach_diagnostic is None:
        return [_skip("curve_reach_bounded", "Package reports no curve_reach_diagnostic.")]
    ceiling = min(package.gross_impressions_viewed, sum(caps.values()))
    curve = package.curve_reach_diagnostic
    return [
        _check(
            "curve_reach_bounded",
            curve <= ceiling + max(1.0, _tol(ceiling)),
            "Solver curve reach stays within min(gross viewed exposures, reachable "
            "audience) — a bound that holds for any saturation constant.",
            f"<= {ceiling:,.0f}",
            f"{curve:,.0f}",
        )
    ]


# --- inventory, geography, dates ----------------------------------------------


def _inventory_checks(package: OptimizedPackage) -> list[ValidationCheck]:
    facts = screen_facts()
    valid_blocks = time_block_ids()

    unknown = sorted({a.screen_id for a in package.allocations if a.screen_id not in facts})
    bad_blocks = sorted(
        {a.time_block_id for a in package.allocations if str(a.time_block_id) not in valid_blocks}
    )

    duplicates = _duplicate_lines(package)

    return [
        _check(
            "screens_exist",
            not unknown,
            "Every allocated screen_id exists in the inventory."
            if not unknown
            else f"Unknown screen_ids: {unknown[:5]}",
            "0 unknown",
            f"{len(unknown)} unknown",
        ),
        _check(
            "time_blocks_valid",
            not bad_blocks,
            "Every time_block_id exists in dim_slot."
            if not bad_blocks
            else f"Unknown time_block_ids: {bad_blocks}",
            f"one of {sorted(valid_blocks)}",
            f"{len(bad_blocks)} invalid",
        ),
        _check(
            "no_duplicate_allocations",
            not duplicates,
            "Each (screen, time block) pair appears at most once."
            if not duplicates
            else f"Duplicated pairs: {duplicates[:5]}",
            "0 duplicates",
            f"{len(duplicates)} duplicates",
        ),
    ]


def _duplicate_lines(package: OptimizedPackage) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    dupes: list[tuple[str, str]] = []
    for a in package.allocations:
        key = (a.screen_id, str(a.time_block_id))
        (dupes.append(key) if key in seen else seen.add(key))
    return dupes


def _geography_checks(spec: CampaignSpec, package: OptimizedPackage) -> list[ValidationCheck]:
    eligible = eligible_screen_ids(spec.city_ids, spec.zone_ids, spec.corridor_ids)
    if not eligible:
        return [
            _check(
                "geography_eligible",
                False,
                "Requested geography resolves to zero screens — the spec itself is unsatisfiable.",
                ">=1 eligible screen",
                "0",
            )
        ]
    offenders = sorted({a.screen_id for a in package.allocations if a.screen_id not in eligible})
    return [
        _check(
            "geography_eligible",
            not offenders,
            f"All allocated screens fall inside the requested geography "
            f"({len(eligible):,} eligible)."
            if not offenders
            else f"Screens outside requested geography: {offenders[:5]}",
            "0 outside",
            f"{len(offenders)} outside",
        )
    ]


def _date_checks(
    spec: CampaignSpec, package: OptimizedPackage, *, today: date | None
) -> list[ValidationCheck]:
    checks = []
    over_run = [a.screen_id for a in package.allocations if a.duration_days > spec.duration_days]
    checks.append(
        _check(
            "duration_within_campaign",
            not over_run,
            "No allocation runs longer than the campaign duration."
            if not over_run
            else f"Allocations exceeding {spec.duration_days} days: {over_run[:5]}",
            f"<= {spec.duration_days} days",
            f"{len(over_run)} over",
        )
    )
    reference = today or date.today()
    checks.append(
        _check(
            "start_date_not_in_past",
            spec.start_date >= reference,
            f"Campaign starts {spec.start_date.isoformat()} "
            f"(reference date {reference.isoformat()}).",
            f">= {reference.isoformat()}",
            spec.start_date.isoformat(),
        )
    )
    return checks


# --- spec-declared hard constraints -------------------------------------------


def _hard_constraint_checks(spec: CampaignSpec, package: OptimizedPackage) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    facts = screen_facts()
    n_screens = len(package.screen_ids)

    if spec.requested_num_screens is not None:
        checks.append(
            _check(
                "requested_num_screens",
                n_screens == spec.requested_num_screens,
                f"Package uses {n_screens} distinct screens.",
                spec.requested_num_screens,
                n_screens,
            )
        )

    hc = spec.hard_constraints
    if (min_screens := hc.get("min_screens")) is not None:
        checks.append(
            _check(
                "min_screens",
                n_screens >= min_screens,
                f"Package uses {n_screens} screens.",
                f">= {min_screens}",
                n_screens,
            )
        )
    if (max_screens := hc.get("max_screens")) is not None:
        checks.append(
            _check(
                "max_screens",
                n_screens <= max_screens,
                f"Package uses {n_screens} screens.",
                f"<= {max_screens}",
                n_screens,
            )
        )

    if allowed_types := hc.get("allowed_screen_types"):
        offenders = sorted(
            {
                sid
                for sid in package.screen_ids
                if sid in facts and facts[sid].screen_type not in set(allowed_types)
            }
        )
        checks.append(
            _check(
                "allowed_screen_types",
                not offenders,
                f"Screen types restricted to {allowed_types}."
                if not offenders
                else f"Disallowed-type screens: {offenders[:5]}",
                allowed_types,
                f"{len(offenders)} violations",
            )
        )

    # Preferred time blocks are a hard constraint only when declared as one.
    required_blocks = hc.get("required_time_blocks") or hc.get("time_blocks")
    if required_blocks:
        allowed = {str(b) for b in required_blocks}
        offenders = sorted(
            {a.time_block_id for a in package.allocations if str(a.time_block_id) not in allowed}
        )
        checks.append(
            _check(
                "required_time_blocks",
                not offenders,
                f"All allocations sit in time blocks {sorted(allowed)}."
                if not offenders
                else f"Out-of-scope blocks used: {offenders}",
                sorted(allowed),
                f"{len(offenders)} violations",
            )
        )

    if (min_zones := hc.get("min_zone_coverage")) is not None:
        covered = {facts[s].zone_id for s in package.screen_ids if s in facts and facts[s].zone_id}
        checks.append(
            _check(
                "min_zone_coverage",
                len(covered) >= min_zones,
                f"Package covers {len(covered)} zones.",
                f">= {min_zones}",
                len(covered),
            )
        )

    if not checks:
        checks.append(_skip("hard_constraints", "Spec declared no additional hard constraints."))
    return checks


# --- availability and model confidence ----------------------------------------


def _availability_checks(
    package: OptimizedPackage, economics: list[ScreenEconomics] | None
) -> list[ValidationCheck]:
    if not economics:
        return [
            _skip(
                "inventory_availability",
                "No ScreenEconomics supplied — slot availability could not be independently "
                "verified.",
            )
        ]
    capacity = {(e.screen_id, str(e.time_block_id)): e.max_slots_per_day for e in economics}
    offenders = [
        f"{a.screen_id}/{a.time_block_id}: {a.slots_per_day} > "
        f"{capacity.get((a.screen_id, str(a.time_block_id)), 0)}"
        for a in package.allocations
        if a.slots_per_day > capacity.get((a.screen_id, str(a.time_block_id)), 0)
    ]
    return [
        _check(
            "inventory_availability",
            not offenders,
            "Purchased slots never exceed available slots."
            if not offenders
            else f"Over-allocated: {offenders[:5]}",
            "0 over-allocations",
            f"{len(offenders)} over-allocations",
        )
    ]


def _confidence_checks(economics: list[ScreenEconomics] | None) -> list[ValidationCheck]:
    """SOLUTION.md section 18 check 6.

    Skipped rather than evaluated: no stage produces a per-screen confidence today. The
    pricing engine reports its trust signals differently — segmentation depth and sample
    size per row in `assumptions`, and a price-coefficient sign check on the
    booking-probability model as a whole (see `describe_pricing_model`). Gating on the
    contract's defaulted 0.5 would report a pass that means nothing. Restore the gate when
    a stage emits a real confidence.
    """
    if not economics:
        return [_skip("model_confidence", "No ScreenEconomics supplied.")]
    return [
        _skip(
            "model_confidence",
            "No stage emits a per-screen confidence; not gating on the contract default. "
            "Pricing trust signals are per-row assumptions plus the booking-probability "
            "sign check.",
        )
    ]


def validate_explanations(
    package: OptimizedPackage, explained_screen_ids: list[str]
) -> ValidationCheck:
    """Check 10: explanations must describe screens that are actually in the package."""
    in_package = set(package.screen_ids)
    stray = sorted(set(explained_screen_ids) - in_package)
    return _check(
        "explanations_consistent",
        not stray,
        "Every screen explanation refers to a screen in the recommended package."
        if not stray
        else f"Explanations reference screens not in the package: {stray[:5]}",
        "0 stray",
        f"{len(stray)} stray",
    )
