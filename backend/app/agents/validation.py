"""Deterministic verification of specialist output.

This is the Master Agent's guard rail and the reason it can be told not to trust its
subagents. Every check here runs in Python against reference data — an LLM never decides
whether a constraint was met, and cannot reason a violation away.

Covers the checklist in SOLUTION.md section 18.
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
    impressions = sum(a.expected_impressions for a in package.allocations)
    checks = [
        _check(
            "impressions_reconcile",
            abs(impressions - package.expected_impressions) <= _tol(impressions),
            "Reported expected_impressions matches the sum over allocations.",
            f"{impressions:,.0f}",
            f"{package.expected_impressions:,.0f}",
        )
    ]

    # Reach is deduplicated exposure, so it can never exceed gross impressions.
    if package.expected_reach:
        checks.append(
            _check(
                "reach_not_above_impressions",
                package.expected_reach <= package.expected_impressions + _tol(impressions),
                "Deduplicated reach does not exceed gross impressions.",
                f"<= {package.expected_impressions:,.0f}",
                f"{package.expected_reach:,.0f}",
            )
        )
        implied_frequency = package.expected_impressions / package.expected_reach
        checks.append(
            _check(
                "frequency_reconciles",
                abs(implied_frequency - package.expected_frequency) <= 0.01,
                "Reported expected_frequency matches impressions / reach.",
                f"{implied_frequency:.3f}",
                f"{package.expected_frequency:.3f}",
            )
        )
    else:
        checks.append(_skip("frequency_reconciles", "No expected_reach reported."))
    return checks


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
    if not economics:
        return [_skip("model_confidence", "No ScreenEconomics supplied.")]
    worst = min(e.confidence for e in economics)
    return [
        _check(
            "model_confidence",
            worst >= MIN_ACCEPTABLE_CONFIDENCE,
            f"Lowest per-screen model confidence is {worst:.2f}.",
            f">= {MIN_ACCEPTABLE_CONFIDENCE:.2f}",
            f"{worst:.2f}",
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
