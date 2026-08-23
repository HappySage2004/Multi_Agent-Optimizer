"""Labels the answer is read from.

`inspect_package` feeds a client-facing table, so its labels are a contract, not
convenience. A rep sending a quote must never see `LH-ZONE-005`, `metro_station` or a bare
block number, and the only place that can be guaranteed is here — a prompt instruction to
"use nice names" is not enforcement.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pytest

from app.data.reference import screen_facts, time_block_labels
from app.tools import master_tools, ml_agent_tools, or_agent_tools, relevance_tools

RAW_ID = re.compile(r"^[A-Z]{2,4}-(ZONE|RT)-")


# ------------------------------------------------------------------ reference layer


def test_every_screen_resolves_to_a_readable_place():
    """A null label would render an empty cell in the package table."""
    facts = screen_facts()
    assert facts
    assert all(f.place_label for f in facts.values())


def test_fixed_screens_are_named_by_zone_and_mobile_by_route():
    facts = screen_facts()
    fixed = next(f for f in facts.values() if f.inventory_class == "fixed")
    mobile = next(f for f in facts.values() if f.inventory_class == "mobile")

    # A zone name, not the id it came from.
    assert fixed.zone_name and not RAW_ID.match(fixed.place_label)
    assert fixed.place_label == fixed.zone_name

    # Vehicle-mounted screens have no zone at all — the route is their geography.
    assert mobile.zone_id is None
    assert mobile.zone_name is None
    assert mobile.place_label == mobile.corridor_name


def test_screen_type_labels_are_not_snake_case():
    labels = {f.screen_type_label for f in screen_facts().values()}
    assert labels == {"Bus", "Bus Stop", "Metro Rail Coach", "Metro Station"}


def test_time_block_labels_carry_clock_hours_and_daypart():
    labels = time_block_labels()
    assert labels["5"] == "16:00-20:00 (Evening)"
    assert labels["2"] == "04:00-08:00 (Morning)"
    # All six, so a package in any block gets a real label rather than "Block N".
    assert len(labels) == 6
    assert all(re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2} \(\w+\)", v) for v in labels.values())


# ------------------------------------------------------------------ inspect_package


@pytest.fixture(scope="module")
def package_run() -> str:
    created = master_tools.create_campaign_spec.invoke(
        {
            "campaign_objective": "premium positioning for an EV launch",
            "optimization_goal": "reach",
            "start_date": (date.today() + timedelta(days=14)).isoformat(),
            "duration_days": 45,
            "budget": 60_000.0,
            "city_ids": ["LH"],
            "zone_ids": ["LH-ZONE-005"],
            "audience_terms": ["high_income"],
            "industry_vertical": "auto",
        }
    )
    run_id = created["run_id"]
    relevance_tools.build_screen_candidates.invoke({"run_id": run_id})
    ml_agent_tools.estimate_screen_economics.invoke({"run_id": run_id})
    assert or_agent_tools.optimize_package.invoke({"run_id": run_id})["status"] in {
        "optimal",
        "feasible",
    }
    return run_id


def test_every_line_carries_the_labels_the_table_needs(package_run):
    out = master_tools.inspect_package.invoke({"run_id": package_run})
    assert out["status"] == "ok"
    assert out["lines"]

    for line in out["lines"]:
        # Named place, never the raw id.
        assert line["zone"] and not RAW_ID.match(line["zone"])
        # Title-cased type.
        assert line["screen_type"] and "_" not in line["screen_type"]
        # A clock range, not "Block 5".
        assert re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2} \(\w+\)", line["time_block"])
        # And the raw ids still travel alongside, for traceability.
        assert line["time_block_id"]


def test_zone_id_is_still_available_beside_the_label(package_run):
    """The label is for the rep; the id is what the run and the artifacts are keyed on."""
    line = master_tools.inspect_package.invoke({"run_id": package_run})["lines"][0]
    assert line["zone_id"] == "LH-ZONE-005"
    assert line["zone"] == "Financial Row"


def test_composition_reports_named_places(package_run):
    out = master_tools.inspect_package.invoke({"run_id": package_run})
    names = out["composition"]["place_names"]
    assert names == ["Financial Row"]
    assert out["composition"]["places_covered"] == len(names)
    # Screen-type counts are keyed by label too, so the mix reads as prose.
    assert all("_" not in key for key in out["composition"]["by_screen_type"])


def test_truncation_is_reported_rather_than_silent(package_run):
    """A partial table presented as the whole package is a misrepresented quote."""
    full = master_tools.inspect_package.invoke({"run_id": package_run})
    assert full["lines_truncated"] == 0
    assert full["lines_returned"] == full["totals"]["allocations"]

    clipped = master_tools.inspect_package.invoke({"run_id": package_run, "limit": 3})
    assert clipped["lines_returned"] == 3
    assert clipped["lines_truncated"] == full["totals"]["allocations"] - 3


def test_lines_are_ordered_by_exposures_so_the_table_leads_with_the_best(package_run):
    lines = master_tools.inspect_package.invoke({"run_id": package_run})["lines"]
    exposures = [line["viewed_exposures"] for line in lines]
    assert exposures == sorted(exposures, reverse=True)


def test_line_costs_reconcile_with_the_reported_total(package_run):
    """The table's Total row must equal the package the validator checked."""
    out = master_tools.inspect_package.invoke({"run_id": package_run})
    assert sum(line["line_cost"] for line in out["lines"]) == pytest.approx(
        out["totals"]["total_cost"], abs=0.02
    )


# ------------------------------------------------------------------ candidate contract


def test_candidates_carry_zone_names_for_the_inspector(package_run):
    from app.models.screens import ScreenCandidate
    from app.services import run_state
    from app.services.artifact_store import read_models

    candidates = read_models(
        run_state.get_artifact(package_run, "screen_candidates"), ScreenCandidate
    )
    named = [c for c in candidates if c.zone_id]
    assert named, "this brief is zone-scoped, so candidates must have zones"
    assert all(c.zone_name == "Financial Row" for c in named)
