"""Tools for the DATA INTELLIGENCE AGENT.

============================== INTEGRATION POINT ==============================
The Data Agent is owned by a separate implementer. Replace the bodies of the
`_stub_*` functions below (or swap the delegation in the @tool wrappers) with the
real feature-engineering and relevance-scoring pipeline.

Keep intact:
  * tool names and argument signatures  -- the Master Agent's prompt refers to them
  * the ScreenCandidate contract        -- app/models/screens.py
  * artifact-reference return shape     -- bulky data goes to the artifact store
  * `provenance="computed"` once real   -- flip this so the Master stops warning

Expected real behaviour (SOLUTION.md sections 4-5):
  screens -> locations -> zone_demographics / vehicles -> route_stops -> POI + events
  -> deterministic feature builders -> hard filtering -> weighted relevance score
  -> top ~250 candidates.
===============================================================================
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.config import get_settings
from app.data.reference import eligible_screen_ids, screen_facts
from app.logging_utils import debug, error, info
from app.models.screens import ScreenCandidate
from app.services import run_state
from app.services.artifact_store import write_records
from app.tools._stub_support import STUB_NOTICE, spread, unit

ARTIFACT_KIND = "screen_candidates"


@tool
def describe_inventory(run_id: str) -> dict:
    """Count the real inventory inside a run's requested geography.

    Reference lookup against the screens/locations/vehicles tables — not a stub.
    Use this to sanity-check that the campaign geography resolves to real screens
    before delegating heavier work.
    """
    spec = run_state.get_spec(run_id)
    eligible = eligible_screen_ids(spec.city_ids, spec.zone_ids, spec.corridor_ids)
    facts = screen_facts()

    by_type: dict[str, int] = {}
    by_class: dict[str, int] = {}
    zones: set[str] = set()
    for sid in eligible:
        f = facts[sid]
        by_type[f.screen_type] = by_type.get(f.screen_type, 0) + 1
        by_class[f.inventory_class] = by_class.get(f.inventory_class, 0) + 1
        if f.zone_id:
            zones.add(f.zone_id)

    return {
        "run_id": run_id,
        "requested_geography": {
            "city_ids": spec.city_ids,
            "zone_ids": spec.zone_ids,
            "corridor_ids": spec.corridor_ids,
        },
        "eligible_screens": len(eligible),
        "by_screen_type": by_type,
        "by_inventory_class": by_class,
        "distinct_zones_covered": len(zones),
        "source": "reference lookup (real data)",
    }


@tool
def build_screen_candidates(run_id: str, top_n: int | None = None) -> dict:
    """Score the eligible inventory and persist the top-N candidate screens.

    Produces the `screen_candidates` artifact that the ML and OR stages consume.
    Returns an artifact reference plus score aggregates — never the candidate rows.

    Args:
        run_id: Handle for the campaign run, from create_campaign_spec.
        top_n: Candidate pool size. Defaults to the configured pool size (250).
    """
    spec = run_state.get_spec(run_id)
    limit = top_n or get_settings().candidate_pool_size

    eligible = sorted(eligible_screen_ids(spec.city_ids, spec.zone_ids, spec.corridor_ids))
    if not eligible:
        error(f"STAGE 2 no eligible screens for run_id={run_id} — geography unsatisfiable")
        return {
            "status": "no_candidates",
            "run_id": run_id,
            "detail": (
                "The requested geography resolves to zero screens. The campaign spec "
                "cannot be satisfied as written — report this instead of proceeding."
            ),
        }

    debug(f"STAGE 2 scoring {len(eligible)} eligible screens, keeping top {limit}")
    candidates = _stub_score_candidates(run_id, eligible, limit)
    ref = write_records(
        ARTIFACT_KIND,
        candidates,
        provenance="stub",
        summary={
            "eligible_screens": len(eligible),
            "candidates": len(candidates),
            "relevance_min": round(min(c.relevance_score for c in candidates), 4),
            "relevance_mean": round(
                sum(c.relevance_score for c in candidates) / len(candidates), 4
            ),
            "relevance_max": round(max(c.relevance_score for c in candidates), 4),
        },
    )
    run_state.set_artifact(run_id, ARTIFACT_KIND, ref)
    info(
        f"STAGE 2 candidates ready [STUB]: {len(candidates)} of {len(eligible)} eligible, "
        f"relevance {ref.summary['relevance_min']}-{ref.summary['relevance_max']}, "
        f"artifact={ref.artifact_id}"
    )

    return {
        "status": "ok",
        "artifact": ref.as_context(),
        "eligible_screens": len(eligible),
        "candidates_selected": len(candidates),
        "top_screens_preview": [
            {"screen_id": c.screen_id, "relevance_score": c.relevance_score} for c in candidates[:5]
        ],
        "warning": STUB_NOTICE,
    }


def _stub_score_candidates(run_id: str, eligible: list[str], limit: int) -> list[ScreenCandidate]:
    """PLACEHOLDER scoring. Replace with the real weighted / learned relevance model."""
    facts = screen_facts()
    scored: list[ScreenCandidate] = []

    for sid in eligible:
        f = facts[sid]
        audience = spread(unit(sid, "audience"), 0.35, 0.98)
        geography = spread(unit(sid, "geography"), 0.55, 0.99)
        contextual = spread(unit(sid, "context"), 0.30, 0.95)
        transit = spread(unit(sid, "transit"), 0.35, 0.97)
        relevance = 0.40 * audience + 0.25 * geography + 0.20 * transit + 0.15 * contextual

        scored.append(
            ScreenCandidate(
                screen_id=sid,
                relevance_score=round(relevance, 4),
                audience_match_score=round(audience, 4),
                geography_score=round(geography, 4),
                contextual_score=round(contextual, 4),
                transit_score=round(transit, 4),
                reasons=[
                    (
                        f"STUB placeholder score for {f.screen_type} screen in "
                        f"{f.zone_id or f.corridor_id or f.city_id}"
                    ),
                    "Real reasons must cite demographic, transit and POI feature values",
                ],
                hard_constraints_passed=True,
                city_id=f.city_id,
                zone_id=f.zone_id,
                screen_type=f.screen_type,
            )
        )

    scored.sort(key=lambda c: (-c.relevance_score, c.screen_id))
    return scored[:limit]


TOOLS = [describe_inventory, build_screen_candidates]
