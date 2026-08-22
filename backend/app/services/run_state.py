"""Per-run shared state.

Agents pass a `run_id` between tools instead of inlining CampaignSpec JSON, artifact
contents, or packages into their messages. Everything bulky lives here or in the artifact
store; the LLM only ever sees the handle plus a summary.

Backed by localDB/runs.json so a run survives a process restart and the UI can replay it.
"""

from __future__ import annotations

from typing import Any

from app.models.artifacts import ArtifactReference, Provenance
from app.models.campaign import CampaignSpec
from app.models.optimization import OptimizationResult
from app.services import local_db


def create_run(spec: CampaignSpec, *, session_id: str | None = None) -> str:
    record = local_db.insert(
        local_db.RUNS,
        {
            "session_id": session_id,
            "status": "spec_ready",
            "campaign_spec": spec.model_dump(mode="json"),
            "artifacts": {},
            "optimization": None,
            "validation": None,
            "stub_stages": [],
        },
    )
    return record["id"]


def latest_run_for_session(session_id: str) -> str | None:
    """The most recent run started in a session, or None if it has never run one.

    This is what makes a follow-up turn possible: the Master Agent resolves the package it
    is being asked about from the session id, rather than depending on a conversational
    memory that a process restart would lose.
    """
    # `created_at` has second granularity, so two runs started in the same second tie —
    # which happens when the agent retries `create_campaign_spec` after a validation
    # error. `max()` would return the first of a tie, i.e. the abandoned run. Scanning
    # with `>=` instead lets the later record win, and localDB appends in insertion order.
    latest: dict[str, Any] | None = None
    for run in local_db.list_records(local_db.RUNS):
        if run.get("session_id") != session_id:
            continue
        if latest is None or (run.get("created_at") or "") >= (latest.get("created_at") or ""):
            latest = run
    return None if latest is None else latest["id"]


def _require(run_id: str) -> dict[str, Any]:
    record = local_db.get_record(local_db.RUNS, run_id)
    if record is None:
        raise KeyError(f"Unknown run_id '{run_id}'")
    return record


def get_spec(run_id: str) -> CampaignSpec:
    return CampaignSpec.model_validate(_require(run_id)["campaign_spec"])


def set_artifact(run_id: str, kind: str, ref: ArtifactReference) -> None:
    record = _require(run_id)
    artifacts = dict(record.get("artifacts") or {})
    artifacts[kind] = ref.model_dump(mode="json")

    stub_stages = list(record.get("stub_stages") or [])
    if ref.provenance == "stub" and kind not in stub_stages:
        stub_stages.append(kind)

    local_db.update(local_db.RUNS, run_id, {"artifacts": artifacts, "stub_stages": stub_stages})


def get_artifact(run_id: str, kind: str) -> ArtifactReference | None:
    raw = (_require(run_id).get("artifacts") or {}).get(kind)
    return ArtifactReference.model_validate(raw) if raw else None


def require_artifact(run_id: str, kind: str) -> ArtifactReference:
    ref = get_artifact(run_id, kind)
    if ref is None:
        raise KeyError(
            f"Run '{run_id}' has no '{kind}' artifact yet — the producing stage must run first."
        )
    return ref


def missing_prerequisite(run_id: str, kind: str) -> dict[str, Any] | None:
    """Return an actionable error payload if `kind` is not on the run yet, else None.

    The pipeline stages are strictly sequential — each consumes the previous stage's
    artifact. A supervisor can still delegate stages concurrently, so every dependent tool
    checks its input here and returns a recoverable result instead of raising.
    """
    if get_artifact(run_id, kind) is not None:
        return None
    producer = {
        "screen_candidates": (
            "the relevance engine (stage 2) — call build_screen_candidates yourself, it is "
            "a Master-owned tool with no subagent"
        ),
        "screen_economics": "the ml_agent (stage 3)",
    }.get(kind, f"the stage that produces '{kind}'")
    return {
        "status": "prerequisite_missing",
        "run_id": run_id,
        "missing_artifact": kind,
        "detail": (
            f"Run '{run_id}' has no '{kind}' artifact yet. Delegate to {producer} first, "
            f"wait for it to finish, then retry this stage. Do not run pipeline stages "
            f"concurrently — each one consumes the previous stage's output."
        ),
    }


def set_optimization(run_id: str, result: OptimizationResult) -> None:
    local_db.update(
        local_db.RUNS,
        run_id,
        {
            "optimization": result.model_dump(mode="json"),
            "status": "optimized" if result.status != "infeasible" else "infeasible",
        },
    )


def get_optimization(run_id: str) -> OptimizationResult | None:
    raw = _require(run_id).get("optimization")
    return OptimizationResult.model_validate(raw) if raw else None


def set_validation(run_id: str, payload: dict[str, Any]) -> None:
    local_db.update(local_db.RUNS, run_id, {"validation": payload, "status": "validated"})


def stub_stages(run_id: str) -> list[str]:
    """Stages whose output came from an unimplemented specialist."""
    return list(_require(run_id).get("stub_stages") or [])


def overall_provenance(run_id: str) -> Provenance:
    return "stub" if stub_stages(run_id) else "computed"


def snapshot(run_id: str) -> dict[str, Any]:
    """Compact view of run progress. Safe to render into a prompt."""
    record = _require(run_id)
    artifacts = record.get("artifacts") or {}
    opt = record.get("optimization")
    return {
        "run_id": run_id,
        "status": record.get("status"),
        "artifacts": {
            kind: {
                "artifact_id": ref["artifact_id"],
                "rows": ref["rows"],
                "provenance": ref["provenance"],
            }
            for kind, ref in artifacts.items()
        },
        "optimization_status": (opt or {}).get("status"),
        "validated": record.get("validation") is not None,
        "stub_stages": record.get("stub_stages") or [],
    }
