"""Campaign endpoints — where a natural-language brief meets the Master Agent."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.master import build_master_agent
from app.agents.tracing import AgentRunLogger
from app.api.schemas import ArtifactRowsOut, CampaignQuery, CampaignRunOut
from app.config import get_settings
from app.logging_utils import error as log_error
from app.logging_utils import info as log_info
from app.services import artifact_store, local_db, run_state, session_titles

router = APIRouter(tags=["campaign"])

# One checkpointer per process keeps multi-turn refinement working within a session.
# Swap for a persistent saver when runs need to survive a restart.
_checkpointer = InMemorySaver()
_agent = None

RECURSION_LIMIT = 80

# Provider failures worth distinguishing for the caller. These are the provider-agnostic
# langchain_core bases, so this mapping survives a model-provider swap.
#   ModelRateLimitError -> quota. Gemini's free tier allows ~20 requests/day/model and one
#       full orchestration costs roughly 15-20 model calls, so a single run can exhaust it.
#   ModelAPIError       -> upstream 5xx (503 UNAVAILABLE / 504 DEADLINE_EXCEEDED under
#       load). Transient and worth retrying.
#   auth / not-found    -> our own misconfiguration, not a client error.
QUOTA_DETAIL = (
    "Model quota exhausted. Gemini's free tier allows ~20 requests/day/model and one full "
    "orchestration costs roughly 15-20 model calls. Enable billing on the API key, or "
    "point MASTER_MODEL / SPECIALIST_MODEL at a model with remaining quota."
)
MODEL_BUSY_DETAIL = (
    "The model is currently unavailable (the provider returned a 5xx under load). This is "
    "transient — retry the request."
)


def _provider_error(exc: ModelError) -> HTTPException:
    """Map a provider failure onto an honest HTTP status instead of a blanket 500."""
    if isinstance(exc, ModelRateLimitError):
        return HTTPException(status_code=429, detail=f"{QUOTA_DETAIL} Provider said: {exc}")
    if isinstance(exc, ModelAPIError):
        return HTTPException(status_code=503, detail=f"{MODEL_BUSY_DETAIL} Provider said: {exc}")
    if isinstance(exc, (ModelAuthenticationError, ModelPermissionDeniedError)):
        return HTTPException(
            status_code=502,
            detail=f"The model provider rejected our credentials. Check GEMINI_API_KEY. {exc}",
        )
    if isinstance(exc, ModelNotFoundError):
        return HTTPException(
            status_code=502,
            detail=f"Configured model id is not available on this key. {exc}",
        )
    return HTTPException(status_code=502, detail=f"Model provider error: {exc}")


def _agent_instance():
    global _agent
    if _agent is None:
        log_info("compiling master agent graph (first request)")
        _agent = build_master_agent(checkpointer=_checkpointer)
    return _agent


def _run_config(session_id: str, tracer: AgentRunLogger) -> dict:
    """One tracer registered here covers the whole tree — callbacks propagate to subagents."""
    return {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": [tracer],
    }


def _require_api_key() -> None:
    if not get_settings().gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is not configured. Set it in the repo-root .env (or backend/.env)."
            ),
        )


def _ensure_session(session_id: str | None, query: str) -> str:
    """Resolve the session for this run, naming it after the brief if it is still unnamed."""
    if session_id:
        if local_db.get_record(local_db.SESSIONS, session_id) is None:
            raise HTTPException(status_code=404, detail=f"No session '{session_id}'")
        session_titles.name_if_unnamed(session_id, query)
        return session_id
    created = local_db.insert(local_db.SESSIONS, {"title": session_titles.title_from_text(query)})
    return created["id"]


def _build_prompt(payload: CampaignQuery, session_id: str) -> str:
    lines = [payload.query.strip(), "", f"session_id: {session_id}"]
    if payload.upload_ids:
        staged = [
            r for r in local_db.list_records(local_db.UPLOADS) if r["id"] in payload.upload_ids
        ]
        if staged:
            lines += ["", "Staged documents (read with the filesystem tools if needed):"]
            lines += [f"- {r['filename']} at {r['stored_path']}" for r in staged]
    return "\n".join(lines)


def _latest_run_for_session(session_id: str) -> str | None:
    runs = [r for r in local_db.list_records(local_db.RUNS) if r.get("session_id") == session_id]
    if not runs:
        return None
    return max(runs, key=lambda r: r.get("created_at") or "")["id"]


def _message_text(message: Any) -> str:
    """Assistant text carried by one message, or "" if it carries none.

    Gemini returns content blocks rather than a bare string, so keep only the `text`
    blocks and drop thinking/tool_use. Shared by the blocking and streaming paths.
    """
    content = getattr(message, "content", None)
    if getattr(message, "type", None) != "ai" or not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def _final_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        if text := _message_text(message):
            return text
    return ""


@router.post("/campaign/run", response_model=CampaignRunOut)
async def run_campaign(payload: CampaignQuery) -> CampaignRunOut:
    """Run the full orchestration for a brief and return the final recommendation."""
    _require_api_key()
    session_id = _ensure_session(payload.session_id, payload.query)

    tracer = AgentRunLogger(label=f"campaign/run session={session_id}")
    tracer.start()
    log_info(f"brief received ({len(payload.query)} chars), model={get_settings().master_model_id}")

    try:
        state = await _agent_instance().ainvoke(
            {"messages": [{"role": "user", "content": _build_prompt(payload, session_id)}]},
            config=_run_config(session_id, tracer),
        )
    except ModelError as exc:
        log_error(f"run aborted: {type(exc).__name__}")
        tracer.log_summary()
        raise _provider_error(exc) from exc

    usage = tracer.log_summary()

    run_id = _latest_run_for_session(session_id)
    snapshot = run_state.snapshot(run_id) if run_id else None
    if run_id:
        log_info(
            f"run_id={run_id} status={snapshot.get('status')} "
            f"stub_stages={snapshot.get('stub_stages')}"
        )
    else:
        log_error("agent finished without creating a campaign run — intake likely never ran")

    return CampaignRunOut(
        session_id=session_id,
        session_title=session_titles.title_of(session_id),
        run_id=run_id,
        answer=_final_text(state),
        stub_stages=(snapshot or {}).get("stub_stages", []),
        provenance=run_state.overall_provenance(run_id) if run_id else "computed",
        run_state=snapshot,
        token_usage=usage,
    )


@router.post("/campaign/stream")
async def stream_campaign(payload: CampaignQuery) -> StreamingResponse:
    """Server-sent events for the chat UI: one event per graph update, then a `done` event."""
    _require_api_key()
    session_id = _ensure_session(payload.session_id, payload.query)
    prompt = _build_prompt(payload, session_id)

    async def events() -> AsyncIterator[str]:
        agent = _agent_instance()
        tracer = AgentRunLogger(label=f"campaign/stream session={session_id}")
        tracer.start()
        # `updates` mode carries progress, not the reply, so keep the latest assistant
        # text as it goes past — the `done` event is where the UI reads the answer.
        answer = ""
        try:
            async for update in agent.astream(
                {"messages": [{"role": "user", "content": prompt}]},
                config=_run_config(session_id, tracer),
                stream_mode="updates",
            ):
                for node, delta in update.items():
                    yield _sse("update", {"node": node, "summary": _summarize(delta)})
                    if isinstance(delta, dict):
                        for message in delta.get("messages") or []:
                            if text := _message_text(message):
                                answer = text
        except ModelError as exc:
            log_error(f"stream aborted: {type(exc).__name__}")
            tracer.log_summary()
            err = _provider_error(exc)
            yield _sse("error", {"status": err.status_code, "detail": err.detail})
            return
        except Exception as exc:  # noqa: BLE001 - any failure must reach the UI, not hang the stream
            log_error(f"stream aborted: {type(exc).__name__}: {exc}")
            tracer.log_summary()
            yield _sse("error", {"status": 500, "detail": str(exc)})
            return

        usage = tracer.log_summary()
        run_id = _latest_run_for_session(session_id)
        yield _sse(
            "done",
            {
                "session_id": session_id,
                "session_title": session_titles.title_of(session_id),
                "run_id": run_id,
                "answer": answer,
                "run_state": run_state.snapshot(run_id) if run_id else None,
                "token_usage": usage,
            },
        )

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _summarize(delta: Any) -> dict[str, Any]:
    """Keep SSE payloads small — the UI wants progress, not full message content."""
    if not isinstance(delta, dict):
        return {}
    messages = delta.get("messages") or []
    out: dict[str, Any] = {"messages": len(messages)}
    for message in messages:
        if calls := getattr(message, "tool_calls", None):
            out["tool_calls"] = [c.get("name") for c in calls]
        if getattr(message, "type", None) == "tool":
            out["tool_result_for"] = getattr(message, "name", None)
    return out


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Full run record: spec, artifact references, optimization result, validation."""
    record = local_db.get_record(local_db.RUNS, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run '{run_id}'")
    return record


@router.get("/runs")
def list_runs(session_id: str | None = None) -> list[dict]:
    runs = local_db.list_records(local_db.RUNS)
    if session_id:
        runs = [r for r in runs if r.get("session_id") == session_id]
    return [run_state.snapshot(r["id"]) for r in runs]


@router.get("/runs/{run_id}/artifacts/{kind}", response_model=ArtifactRowsOut)
def get_artifact_rows(
    run_id: str,
    kind: str,
    limit: Annotated[int, Query(ge=1, le=1000)] = 25,
    screen_ids: Annotated[list[str] | None, Query()] = None,
) -> ArtifactRowsOut:
    """Rows of one run artifact, for the inspector panel.

    A UI-only read path: the artifact stays on disk and out of every agent's context.
    Rows come back in the artifact's own order, which is already ranked for
    `screen_candidates`.

    Pass `screen_ids` to pull the rows for specific screens — the ones in a package sit
    anywhere in the artifact, so a plain top-N slice would mostly miss them. Filtering
    happens before `limit`.
    """
    try:
        ref = run_state.get_artifact(run_id, kind)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No run '{run_id}'") from exc
    if ref is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Run '{run_id}' has no '{kind}' artifact yet — the producing stage has not run."
            ),
        )

    try:
        rows = artifact_store.read_rows(ref, limit=limit, screen_ids=screen_ids)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=410,
            detail=f"Artifact '{ref.artifact_id}' is recorded but its file is gone.",
        ) from exc

    return ArtifactRowsOut(
        run_id=run_id,
        kind=kind,
        artifact_id=ref.artifact_id,
        provenance=ref.provenance,
        total_rows=ref.rows,
        returned_rows=len(rows),
        columns=ref.columns,
        summary=ref.summary,
        rows=rows,
    )
