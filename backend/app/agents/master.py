"""The Master Deep Agent — the only component that orchestrates end to end."""

from __future__ import annotations

from functools import lru_cache

from deepagents import create_deep_agent
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph

from app.agents.prompts import MASTER_SYSTEM_PROMPT
from app.agents.subagents import build_subagents
from app.config import get_settings
from app.logging_utils import debug, info
from app.tools import master_tools

# deepagents also accepts a "provider:model" string (e.g. "google_genai:gemini-3.5-flash-lite"),
# which it resolves through init_chat_model. We build the client explicitly instead so the
# API key comes from settings rather than a GOOGLE_API_KEY env var, and so token limits
# are pinned.
MAX_OUTPUT_TOKENS = 8_192


@lru_cache(maxsize=1)
def _rate_limiter() -> InMemoryRateLimiter:
    """One limiter shared by the Master and all specialists.

    It must be shared: the free-tier cap is per project+model, so three separate limiters
    would each think they had the full budget and together exceed it.
    """
    per_second = get_settings().model_requests_per_minute / 60.0
    return InMemoryRateLimiter(
        requests_per_second=per_second,
        check_every_n_seconds=0.1,
        # Allow a small burst so short exchanges are not needlessly slowed.
        max_bucket_size=3,
    )


def _chat_model(model_id: str) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Set it in the repo-root .env "
            "(or backend/.env)."
        )
    debug(
        f"chat model {model_id} (max_output_tokens={MAX_OUTPUT_TOKENS}, "
        f"rate_limit={settings.model_requests_per_minute:g}/min shared, "
        f"timeout={settings.request_timeout_seconds:g}s, retries={settings.model_max_retries})"
    )
    return ChatGoogleGenerativeAI(
        model=model_id,
        api_key=settings.gemini_api_key,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.model_max_retries,
        rate_limiter=_rate_limiter(),
        # Thinking is left at the model default: the model reasons on its own, and
        # pinning `thinking_level` on this client version stalls the request.
    )


def build_master_agent(
    *,
    master_model: str | None = None,
    specialist_model: str | None = None,
    checkpointer=None,
) -> CompiledStateGraph:
    """Compile the Master Agent with the Data, ML and OR specialists attached.

    Args:
        master_model: Override the orchestrator model id.
        specialist_model: Override the model id used by all three specialists.
        checkpointer: LangGraph checkpointer, for multi-turn sessions.
    """
    settings = get_settings()
    master = _chat_model(master_model or settings.master_model_id)
    specialist = _chat_model(specialist_model or settings.specialist_model_id)

    info(
        f"master agent: model={master_model or settings.master_model_id}, "
        f"specialists={specialist_model or settings.specialist_model_id}, "
        f"subagents=data_agent/ml_agent/or_agent"
    )
    return create_deep_agent(
        model=master,
        tools=master_tools.TOOLS,
        system_prompt=MASTER_SYSTEM_PROMPT,
        subagents=build_subagents(specialist),
        checkpointer=checkpointer,
        name="master_agent",
    )


@lru_cache(maxsize=1)
def get_master_agent() -> CompiledStateGraph:
    """Process-wide singleton. Compiling the graph is not free; the agent is stateless
    across runs because all run state lives in localDB and the artifact store."""
    return build_master_agent()
