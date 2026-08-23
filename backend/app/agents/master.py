"""The Master Deep Agent — the only component that orchestrates end to end."""

from __future__ import annotations

from functools import lru_cache

from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph

from app.agents import providers
from app.agents.prompts import MASTER_SYSTEM_PROMPT
from app.agents.providers import ModelSelection
from app.agents.subagents import build_subagents
from app.logging_utils import info
from app.tools import master_tools, relevance_tools

# deepagents also accepts a "provider:model" string (e.g. "google_genai:gemini-3.5-flash-lite"),
# which it resolves through init_chat_model. We build the clients explicitly instead --
# see app/agents/providers.py -- so credentials come from settings rather than from
# provider-specific env vars, and so the provider is selectable per request.


def build_master_agent(
    selection: ModelSelection | None = None,
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Compile the Master Agent with the ML and OR specialists attached.

    The Master's own tool surface carries stage 2: `relevance_tools` is the deterministic
    audience relevance engine, called directly rather than wrapped in a subagent.

    Args:
        selection: Which provider and per-tier models to run on. Defaults to
            `providers.resolve()`, i.e. the configured default provider.
        checkpointer: LangGraph checkpointer, for multi-turn sessions.
    """
    selection = selection or providers.resolve()
    master = providers.build_chat_model(selection, selection.master_model)
    specialist = providers.build_chat_model(selection, selection.specialist_model)

    info(
        f"master agent: provider={selection.provider}, model={selection.master_model}, "
        f"specialists={selection.specialist_model}, "
        f"subagents=ml_agent/or_agent, master_tools+relevance_engine"
    )
    return create_deep_agent(
        model=master,
        tools=[*master_tools.TOOLS, *relevance_tools.TOOLS],
        system_prompt=MASTER_SYSTEM_PROMPT,
        subagents=build_subagents(specialist),
        checkpointer=checkpointer,
        name="master_agent",
    )


@lru_cache(maxsize=1)
def get_master_agent() -> CompiledStateGraph:
    """Process-wide singleton on the default selection. Compiling the graph is not free;
    the agent is stateless across runs because all run state lives in localDB and the
    artifact store.

    The API keeps its own per-selection cache (`api/campaign.py`), because there the
    provider is a per-request choice.
    """
    return build_master_agent()
