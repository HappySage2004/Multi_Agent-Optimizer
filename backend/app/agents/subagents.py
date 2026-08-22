"""Aggregator for the specialist subagents.

One module per agent — `ml_agent.py`, `or_agent.py` — each holding its name, description,
system prompt and `build()`. This file only assembles them in pipeline order, so
`master.py` has a single import regardless of how many specialists exist.

There is no data/relevance subagent. Stage 2 is the deterministic audience relevance engine
in `app/tools/relevance_tools.py`, which the Master Agent calls directly: an LLM shell
around a fixed calculation adds latency and a chance to paraphrase numbers wrongly, and
SOLUTION.md section 31.2 puts calculation in tools rather than agents. Delegation is
reserved for the stages where a specialist genuinely reasons about its own output.

The name constants are re-exported because tests and logging refer to them.
"""

from __future__ import annotations

from deepagents import SubAgent
from langchain_core.language_models import BaseChatModel

from app.agents import ml_agent, or_agent

ML_AGENT = ml_agent.NAME
OR_AGENT = or_agent.NAME

__all__ = ["ML_AGENT", "OR_AGENT", "build_subagents"]


def build_subagents(model: str | BaseChatModel) -> list[SubAgent]:
    """Specs for the ML and OR agents, in pipeline order."""
    return [
        ml_agent.build(model),
        or_agent.build(model),
    ]
