"""The three specialist subagents.

Each is a thin delegation shell over its tool module. The tools are where the real work
happens (or, today, where the stub lives) — so integrating a teammate's implementation
means replacing a tool module, not touching this file.
"""

from __future__ import annotations

from deepagents import SubAgent
from langchain_core.language_models import BaseChatModel

from app.agents.prompts import DATA_AGENT_PROMPT, ML_AGENT_PROMPT, OR_AGENT_PROMPT
from app.tools import data_agent_tools, ml_agent_tools, or_agent_tools

DATA_AGENT = "data_agent"
ML_AGENT = "ml_agent"
OR_AGENT = "or_agent"


def build_subagents(model: str | BaseChatModel) -> list[SubAgent]:
    """Specs for the Data, ML and OR agents, in pipeline order."""
    return [
        SubAgent(
            name=DATA_AGENT,
            description=(
                "Data Intelligence Agent. Resolves campaign geography against real "
                "inventory, engineers screen-level features, and returns a ranked "
                "candidate pool as an artifact reference. Delegate stage 2, before any "
                "forecasting or optimization."
            ),
            system_prompt=DATA_AGENT_PROMPT,
            tools=data_agent_tools.TOOLS,
            model=model,
        ),
        SubAgent(
            name=ML_AGENT,
            description=(
                "ML / Forecasting Agent. Forecasts demand and recommends pricing for "
                "the candidate screens, returning a screen_economics artifact "
                "reference. Delegate stage 3, after the Data Agent and before the OR "
                "Agent."
            ),
            system_prompt=ML_AGENT_PROMPT,
            tools=ml_agent_tools.TOOLS,
            model=model,
        ),
        SubAgent(
            name=OR_AGENT,
            description=(
                "OR / Optimization Agent. Selects the inventory package that maximizes "
                "the campaign objective under budget, availability, geography and date "
                "constraints, or returns an explicit infeasibility report. Delegate "
                "stage 4, after the ML Agent."
            ),
            system_prompt=OR_AGENT_PROMPT,
            tools=or_agent_tools.TOOLS,
            model=model,
        ),
    ]
