"""The agent trace is a product surface, so its extraction logic is tested like one.

No model calls: the callbacks are driven directly with the payload shapes the provider
actually sends. These tests exist because every assertion here corresponds to a way the
trace has been, or could silently become, useless — truncated agent text, a `Command(...)`
repr where a specialist's report should be, or a specialist's output attributed to the
Master.
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.types import Command

from app.agents.tracing import AgentRunLogger
from app.logging_utils import BODY_INDENT

USAGE = {
    "input_tokens": 4102,
    "output_tokens": 318,
    "total_tokens": 4420,
    "output_token_details": {"reasoning": 210},
    "input_token_details": {"cache_read": 3800},
}


def _result(message: AIMessage) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _ai(content, tool_calls=None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [], usage_metadata=USAGE)


def _flat(out: str) -> str:
    """Captured stdout with the block gutter stripped and whitespace collapsed.

    `block` prefixes body lines with `BODY_INDENT` and wraps long ones, so a raw flatten
    would splice `|` between words and defeat a substring check.
    """
    lines = [line.removeprefix(BODY_INDENT) for line in out.splitlines()]
    return " ".join(" ".join(lines).split())


@pytest.fixture
def tracer() -> AgentRunLogger:
    return AgentRunLogger(label="test")


def test_gemini_content_blocks_are_unwrapped_not_repr_printed(tracer, capsys):
    """Gemini returns list[dict], so a naive str() would print a Python list repr."""
    tracer.on_llm_end(
        _result(
            _ai(
                [
                    {"type": "thinking", "thinking": "I should resolve geography first."},
                    {"type": "text", "text": "Resolving the named places."},
                ]
            )
        ),
        run_id=str(uuid.uuid4()),
    )
    out = capsys.readouterr().out
    assert "Resolving the named places." in out
    assert "I should resolve geography first." in out
    # The giveaway that block extraction regressed.
    assert "'type':" not in out


def test_agent_text_is_not_truncated(tracer, capsys):
    """The whole point: an agent's answer must survive to the log in full.

    MAX_PREVIEW is 220 characters, which is where this used to be cut.
    """
    answer = "Recommended package. " + ("audience deduplication matters. " * 120)
    tracer.on_llm_end(_result(_ai(answer)), run_id=str(uuid.uuid4()))
    assert len(answer) > 3000
    # Every word of it survives, once the gutter and wrapping are undone.
    assert " ".join(answer.split()) in _flat(capsys.readouterr().out)


def test_token_counts_are_reported_for_every_call(tracer, capsys):
    tracer.on_llm_end(_result(_ai("done")), run_id=str(uuid.uuid4()))
    out = capsys.readouterr().out
    assert "in=4,102" in out
    assert "out=318" in out
    assert "reasoning=210" in out
    assert tracer.usage.as_dict()["total_tokens"] == 4420


def test_tool_calls_the_model_asked_for_are_shown_with_arguments(tracer, capsys):
    tracer.on_llm_end(
        _result(
            _ai(
                "",
                tool_calls=[
                    {
                        "name": "create_campaign_spec",
                        "args": {"budget": 50000.0, "city_ids": ["LH"]},
                        "id": "c1",
                    }
                ],
            )
        ),
        run_id=str(uuid.uuid4()),
    )
    out = capsys.readouterr().out
    assert "create_campaign_spec" in out
    assert "budget: 50000.0" in out
    assert '["LH"]' in out


def test_delegation_shows_the_instruction_and_the_report_in_full(tracer, capsys):
    """The two hand-offs a multi-agent trace is read for."""
    instruction = "Price the candidates for run-1. " + ("Report occupancy precisely. " * 40)
    report = "Priced 250 screens. " + ("Mean occupancy 0.41. " * 40)
    deleg = str(uuid.uuid4())

    tracer.on_tool_start(
        {"name": "task"},
        "",
        run_id=deleg,
        name="task",
        inputs={"subagent_type": "ml_agent", "description": instruction},
    )
    tracer.on_tool_end(
        Command(update={"messages": [ToolMessage(content=report, tool_call_id="d1")]}),
        run_id=deleg,
    )

    flattened = _flat(capsys.readouterr().out)
    assert "master ──▶ ml_agent" in flattened
    assert "ml_agent ──▶ master" in flattened
    assert " ".join(instruction.split()) in flattened
    assert " ".join(report.split()) in flattened
    # A Command repr instead of the report is the failure mode this guards.
    assert "Command(" not in flattened
    assert tracer.summary()["delegations"] == 1


def test_specialist_output_is_attributed_to_the_specialist(tracer, capsys):
    """A model call nested under a delegation belongs to the delegate, not the Master.

    Resolution walks the run_id parent chain, so this also covers a grandchild run — the
    real shape, since the specialist's graph sits between the task tool and its model.
    """
    deleg = str(uuid.uuid4())
    tracer.on_tool_start(
        {"name": "task"},
        "",
        run_id=deleg,
        name="task",
        inputs={"subagent_type": "or_agent", "description": "Optimize run-1."},
    )
    graph = str(uuid.uuid4())
    tracer.on_chat_model_start({}, [[1]], run_id=graph, parent_run_id=deleg)
    model = str(uuid.uuid4())
    tracer.on_llm_end(
        _result(_ai("The MILP solved to optimality.")),
        run_id=model,
        parent_run_id=graph,
    )

    out = capsys.readouterr().out
    assert "or_agent · says" in out
    assert "master · says" not in out


def test_master_owns_events_with_no_delegation_ancestor(tracer, capsys):
    tracer.on_llm_end(_result(_ai("Writing the recommendation.")), run_id=str(uuid.uuid4()))
    assert "master · says" in capsys.readouterr().out


def test_a_response_with_neither_text_nor_tool_call_is_stated_not_silent(tracer, capsys):
    tracer.on_llm_end(_result(_ai("")), run_id=str(uuid.uuid4()))
    assert "returned no text and no tool call" in capsys.readouterr().out
