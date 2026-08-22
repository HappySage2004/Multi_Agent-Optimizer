"""Execution tracing and token accounting for an agent run.

`AgentRunLogger` is a LangChain callback handler. Passing it once via
`config={"callbacks": [logger]}` covers the whole tree — the Master Agent's own model
calls and tools, plus every nested subagent call — because callbacks propagate into child
runs.

WHAT THIS TRACE IS FOR. It answers "what did the agents decide, and why", not "what did
the code do". So every model response is printed IN FULL — its reasoning, its answer text
and the tool calls it chose — and so is every delegation instruction and every specialist
report. The engine internals have their own `[DEBUG]` lines in `app/tools/` and
`app/optimize/`; this module deliberately does not repeat them.

ATTRIBUTION. A flat trace of a multi-agent run is unreadable: with the Master and two
specialists interleaved you cannot tell who said what. Every callback carries `run_id` and
`parent_run_id`, so this handler keeps a parent chain and resolves each event to the agent
that owns it, then indents specialist lines one level. The mapping is seeded when the
built-in `task` tool fires, because its `subagent_type` argument names the delegate — no
dependency on langgraph or deepagents internals, which is the point.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.logging_utils import block, debug, error, info, preview

# Tool names that represent a delegation to a specialist rather than a data operation.
DELEGATION_TOOLS = {"task"}

MASTER = "master"

# Tool results worth printing in full rather than previewing. These are the ones whose text
# the Master is required to reason over or repeat accurately; the rest are bulky artifact
# summaries that `preview` handles fine.
VERBOSE_TOOL_RESULTS = {
    "verify_package",
    "inspect_package",
    "check_explanations",
    "describe_relevance_model",
    "describe_pricing_model",
    "compare_objectives",
}


class TokenUsage:
    """Thread-safe running total of model token consumption."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.reasoning_tokens = 0
        self.cache_read_tokens = 0
        self.model_calls = 0

    def add(self, usage: dict[str, Any] | None) -> dict[str, int]:
        """Accumulate one response's usage_metadata; returns just that call's numbers."""
        usage = usage or {}
        call = {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "reasoning_tokens": int(
                (usage.get("output_token_details") or {}).get("reasoning") or 0
            ),
            "cache_read_tokens": int(
                (usage.get("input_token_details") or {}).get("cache_read") or 0
            ),
        }
        with self._lock:
            self.model_calls += 1
            self.input_tokens += call["input_tokens"]
            self.output_tokens += call["output_tokens"]
            # Some providers omit total; derive it so the summary always reconciles.
            self.total_tokens += call["total_tokens"] or (
                call["input_tokens"] + call["output_tokens"]
            )
            self.reasoning_tokens += call["reasoning_tokens"]
            self.cache_read_tokens += call["cache_read_tokens"]
        return call

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "model_calls": self.model_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cache_read_tokens": self.cache_read_tokens,
            }


class AgentRunLogger(BaseCallbackHandler):
    """Prints an agent-level trace and totals tokens for one agent run."""

    def __init__(self, label: str = "run") -> None:
        self.label = label
        self.usage = TokenUsage()
        self.started_at = time.time()
        self._tool_calls = 0
        self._delegations = 0
        self._errors = 0
        self._lock = threading.Lock()

        # run_id -> parent run_id, and run_id -> owning agent name. Together these resolve
        # any nested event back to the agent responsible for it.
        self._parent: dict[str, str] = {}
        self._agent: dict[str, str] = {}
        # tool run_id -> (tool name, agent) so on_tool_end can name what finished.
        self._pending_tool: dict[str, tuple[str, str]] = {}

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        info(f"=== agent run start [{self.label}] ===")

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    # --- model calls -------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        self._link(kwargs)
        agent = self._agent_for(kwargs)
        turns = len(messages[0]) if messages else 0
        debug(
            f"{self._indent(agent)}{agent} · model call -> "
            f"{self._model_name(serialized, kwargs)} ({turns} messages in context)"
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._link(kwargs)
        agent = self._agent_for(kwargs)
        call = self.usage.add(self._usage_from(response))
        pad = self._indent(agent)

        message = self._message_of(response)
        reasoning = self._reasoning_text(message)
        answer = self._answer_text(message)
        calls = self._tool_call_lines(message)

        # The lines the whole trace exists for: what the model actually produced, in full,
        # printed between `model call` and `model done`. All three are [INFO], including
        # reasoning — it is agent content, not code detail, so LOG_DEBUG=0 must not hide it.
        # LOG_DEBUG=0 is how you get the agent narrative WITHOUT the engine internals.
        if reasoning:
            block(f"{pad}{agent} · reasoning", reasoning)
        if answer:
            block(f"{pad}{agent} · says", answer)
        if calls:
            block(f"{pad}{agent} · wants to call", "\n".join(calls))
        if not (reasoning or answer or calls):
            info(f"{pad}{agent} · returned no text and no tool call")

        info(
            f"{pad}{agent} · model done <- in={call['input_tokens']:,} "
            f"out={call['output_tokens']:,} reasoning={call['reasoning_tokens']:,} "
            f"| run total in={self.usage.input_tokens:,} out={self.usage.output_tokens:,} "
            f"({self.usage.total_tokens:,})"
        )

    def on_llm_error(self, err: BaseException, **kwargs: Any) -> None:
        self._link(kwargs)
        with self._lock:
            self._errors += 1
        agent = self._agent_for(kwargs)
        error(
            f"{self._indent(agent)}{agent} · model call FAILED: "
            f"{type(err).__name__}: {preview(err, 400)}"
        )

    # --- tools and delegation ---------------------------------------------

    def on_tool_start(self, serialized, input_str: str, **kwargs: Any) -> None:
        self._link(kwargs)
        name = self._tool_name(serialized, kwargs)
        caller = self._agent_for(kwargs)
        run_id = self._run_id(kwargs)

        with self._lock:
            self._tool_calls += 1
            if name in DELEGATION_TOOLS:
                self._delegations += 1

        args = self._tool_args(input_str, kwargs)

        if name in DELEGATION_TOOLS:
            target = str(args.get("subagent_type") or "subagent")
            instruction = str(args.get("description") or input_str)
            # Claim this tool run for the delegate, so every model call and tool call the
            # specialist makes inside it is attributed to the specialist, not the Master.
            if run_id:
                self._agent[run_id] = target
                self._pending_tool[run_id] = (name, target)
            block(f"{self._indent(caller)}{caller} ──▶ {target} · DELEGATES", instruction)
            return

        if run_id:
            self._pending_tool[run_id] = (name, caller)
        pad = self._indent(caller)
        rendered = self._render_args(args) or preview(input_str, 400)
        block(f"{pad}{caller} · calls {name}", rendered)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._link(kwargs)
        run_id = self._run_id(kwargs)
        name, owner = self._pending_tool.pop(run_id, ("tool", self._agent_for(kwargs)))

        if name in DELEGATION_TOOLS:
            # `owner` is the specialist. Print its report back to the Master in full — this
            # is the hand-off the Master's next decision is based on.
            report = self._delegation_report(output)
            block(f"{self._indent(MASTER)}{owner} ──▶ {MASTER} · REPORTS", report)
            return

        pad = self._indent(owner)
        text = self._tool_output_text(output)
        if name in VERBOSE_TOOL_RESULTS:
            block(f"{pad}{name} · result", text, level="DEBUG")
        else:
            debug(f"{pad}{name} · result <- {preview(text, 400)}")

    def on_tool_error(self, err: BaseException, **kwargs: Any) -> None:
        self._link(kwargs)
        with self._lock:
            self._errors += 1
        run_id = self._run_id(kwargs)
        name, owner = self._pending_tool.pop(run_id, ("tool", self._agent_for(kwargs)))
        error(
            f"{self._indent(owner)}{owner} · {name} FAILED: "
            f"{type(err).__name__}: {preview(err, 400)}"
        )

    # --- summary -----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            **self.usage.as_dict(),
            "tool_calls": self._tool_calls,
            "delegations": self._delegations,
            "errors": self._errors,
            "elapsed_seconds": round(self.elapsed, 1),
        }

    def log_summary(self) -> dict[str, Any]:
        """Print the end-of-run token report and return it."""
        s = self.summary()
        info(f"=== agent run end [{self.label}] in {s['elapsed_seconds']}s ===")
        info(
            f"TOKENS  total={s['total_tokens']:,}  input={s['input_tokens']:,}  "
            f"output={s['output_tokens']:,}  reasoning={s['reasoning_tokens']:,}  "
            f"cache_read={s['cache_read_tokens']:,}"
        )
        info(
            f"CALLS   model={s['model_calls']}  tools={s['tool_calls']}  "
            f"delegations={s['delegations']}  errors={s['errors']}"
        )
        return s

    # --- attribution -------------------------------------------------------

    @staticmethod
    def _run_id(kwargs: dict) -> str:
        return str(kwargs.get("run_id") or "")

    def _link(self, kwargs: dict) -> None:
        """Record this run's parent so `_agent_for` can walk up to the owning agent."""
        run_id = self._run_id(kwargs)
        parent = kwargs.get("parent_run_id")
        if run_id and parent:
            self._parent.setdefault(run_id, str(parent))

    def _agent_for(self, kwargs: dict) -> str:
        """The agent that owns this event: the nearest ancestor claimed by a delegation.

        Walks the parent chain rather than reading langgraph metadata, so it keeps working
        across provider and framework versions. The chain is short (Master -> task ->
        subagent graph -> model) and the loop is bounded against a malformed cycle.
        """
        node: str | None = self._run_id(kwargs)
        if node and node in self._agent:
            return self._agent[node]
        parent = kwargs.get("parent_run_id")
        node = str(parent) if parent else None
        for _ in range(64):
            if not node:
                return MASTER
            if (owner := self._agent.get(node)) is not None:
                return owner
            node = self._parent.get(node)
        return MASTER

    @staticmethod
    def _indent(agent: str) -> str:
        """Specialists sit one level in, so the Master's spine reads down the left edge."""
        return "" if agent == MASTER else "  "

    # --- message extraction ------------------------------------------------

    @staticmethod
    def _message_of(response: LLMResult) -> Any:
        for generations in response.generations:
            for generation in generations:
                if (message := getattr(generation, "message", None)) is not None:
                    return message
        return None

    @staticmethod
    def _blocks(message: Any) -> list[Any]:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            return content
        return [content] if content else []

    @classmethod
    def _answer_text(cls, message: Any) -> str:
        """Assistant text only — thinking and tool-use blocks excluded.

        Gemini returns content blocks rather than a bare string, so a naive `str(content)`
        prints a Python list repr full of metadata. Mirrors `api/campaign._message_text`.
        """
        if message is None:
            return ""
        parts: list[str] = []
        for item in cls._blocks(message):
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in (None, "text"):
                parts.append(str(item.get("text") or ""))
        return "\n".join(p for p in parts if p.strip()).strip()

    @classmethod
    def _reasoning_text(cls, message: Any) -> str:
        """Whatever the provider exposed of the model's private reasoning, if anything.

        Gemini bills reasoning tokens but usually does not return the text; when it does,
        the block type varies by provider and version, so several spellings are accepted.
        """
        if message is None:
            return ""
        parts: list[str] = []
        for item in cls._blocks(message):
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("thinking", "reasoning", "reasoning_content"):
                parts.append(
                    str(item.get("thinking") or item.get("reasoning") or item.get("text") or "")
                )
        extra = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content")
        if isinstance(extra, str):
            parts.append(extra)
        return "\n".join(p for p in parts if p.strip()).strip()

    @classmethod
    def _tool_call_lines(cls, message: Any) -> list[str]:
        """The tool calls this response asked for, with their arguments rendered readably."""
        raw = getattr(message, "tool_calls", None) or []
        lines: list[str] = []
        for call in raw:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or "tool"
            args = call.get("args") or {}
            if name in DELEGATION_TOOLS:
                target = args.get("subagent_type") or "subagent"
                lines.append(f"{name} -> {target}")
                if description := args.get("description"):
                    lines.append(cls._indent_arg(str(description)))
                continue
            lines.append(f"{name}(")
            rendered = cls._render_args(args)
            if rendered:
                lines.append(cls._indent_arg(rendered))
            lines.append(")")
        return lines

    @staticmethod
    def _indent_arg(text: str) -> str:
        return "\n".join(f"    {line}" for line in text.splitlines())

    @staticmethod
    def _render_args(args: Any) -> str:
        """One `key: value` line per argument. Values are JSON so lists stay readable."""
        if not isinstance(args, dict) or not args:
            return ""
        lines: list[str] = []
        for key, value in args.items():
            if isinstance(value, str):
                rendered = value
            else:
                try:
                    rendered = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    rendered = str(value)
            lines.append(f"{key}: {rendered}")
        return "\n".join(lines)

    @classmethod
    def _tool_args(cls, input_str: str, kwargs: dict) -> dict:
        """The tool's arguments as a dict.

        `on_tool_start` passes `inputs` on recent langchain-core versions and only the
        stringified form on older ones, so both are handled rather than assumed.
        """
        inputs = kwargs.get("inputs")
        if isinstance(inputs, dict):
            return inputs
        text = (input_str or "").strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _delegation_report(cls, output: Any) -> str:
        """The specialist's closing message, dug out of whatever the `task` tool returned.

        deepagents' `task` returns a langgraph `Command` carrying a state update, so the
        report is the content of the ToolMessage inside it — not `str(output)`, which is a
        `Command(...)` repr. Falls back through the plainer shapes so a framework change
        degrades to a readable line instead of nothing.
        """
        update = getattr(output, "update", None)
        if isinstance(update, dict):
            messages = update.get("messages") or []
            for message in reversed(messages):
                if text := cls._message_content_text(message):
                    return text
        if text := cls._message_content_text(output):
            return text
        return preview(output, 2000)

    @classmethod
    def _message_content_text(cls, message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return cls._answer_text(message)
        return ""

    @classmethod
    def _tool_output_text(cls, output: Any) -> str:
        if text := cls._message_content_text(output):
            return text
        return str(output)

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _usage_from(response: LLMResult) -> dict[str, Any] | None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if usage := getattr(message, "usage_metadata", None):
                    return usage
        # Fall back to the provider-level payload when the message carries no metadata.
        return (response.llm_output or {}).get("usage_metadata")

    @staticmethod
    def _model_name(serialized: dict | None, kwargs: dict) -> str:
        invocation = kwargs.get("invocation_params") or {}
        return (
            invocation.get("model")
            or invocation.get("model_name")
            or (serialized or {}).get("name")
            or "model"
        )

    @staticmethod
    def _tool_name(serialized: dict | None, kwargs: dict) -> str:
        return kwargs.get("name") or (serialized or {}).get("name") or "tool"
