"""Execution tracing and token accounting for an agent run.

`AgentRunLogger` is a LangChain callback handler. Passing it once via
`config={"callbacks": [logger]}` covers the whole tree — the Master Agent's own model
calls and tools, plus every nested subagent call — because callbacks propagate into child
runs.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.logging_utils import debug, error, info, preview

# Tool names that represent a delegation to a specialist rather than a data operation.
DELEGATION_TOOLS = {"task"}


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
    """Prints an [INFO]/[DEBUG]/[ERROR] trace and totals tokens for one agent run."""

    def __init__(self, label: str = "run") -> None:
        self.label = label
        self.usage = TokenUsage()
        self.started_at = time.time()
        self._tool_calls = 0
        self._delegations = 0
        self._errors = 0
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        info(f"=== agent run start [{self.label}] ===")

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    # --- model calls -------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        turns = len(messages[0]) if messages else 0
        debug(f"model call -> {self._model_name(serialized, kwargs)} ({turns} messages in context)")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        call = self.usage.add(self._usage_from(response))
        debug(
            f"model done <- in={call['input_tokens']} out={call['output_tokens']} "
            f"(reasoning={call['reasoning_tokens']}) | running total="
            f"{self.usage.as_dict()['total_tokens']}"
        )

    def on_llm_error(self, err: BaseException, **kwargs: Any) -> None:
        with self._lock:
            self._errors += 1
        error(f"model call failed: {type(err).__name__}: {preview(err, 300)}")

    # --- tools and delegation ---------------------------------------------

    def on_tool_start(self, serialized, input_str: str, **kwargs: Any) -> None:
        name = self._tool_name(serialized, kwargs)
        with self._lock:
            self._tool_calls += 1
            if name in DELEGATION_TOOLS:
                self._delegations += 1
        if name in DELEGATION_TOOLS:
            info(f"delegating -> {preview(input_str, 160)}")
        else:
            info(f"tool -> {name}({preview(input_str, 160)})")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        debug(f"tool done <- {preview(output)}")

    def on_tool_error(self, err: BaseException, **kwargs: Any) -> None:
        with self._lock:
            self._errors += 1
        error(f"tool failed: {type(err).__name__}: {preview(err, 300)}")

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
