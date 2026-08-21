"""Plain print-based logging for agent execution traces.

Deliberately prints rather than using the `logging` module so a trace always shows up on
stdout under `uvicorn`, `pytest -s`, or a bare script with no handler configuration.

Levels: [INFO] pipeline milestones, [DEBUG] per-call detail, [ERROR] failures.
Set LOG_DEBUG=0 to silence [DEBUG] and keep the trace readable.
"""

from __future__ import annotations

import os
import sys
import threading

_print_lock = threading.Lock()

DEBUG_ENABLED = os.getenv("LOG_DEBUG", "1") not in ("0", "false", "False")
MAX_PREVIEW = 220


def _emit(level: str, message: str) -> None:
    with _print_lock:  # keeps concurrent agent branches from interleaving mid-line
        print(f"[{level}] {message}", file=sys.stdout, flush=True)


def info(message: str) -> None:
    _emit("INFO", message)


def debug(message: str) -> None:
    if DEBUG_ENABLED:
        _emit("DEBUG", message)


def error(message: str) -> None:
    _emit("ERROR", message)


def preview(value: object, limit: int = MAX_PREVIEW) -> str:
    """One-line, length-capped rendering of arbitrary values for log lines."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit]}..."
