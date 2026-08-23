"""Plain print-based logging for agent execution traces.

Deliberately prints rather than using the `logging` module so a trace always shows up on
stdout under `uvicorn`, `pytest -s`, or a bare script with no handler configuration.

Levels: [INFO] pipeline milestones and everything an agent said, [DEBUG] code-facing detail
(row counts, solver dimensions, raw tool results), [ERROR] failures.

**LOG_DEBUG=0 gives you the agent narrative on its own** — every model response, delegation
and specialist report, with none of the engine internals. That is the setting to read a run
by. LOG_DEBUG=1 adds the code trace underneath it for debugging the pipeline itself.

TWO KINDS OF LINE, AND THEY ARE READ BY DIFFERENT PEOPLE:

* `info` / `debug` / `error` are one-liners about what the CODE did — stage milestones,
  row counts, solver status. `preview` caps them, because nobody reads a 4,000-character
  DataFrame repr in a log.
* `block` is for what an AGENT SAID — model output, a delegation instruction, a
  specialist's report. It prints the text in full, wrapped and indented under a header, and
  is NOT capped by `MAX_PREVIEW`. Truncating an agent's reasoning to 220 characters is what
  makes an agent trace useless: the interesting part is always past the cut.

Set LOG_MAX_BLOCK to bound `block` output if a run ever produces something pathological;
the default is generous on purpose and the truncation, when it happens, is explicit.
"""

from __future__ import annotations

import contextlib
import os
import sys
import textwrap
import threading

_print_lock = threading.Lock()

DEBUG_ENABLED = os.getenv("LOG_DEBUG", "1") not in ("0", "false", "False")

# One-liner cap, for code-facing detail.
MAX_PREVIEW = 220

# Agent-facing cap. Deliberately ~40x MAX_PREVIEW: a model's answer or a specialist's
# report is the payload of the trace, not noise around it.
MAX_BLOCK = int(os.getenv("LOG_MAX_BLOCK", "8000"))

# Wrap width for block bodies. Keeps long paragraphs readable in a terminal without
# destroying the model's own line breaks, which are meaningful in markdown answers.
WRAP_WIDTH = int(os.getenv("LOG_WRAP_WIDTH", "100"))

BODY_INDENT = "         | "


def _configure_stdout() -> None:
    """Force UTF-8 on stdout so agent text survives the trip to the console.

    Model output is full of em dashes, arrows and curly quotes. On a Windows console the
    default cp1252 encoding turns each one into a replacement character, which is precisely
    the text this module exists to show. Best-effort: if the stream cannot be reconfigured
    we fall back to replacement rather than letting a log line raise.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is None:
        return
    # Detached, redirected to a non-text sink, or already fixed by the host.
    with contextlib.suppress(ValueError, OSError):
        reconfigure(encoding="utf-8", errors="replace")


_configure_stdout()


def _emit(level: str, message: str) -> None:
    with _print_lock:  # keeps concurrent agent branches from interleaving mid-line
        print(f"[{level}] {message}", file=sys.stdout, flush=True)


def info(message: str) -> None:
    _emit("INFO", message)


def debug(message: str) -> None:
    if DEBUG_ENABLED:
        _emit("DEBUG", message)


def warning(message: str) -> None:
    _emit("WARN", message)


def error(message: str) -> None:
    _emit("ERROR", message)


def preview(value: object, limit: int = MAX_PREVIEW) -> str:
    """One-line, length-capped rendering of arbitrary values for log lines."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def block(header: str, body: object, *, level: str = "INFO", limit: int = MAX_BLOCK) -> None:
    """Print `header` then `body` in full, wrapped and indented beneath it.

    Used for agent-authored text. The body's own newlines are preserved (they carry the
    structure of a markdown answer); only over-long single lines are wrapped. Emitted under
    one lock acquisition so a multi-line agent message never interleaves with another
    branch's output.
    """
    if level == "DEBUG" and not DEBUG_ENABLED:
        return

    text = str(body).strip()
    if not text:
        return
    if len(text) > limit:
        text = f"{text[:limit]}\n... [truncated, {len(text) - limit:,} more characters]"

    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if not stripped:
            lines.append("")
        elif len(stripped) <= WRAP_WIDTH:
            lines.append(stripped)
        else:
            lines.extend(
                textwrap.wrap(
                    stripped,
                    width=WRAP_WIDTH,
                    # Keep markdown bullets and numbering legible under wrapping.
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

    with _print_lock:
        print(f"[{level}] {header}", file=sys.stdout)
        for line in lines:
            print(f"{BODY_INDENT}{line}", file=sys.stdout)
        sys.stdout.flush()
