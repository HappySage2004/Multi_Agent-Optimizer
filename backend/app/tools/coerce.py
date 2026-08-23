"""Coercion of LLM-supplied tool arguments into the shapes the pipeline requires.

Every tool on every agent is called by a language model, and a model emits JSON that is
*plausible* rather than *typed*. Three failures showed up repeatedly, and each one was
silent rather than loud:

    hard_constraints={"allowed_screen_types": "metro_station"}
        `list("metro_station")` is thirteen single characters, so `.isin()` matched
        nothing, the candidate pool filtered to zero, and stage 2 reported "all screens
        were removed by the spec's hard constraints" -- a true sentence about a constraint
        the brief never stated. The agent's next move is to relax a constraint that was
        never the problem.

    hard_constraints='{"min_screens": 20}'
        A JSON *string* where a dict was declared. Where it survives into the spec, every
        consumer's `.get()` raises AttributeError three frames down from the tool call.

    audience_terms="young_professionals, commuters"
        One string that reads correctly to a human and fails the closed-vocabulary
        validator as a single unknown term.

So the tool boundary coerces BEFORE it validates. The rules here are deliberately narrow:
this module reshapes what a value obviously IS and never guesses at what it might have
meant. A comma-separated string is a list of its parts. A JSON object string is a dict.
An unknown screen type is an error, not a near-miss -- guessing at a vocabulary is what
`INDUSTRY_VERTICALS` already proved silently deletes a quarter of the relevance model.

Every failure raises `ArgumentError`, which the calling tool turns into a
`status: "invalid"` result naming the field and the accepted shape. Raising out of a tool
aborts the SSE stream and costs the rep the turn; a described error is one the agent can
act on by itself.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from app.models.campaign import (
    DISPLAY_TYPE_NON_CONSTRAINTS,
    HARD_CONSTRAINT_SHAPES,
    SCREEN_TYPES,
    TIME_BLOCK_IDS,
)

# Separators a model uses when it writes a list as prose. `|` is included because it is
# what appears when the model has been thinking in markdown tables.
_LIST_SEPARATORS = (",", ";", "|", "\n")

# The several ways a model writes "nothing" when it feels obliged to fill a field.
_EMPTY_TOKENS = frozenset({"", "none", "null", "n/a", "na", "nil", "-", "[]", "{}", "unknown"})


class ArgumentError(ValueError):
    """A tool argument that cannot be reshaped into what the pipeline needs.

    The message is written for the model that made the call: what was passed, what shape
    is accepted, and an example of it.
    """


def _norm_token(value: str) -> str:
    """Vocabulary-comparison form. 'Metro Station' and 'METRO-STATION' are one value."""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _scalar_to_str(value: Any) -> str:
    """One non-container value as the string the vocabularies are written in.

    Floats collapse to their integer form when they have no fractional part, because a
    model asked for time blocks writes `[2.0, 3.0]` about as often as `["2", "3"]`.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _parse_embedded(text: str) -> Any:
    """A JSON or Python-literal container written as a string, or None if it is neither.

    `ast.literal_eval` is the second attempt on purpose: a model that has been thinking in
    Python emits `"{'min_screens': 20}"`, which is not JSON. `literal_eval` evaluates
    literals only -- no names, no calls -- so this parses data without executing anything.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(stripped)
        except (ValueError, SyntaxError, TypeError):
            continue
    raise ArgumentError(
        f"{stripped[:80]!r} opens like a list or an object but is not valid JSON. Pass a "
        f"real JSON value, not a string containing one."
    )


def as_str_list(
    value: Any,
    *,
    field: str,
    vocabulary: tuple[str, ...] | None = None,
    example: str = '["a", "b"]',
) -> list[str]:
    """Any plausible spelling of "a list of strings", as a list of strings.

    Accepts a list, a tuple, a single scalar, a JSON array written as a string, and a
    separated string. A single bare value becomes a one-item list -- that is the fix for
    `allowed_screen_types="metro_station"`, which used to become thirteen characters.

    `vocabulary` closes the list: values are normalized for case and separators first (so
    'Metro Station' resolves), and anything still unmatched raises rather than passing
    through to be scored as a neutral default somewhere downstream.
    """
    items: list[str]

    if value is None:
        items = []
    elif isinstance(value, dict):
        raise ArgumentError(
            f"{field} was passed an object, but it takes a list of strings. Pass the "
            f"values themselves, e.g. {field}={example} -- not a wrapper object."
        )
    elif isinstance(value, (list, tuple, set)):
        items = []
        for entry in value:
            if isinstance(entry, (dict, list, tuple, set)):
                raise ArgumentError(
                    f"{field} contains a nested {type(entry).__name__}, but every entry "
                    f"must be a plain string. Pass {field}={example}."
                )
            items.append(_scalar_to_str(entry))
    elif isinstance(value, str):
        if _norm_token(value) in _EMPTY_TOKENS:
            items = []
        elif (parsed := _parse_embedded(value)) is not None:
            return as_str_list(parsed, field=field, vocabulary=vocabulary, example=example)
        else:
            separator = next((s for s in _LIST_SEPARATORS if s in value), None)
            items = (
                [part for part in (p.strip() for p in value.split(separator)) if part]
                if separator
                else [value.strip()]
            )
    else:
        items = [_scalar_to_str(value)]

    items = [i for i in items if i and _norm_token(i) not in _EMPTY_TOKENS]

    if vocabulary is None:
        # Ids and free text keep their exact spelling -- only the closed vocabularies
        # normalize, because only they have a canonical form to normalize onto.
        return list(dict.fromkeys(items))

    allowed = {_norm_token(v): v for v in vocabulary}
    resolved: list[str] = []
    unknown: list[str] = []
    for item in items:
        canonical = allowed.get(_norm_token(item))
        if canonical is None:
            unknown.append(item)
        else:
            resolved.append(canonical)
    if unknown:
        raise ArgumentError(
            f"{field} contains {unknown}, which the pipeline does not accept. Allowed "
            f"values are {list(vocabulary)}. Pick from that list or omit the field -- an "
            f"unrecognized value does not fail loudly downstream, it neutralizes the part "
            f"of the model that reads it."
        )
    return list(dict.fromkeys(resolved))


def as_dict(value: Any, *, field: str, example: str = '{"key": value}') -> dict[str, Any]:
    """Any plausible spelling of "an object", as a dict.

    Accepts a dict, a JSON object written as a string, a Python-repr object written as a
    string, and the several ways a model writes "nothing".
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        if _norm_token(value) in _EMPTY_TOKENS:
            return {}
        parsed = _parse_embedded(value)
        if isinstance(parsed, dict):
            return parsed
        raise ArgumentError(
            f"{field} was passed the string {value[:80]!r}, but it takes an object. Pass "
            f"{field}={example}."
        )
    if isinstance(value, (list, tuple)):
        raise ArgumentError(
            f"{field} was passed a list, but it takes an object keyed by name. Pass "
            f"{field}={example}."
        )
    raise ArgumentError(
        f"{field} was passed a {type(value).__name__}, but it takes an object. Pass "
        f"{field}={example}."
    )


def as_dict_list(value: Any, *, field: str, example: str = '[{"key": value}]') -> list[dict]:
    """A list of objects, tolerating a single object and a JSON array written as a string.

    A model asked for one question inside a list of questions frequently sends the
    question by itself.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, str):
        if _norm_token(value) in _EMPTY_TOKENS:
            return []
        parsed = _parse_embedded(value)
        if parsed is None:
            raise ArgumentError(
                f"{field} was passed the string {value[:80]!r}, but it takes a list of "
                f"objects. Pass {field}={example}."
            )
        return as_dict_list(parsed, field=field, example=example)
    if isinstance(value, (list, tuple)):
        out: list[dict] = []
        for index, entry in enumerate(value, start=1):
            if isinstance(entry, str) and (parsed := _parse_embedded(entry)) is not None:
                entry = parsed
            if not isinstance(entry, dict):
                raise ArgumentError(
                    f"{field} entry {index} is a {type(entry).__name__}, but every entry "
                    f"must be an object. Pass {field}={example}."
                )
            out.append(dict(entry))
        return out
    raise ArgumentError(
        f"{field} was passed a {type(value).__name__}, but it takes a list of objects. "
        f"Pass {field}={example}."
    )


def as_int(value: Any, *, field: str, minimum: int | None = None) -> int | None:
    """A whole number, from an int or from the string a model writes one as.

    `bool` and a fractional float are refused rather than coerced: `int(True)` is 1 and
    `int(1.5)` is 1, so coercion turns a nonsense value into a plausible one and then
    enforces it as if it had been asked for. Same rule `contract.resolve_slot_cap` follows
    for `max_slots_per_day`, and for the same reason.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ArgumentError(f"{field}={value!r} is a flag, not a whole number.")
    if isinstance(value, float):
        if not value.is_integer():
            raise ArgumentError(f"{field}={value!r} is not a whole number of units.")
        number = int(value)
    elif isinstance(value, int):
        number = value
    elif isinstance(value, str):
        if _norm_token(value) in _EMPTY_TOKENS:
            return None
        # Thousands separators and a currency symbol are how a model writes back a number
        # it read in prose.
        cleaned = value.strip().replace(",", "").replace("_", "").lstrip("$EUR").strip()
        try:
            parsed = float(cleaned)
        except ValueError:
            raise ArgumentError(
                f"{field}={value!r} is not a whole number. Pass a number, not prose."
            ) from None
        if not parsed.is_integer():
            raise ArgumentError(f"{field}={value!r} is not a whole number of units.")
        number = int(parsed)
    else:
        raise ArgumentError(f"{field}={value!r} is not a whole number.")

    if minimum is not None and number < minimum:
        raise ArgumentError(f"{field}={number} is below the minimum of {minimum}.")
    return number


def as_fraction(value: Any, *, field: str) -> float | None:
    """A 0-1 fraction, accepting the percentage a model writes instead.

    A fraction above 1.0 is unsatisfiable by definition, so `90` and `"90%"` are
    unambiguously 0.90 rather than a request that can never be met. That is the one place
    this module infers intent, and it does so only because the alternative reading has no
    valid meaning: `min_budget_utilization=90` used to reach the solver as 9000% and come
    back as an infeasibility report quoting that figure at the rep.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ArgumentError(f"{field}={value!r} is a flag, not a fraction.")
    if isinstance(value, str):
        if _norm_token(value) in _EMPTY_TOKENS:
            return None
        text = value.strip().replace(",", "").rstrip("%").strip()
        try:
            number = float(text)
        except ValueError:
            raise ArgumentError(
                f"{field}={value!r} is not a number. Pass a 0-1 fraction, e.g. 0.9 for 90%."
            ) from None
        if value.strip().endswith("%"):
            number /= 100.0
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ArgumentError(f"{field}={value!r} is not a number.")

    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise ArgumentError(
            f"{field}={value!r} is not a usable share. Pass a 0-1 fraction, e.g. 0.9 for 90%."
        )
    return number


def clamp_int(value: Any, *, field: str, low: int, high: int) -> tuple[int | None, str | None]:
    """Bound a whole number to a range, REPORTING rather than rejecting.

    Same trade-off `PricingLevers.clamp` makes: in an agent loop a rejected call becomes a
    retry against a per-minute rate limit to arrive at the number clamping returns
    directly. Returns `(value, note)`, where `note` is None when nothing moved.
    """
    number = as_int(value, field=field)
    if number is None:
        return None, None
    bounded = min(max(number, low), high)
    if bounded != number:
        return bounded, f"{field} {number} clamped to {bounded} (allowed {low}-{high})"
    return bounded, None


# --------------------------------------------------------------- hard constraints


def normalize_hard_constraints(value: Any) -> tuple[dict[str, Any], list[str]]:
    """The brief's `hard_constraints` in the exact types every consumer expects.

    Returns the normalized dict plus notes describing anything that was reshaped, so the
    tool result can tell the agent what was read rather than leaving it to assume.

    Unknown keys raise HERE, at intake, rather than at verification. `hard_constraints` is
    a free-form dict and that freedom already cost a shipped package: a key no stage reads
    is persisted, echoed back in `normalized_spec` and in `campaign_inputs`, and believed.
    `validation._recognized_constraint_check` stays as the backstop for specs written
    before this gate existed, but the cheapest place to reject a key is the call that
    invents it.
    """
    raw = as_dict(
        value,
        field="hard_constraints",
        example='{"max_slots_per_day": 1, "allowed_screen_types": ["metro_station"]}',
    )
    if not raw:
        return {}, []

    notes: list[str] = []

    # Every screen in this network is digital, so a display-type key selects nothing. Drop
    # it and say so: raising would fail a run over a phrase that is already true, and
    # keeping it failed verification instead — the plan came back "blocked by the
    # digital-only constraint", which was never a constraint at all.
    inert = [k for k in raw if k in DISPLAY_TYPE_NON_CONSTRAINTS]
    if inert:
        raw = {k: v for k, v in raw.items() if k not in DISPLAY_TYPE_NON_CONSTRAINTS}
        notes.append(
            f"dropped {inert}: every screen in this network is digital, so there is nothing "
            f"for a display-type constraint to select. Nothing was filtered out."
        )

    unknown = [k for k in raw if k not in HARD_CONSTRAINT_SHAPES]
    if unknown:
        raise ArgumentError(
            f"hard_constraints contains {unknown}, which no stage enforces. Only "
            f"{sorted(HARD_CONSTRAINT_SHAPES)} are read. A key outside that list is "
            f"persisted, echoed back to you and honoured by nobody, so record the brief's "
            f"constraint under the right key or put it in missing_information instead."
        )

    out: dict[str, Any] = {}
    for key, given in raw.items():
        shape = HARD_CONSTRAINT_SHAPES[key]
        if shape == "int":
            coerced: Any = as_int(given, field=f"hard_constraints.{key}", minimum=1)
        elif shape == "int_passthrough":
            # Normalize "1" to 1 and leave everything else exactly as passed, so the
            # constraint's own enforcement site is the one that rejects it, with the message
            # it has written for that job.
            try:
                coerced = as_int(given, field=f"hard_constraints.{key}")
            except ArgumentError:
                coerced = given
        elif shape == "fraction":
            coerced = as_fraction(given, field=f"hard_constraints.{key}")
        elif shape == "screen_types":
            coerced = as_str_list(
                given,
                field=f"hard_constraints.{key}",
                vocabulary=SCREEN_TYPES,
                example='["metro_station", "bus_stop"]',
            )
        elif shape == "time_blocks":
            coerced = as_str_list(
                given,
                field=f"hard_constraints.{key}",
                vocabulary=TIME_BLOCK_IDS,
                example='["2", "5"]',
            )
        else:
            coerced = as_str_list(given, field=f"hard_constraints.{key}", example='["LH-ZONE-001"]')

        if coerced is None or (isinstance(coerced, list) and not coerced):
            notes.append(f"{key} was empty, so it has been dropped rather than enforced")
            continue
        if coerced != given:
            notes.append(f"{key} read as {coerced!r} (you passed {given!r})")
        out[key] = coerced

    return out, notes
