"""Pre-flight clarifying questions — the contract between the Master Agent and the UI.

The Master Agent may ask the rep at exactly one point: after geography resolves and before
`create_campaign_spec`. This module is the shape of that ask.

WHY THE AGENT DOES NOT WRITE THE OPTIONS ITSELF. It supplies only the two probable answers
and which one it would take; `build_question` assembles the four options the rep sees. That
split is deliberate and follows SOLUTION.md section 31.2 — the LLM reasons about what the
likely answers are, the code guarantees the shape. An agent free-handing its own option set
drifts: three options one turn, five the next, a "decide for yourself" that forgets to say
what it would decide. The UI renders whatever it is given, so the shape has to be
guaranteed somewhere, and code is the only place it can be.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

OptionKind = Literal["answer", "defer", "custom"]
"""`answer` = a concrete choice the agent proposed. `defer` = the agent decides. `custom` =
the rep types their own, so the UI must show a text field for this one and only this one."""

DEFER_KEY = "C"
CUSTOM_KEY = "D"

# Which campaign inputs a question is allowed to be about. Anything outside this set either
# has a defensible default (so asking is friction) or is a lookup rather than a question.
# Kept as a closed set so a prompt change cannot quietly widen the gate.
ASKABLE_FIELDS: tuple[str, ...] = (
    "audience_terms",
    "industry_vertical",
    "budget",
    "duration_days",
    "start_date",
    "geography",
    "screen_count_vs_budget",
)


class ClarifyingOption(BaseModel):
    key: str = Field(description="A, B, C or D — the label the UI shows and the rep replies with")
    label: str = Field(description="Short, selectable text")
    detail: str | None = Field(
        default=None, description="What choosing this changes. Rendered under the label."
    )
    kind: OptionKind = "answer"
    value: str | None = Field(
        default=None,
        description=(
            "The machine-readable answer for an `answer` option — an AUDIENCE_TERMS value, "
            "an industry, a number. Null for `defer` and `custom`."
        ),
    )


class ClarifyingQuestion(BaseModel):
    id: str
    field: str = Field(description="Which campaign input this question fills. In ASKABLE_FIELDS.")
    question: str
    options: list[ClarifyingOption]
    recommended_key: str = Field(
        description="A or B — what `Decide for yourself` resolves to. Always populated."
    )


class ClarificationRequest(BaseModel):
    """One round of questions. There is never a second round for the same brief."""

    session_id: str
    understood: str = Field(
        description="What the agent already took from the brief, so the gaps read as narrow."
    )
    questions: list[ClarifyingQuestion]
    asked_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    answered: bool = False

    @property
    def open(self) -> bool:
        return bool(self.questions) and not self.answered


def build_question(
    *,
    index: int,
    field: str,
    question: str,
    option_a: str,
    option_b: str,
    recommended: str,
    recommendation_reason: str,
    option_a_detail: str | None = None,
    option_b_detail: str | None = None,
    option_a_value: str | None = None,
    option_b_value: str | None = None,
) -> ClarifyingQuestion:
    """Assemble the canonical four options from the agent's two candidates.

    `recommended` must be "A" or "B"; it is what `Decide for yourself` commits to, and it is
    quoted in that option's own text so choosing C is an informed decision rather than a
    shrug. A defer option that does not say what it defers to is just a silent default with
    an extra click.
    """
    key = recommended.strip().upper()
    if key not in ("A", "B"):
        raise ValueError(f"recommended must be 'A' or 'B', got {recommended!r}")

    chosen_label = option_a if key == "A" else option_b
    return ClarifyingQuestion(
        id=f"q{index}",
        field=field,
        question=question.strip(),
        recommended_key=key,
        options=[
            ClarifyingOption(
                key="A",
                label=option_a.strip(),
                detail=(option_a_detail or "").strip() or None,
                kind="answer",
                value=option_a_value or option_a.strip(),
            ),
            ClarifyingOption(
                key="B",
                label=option_b.strip(),
                detail=(option_b_detail or "").strip() or None,
                kind="answer",
                value=option_b_value or option_b.strip(),
            ),
            ClarifyingOption(
                key=DEFER_KEY,
                label="Decide for yourself",
                detail=f"I'd take {key} — {chosen_label}. {recommendation_reason.strip()}",
                kind="defer",
            ),
            ClarifyingOption(
                key=CUSTOM_KEY,
                label="Something else",
                detail="Type it and I'll use that instead.",
                kind="custom",
            ),
        ],
    )
