"""Artifact references — how bulky data moves between agents without entering LLM context."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Provenance = Literal["stub", "computed"]
"""`stub` = placeholder output from an unimplemented specialist. `computed` = real analysis.

Every artifact and specialist result carries this. The Master Agent propagates it into the
final recommendation so a stubbed pipeline can never be mistaken for a real one.
"""


class ArtifactReference(BaseModel):
    """A pointer to a durable artifact on disk plus enough metadata to reason about it."""

    artifact_id: str
    kind: str = Field(description="Logical type, e.g. 'screen_candidates', 'screen_economics'")
    path: str
    rows: int
    columns: list[str] = []
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Aggregates only — never row-level data"
    )
    provenance: Provenance = "computed"
    created_at: datetime = Field(default_factory=datetime.now)

    def as_context(self) -> str:
        """Compact, LLM-safe rendering. Use this in prompts instead of the artifact contents."""
        parts = [
            f"artifact_id={self.artifact_id}",
            f"kind={self.kind}",
            f"rows={self.rows}",
            f"provenance={self.provenance}",
        ]
        if self.columns:
            parts.append(f"columns={','.join(self.columns)}")
        if self.summary:
            parts.append(f"summary={self.summary}")
        return " | ".join(parts)
