"""Durable artifact storage.

Specialists write bulky results here and hand the Master Agent an ArtifactReference.
Rule: row-level data never crosses an agent boundary — only reference + schema + summary.

Records are stored with their nested structure intact (parquet struct/list columns), so a
contract round-trips exactly. Summaries are computed from a flattened copy purely for
reporting.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.config import get_settings
from app.models.artifacts import ArtifactReference, Provenance

T = TypeVar("T", bound=BaseModel)


def _new_id(kind: str) -> str:
    return f"{kind}-{uuid.uuid4().hex[:12]}"


def write_records(
    kind: str,
    records: Sequence[BaseModel] | Sequence[dict[str, Any]],
    *,
    summary: dict[str, Any] | None = None,
    provenance: Provenance = "computed",
) -> ArtifactReference:
    """Persist Pydantic models (or dicts) as parquet and return a reference to them."""
    rows = [r.model_dump(mode="json") if isinstance(r, BaseModel) else r for r in records]
    df = pd.DataFrame(rows)

    settings = get_settings()
    artifact_id = _new_id(kind)
    path = settings.artifacts_dir / f"{artifact_id}.parquet"
    df.to_parquet(path, index=False)

    return ArtifactReference(
        artifact_id=artifact_id,
        kind=kind,
        path=str(path),
        rows=len(df),
        columns=list(df.columns),
        summary=summary or _auto_summary(rows),
        provenance=provenance,
    )


def _auto_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Numeric min/mean/max over a flattened view — cheap, and safe to put in a prompt."""
    if not rows:
        return {}
    flat = pd.json_normalize(rows, sep=".")
    out: dict[str, Any] = {}
    for col in flat.select_dtypes("number").columns[:12]:
        series = flat[col].dropna()
        if series.empty:
            continue
        out[col] = {
            "min": round(float(series.min()), 4),
            "mean": round(float(series.mean()), 4),
            "max": round(float(series.max()), 4),
        }
    return out


def resolve_path(ref: ArtifactReference | str) -> Path:
    """Where this artifact actually lives, or raise FileNotFoundError.

    `artifact_id` is the portable handle; `ArtifactReference.path` is not. It records the
    absolute path of whichever machine wrote the artifact, and localDB/runs.json is
    committed to git while `backend/artifacts/` is ignored — so a run record cloned from
    another checkout arrives pointing at a path like
    `/Users/someone/projects/.../screen_candidates-abc123.parquet`, which will never exist
    here even when the file has been regenerated locally under the same id.

    So resolve against the configured `artifacts_dir` first and treat the recorded path as
    a fallback, not as the truth. The reverse order was a real bug: a whole session's
    inspector rendered empty because the reference was trusted over the filesystem.
    """
    settings = get_settings()
    artifact_id = ref.artifact_id if isinstance(ref, ArtifactReference) else ref

    candidates = [settings.artifacts_dir / f"{artifact_id}.parquet"]
    if isinstance(ref, ArtifactReference) and ref.path:
        candidates.append(Path(ref.path))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No artifact '{artifact_id}' — looked in {[str(c) for c in candidates]}"
    )


def read_artifact(ref: ArtifactReference | str) -> pd.DataFrame:
    """Load an artifact back as a DataFrame. Tools use this; agents never see the rows."""
    return pd.read_parquet(resolve_path(ref))


def read_rows(
    ref: ArtifactReference | str,
    *,
    limit: int | None = None,
    screen_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """JSON-safe rows for the HTTP read path that backs the inspector panel.

    Row-level data still never crosses an agent boundary — this exists for the UI only.
    Ordering is the artifact's own, which is already ranked for `screen_candidates`.

    `screen_ids` filters before `limit`, which is what lets the UI pull the rows for the
    screens actually in a package: those sit anywhere in the artifact, so a plain top-N
    slice would mostly miss them.
    """
    df = read_artifact(ref)
    if screen_ids is not None and "screen_id" in df.columns:
        df = df[df["screen_id"].isin(list(screen_ids))]
    if limit is not None:
        df = df.head(limit)
    return [_pythonize(rec) for rec in df.to_dict(orient="records")]


def read_models(ref: ArtifactReference | str, model: type[T]) -> list[T]:
    """Rehydrate an artifact into its Pydantic contract type."""
    df = read_artifact(ref)
    return [model.model_validate(_pythonize(rec)) for rec in df.to_dict(orient="records")]


def _pythonize(value: Any) -> Any:
    """Convert pyarrow/numpy containers back to plain Python for Pydantic validation."""
    if isinstance(value, dict):
        return {k: _pythonize(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_pythonize(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_pythonize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value
