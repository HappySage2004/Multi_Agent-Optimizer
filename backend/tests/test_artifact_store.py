"""Artifact resolution — the portable handle is `artifact_id`, not `path`.

`localDB/runs.json` is committed to git while `backend/artifacts/` is gitignored, so a run
record routinely arrives in a checkout that never wrote its parquet. Worse, the recorded
`path` is the absolute path of whichever machine *did* write it — a real run in this repo
carries `/Users/aaryanawadh/projects/...` — so trusting it makes an artifact unreadable
even when the file is present locally under the same id.

That was a live bug: a whole session's inspector rendered empty tabs while the run record
cheerfully reported 250 rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import get_settings
from app.models.artifacts import ArtifactReference
from app.services import artifact_store


def _ref(artifact_id: str, path: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        kind="screen_candidates",
        path=path,
        rows=2,
        columns=["screen_id"],
        summary={},
        provenance="computed",
    )


@pytest.fixture
def written() -> ArtifactReference:
    return artifact_store.write_records(
        "screen_candidates",
        [{"screen_id": "SCR-1"}, {"screen_id": "SCR-2"}],
    )


def test_a_freshly_written_artifact_reads_back(written: ArtifactReference) -> None:
    assert [r["screen_id"] for r in artifact_store.read_rows(written)] == ["SCR-1", "SCR-2"]


def test_a_foreign_absolute_path_still_resolves(written: ArtifactReference) -> None:
    """The exact shape of the bug: a run record cloned from another machine."""
    foreign = _ref(
        written.artifact_id,
        f"/Users/someone/projects/Multi_Agent-Optimizer/backend/artifacts/{written.artifact_id}.parquet",
    )
    assert [r["screen_id"] for r in artifact_store.read_rows(foreign)] == ["SCR-1", "SCR-2"]


def test_the_local_store_wins_over_the_recorded_path(written: ArtifactReference) -> None:
    """A stale path pointing at a *different* real file must not be preferred."""
    decoy = get_settings().artifacts_dir / "decoy.parquet"
    pd.DataFrame([{"screen_id": "WRONG"}]).to_parquet(decoy, index=False)

    ref = _ref(written.artifact_id, str(decoy))
    assert [r["screen_id"] for r in artifact_store.read_rows(ref)] == ["SCR-1", "SCR-2"]


def test_a_genuinely_missing_artifact_still_raises() -> None:
    """Resolution must not paper over an artifact that was never written."""
    ref = _ref("screen_candidates-deadbeef", "/nowhere/screen_candidates-deadbeef.parquet")
    with pytest.raises(FileNotFoundError) as exc:
        artifact_store.read_rows(ref)
    # The message names both places it looked, since that is what a developer needs.
    assert "screen_candidates-deadbeef" in str(exc.value)
    assert "nowhere" in str(exc.value)


def test_the_bare_id_form_still_works(written: ArtifactReference) -> None:
    assert len(artifact_store.read_rows(written.artifact_id)) == 2


def test_resolve_path_points_into_the_configured_store(written: ArtifactReference) -> None:
    resolved = artifact_store.resolve_path(written)
    assert resolved.parent == get_settings().artifacts_dir
    assert resolved.name == f"{written.artifact_id}.parquet"
