"""Test isolation: redirect localDB and the artifact store into a temp directory.

Without this, running the suite writes campaign runs and parquet artifacts into the real
localDB/ and backend/artifacts/.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_storage():
    tmp = Path(tempfile.mkdtemp(prefix="tmcrs-test-"))
    os.environ["LOCAL_DB_DIR"] = str(tmp / "localDB")
    os.environ["ARTIFACTS_DIR"] = str(tmp / "artifacts")
    os.environ["STAGE_DIR"] = str(tmp / "stage")

    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.local_db_dir.parent == tmp, "settings did not pick up the temp dirs"

    yield tmp

    get_settings.cache_clear()
    for key in ("LOCAL_DB_DIR", "ARTIFACTS_DIR", "STAGE_DIR"):
        os.environ.pop(key, None)
