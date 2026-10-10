"""Suite-wide fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_request_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point offline batch files at a temp directory.

    Offline mode is the default for every LLM stage, so any test that runs a
    stage with an uncached request would otherwise drop a batch file into the
    real ``data/llm_requests/``, where ``/process-llm-batch`` would pick it up.
    """
    directory = tmp_path_factory.mktemp("llm_requests")
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", directory)
    return directory
