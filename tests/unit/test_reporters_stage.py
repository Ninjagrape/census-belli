"""The report stage's file handling: stale manifests and atomic writes. No database."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from pipeline.reporters.atomic import write_text_atomic
from pipeline.stages import report as report_stage
from pipeline.stages.base import StageContext


def _spec_params(tmp_path: Path) -> dict[str, Any]:
    return {
        "output_root": str(tmp_path / "report"),
        "diagnostics_root": str(tmp_path / "diag"),
        "evaluation_root": str(tmp_path / "eval"),
        "report_path": str(tmp_path / "report" / "manifest.json"),
    }


def _old_outputs(tmp_path: Path) -> Path:
    out = tmp_path / "report"
    out.mkdir()
    (out / "manifest.json").write_text('{"model_run_id": 1}', encoding="utf-8")
    (out / "rankings.csv").write_text("old", encoding="utf-8")
    return out


def test_no_evaluable_run_deletes_a_stale_manifest_and_keeps_other_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _old_outputs(tmp_path)
    monkeypatch.setattr(report_stage.load, "select_run", lambda conn: None)

    report_stage._run_body(None, _spec_params(tmp_path), StageContext())  # type: ignore[arg-type]

    assert not (out / "manifest.json").exists()
    assert (out / "rankings.csv").read_text(encoding="utf-8") == "old"


def test_writing_starts_by_deleting_the_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _old_outputs(tmp_path)
    monkeypatch.setattr(report_stage.load, "select_run", lambda conn: 7)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("build failed")

    monkeypatch.setattr(report_stage, "_build", boom)

    with pytest.raises(RuntimeError):
        report_stage._run_body(None, _spec_params(tmp_path), StageContext())  # type: ignore[arg-type]

    assert not (out / "manifest.json").exists()


def test_a_dry_run_leaves_the_manifest_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _old_outputs(tmp_path)
    monkeypatch.setattr(report_stage.load, "select_run", lambda conn: None)

    report_stage._run_body(None, _spec_params(tmp_path), StageContext(dry_run=True))  # type: ignore[arg-type]

    assert (out / "manifest.json").exists()


def test_an_atomic_write_replaces_the_file_whole(tmp_path: Path) -> None:
    target = tmp_path / "a" / "x.txt"

    write_text_atomic(target, "one")
    write_text_atomic(target, "two")

    assert target.read_text(encoding="utf-8") == "two"
    assert [p.name for p in target.parent.iterdir()] == ["x.txt"]


def test_a_failed_atomic_write_keeps_the_old_file_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "x.txt"
    target.write_text("old", encoding="utf-8")

    def refuse(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)

    with pytest.raises(OSError):
        write_text_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]
