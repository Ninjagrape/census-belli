"""Unit tests for the inspector's file readers, on tmp_path fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.inspector import files
from pipeline.inspector.connection import redact_url


def _write_round(root: Path, scale: str, n: int, imputed: bool) -> None:
    directory = root / "imputed" / scale / f"round_{n}"
    directory.mkdir(parents=True)
    (directory / "sides.csv").write_text(
        "side_id,battle_id,log_troops,is_imputed,scale\n"
        f"1,1,9.0,False,{scale}\n2,1,{9.5 + n},{imputed},{scale}\n",
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(json.dumps({"round": n}), encoding="utf-8")


def test_absent_data_root_returns_empty_everywhere(tmp_path: Path) -> None:
    missing = tmp_path / "nope"

    assert files.raw_file_counts(missing).empty
    frame, bad = files.read_resolution_log(missing)
    assert frame.empty and bad == 0
    assert files.list_imputed_scales(missing) == []
    assert files.read_rounds(missing, "neutral").empty
    assert files.read_manifests(missing, "neutral").empty
    assert files.read_gate_report(missing, "neutral") == {}
    assert files.find_neutral_draws(missing) == []
    assert files.summarise_neutral_draws(missing / "x.npz").empty


def test_raw_file_counts_counts_each_subdir_and_root(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "wikidata").mkdir(parents=True)
    (raw / "wikidata" / "a.json").write_text("{}", encoding="utf-8")
    (raw / "wikidata" / "b.json").write_text("{}", encoding="utf-8")
    (raw / "crawl_state.json").write_text("{}", encoding="utf-8")

    counts = files.raw_file_counts(tmp_path).set_index("subdir")["n_files"].to_dict()

    assert counts == {"(root)": 1, "wikidata": 2}


def test_resolution_log_skips_malformed_lines_and_counts_them(tmp_path: Path) -> None:
    path = tmp_path / "processed" / "resolution_log.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"group_key": "a", "years": [1, 2]}\n{broken\n\n[1, 2]\n{"group_key": "b"}\n',
        encoding="utf-8",
    )

    frame, bad = files.read_resolution_log(tmp_path)

    assert list(frame["group_key"]) == ["a", "b"]
    assert bad == 2
    assert frame.loc[0, "years"] == "[1, 2]"


def test_rounds_are_stacked_with_a_round_column(tmp_path: Path) -> None:
    _write_round(tmp_path, "neutral", 0, True)
    _write_round(tmp_path, "neutral", 1, True)
    _write_round(tmp_path, "published", 0, False)
    (tmp_path / "imputed" / "neutral" / "round_x").mkdir()

    rounds = files.read_rounds(tmp_path, "neutral")

    assert files.list_imputed_scales(tmp_path) == ["neutral", "published"]
    assert sorted(rounds["round"].unique()) == [0, 1]
    assert len(rounds) == 4
    assert list(files.read_manifests(tmp_path, "neutral")["round"]) == [0, 1]


def test_unreadable_round_csv_is_skipped(tmp_path: Path) -> None:
    _write_round(tmp_path, "neutral", 0, True)
    bad = tmp_path / "imputed" / "neutral" / "round_1"
    bad.mkdir()
    (bad / "sides.csv").write_bytes(b"")

    rounds = files.read_rounds(tmp_path, "neutral")

    assert set(rounds["round"]) == {0}


def test_gate_report_round_trips_and_tolerates_garbage(tmp_path: Path) -> None:
    scale_dir = tmp_path / "imputed" / "neutral"
    scale_dir.mkdir(parents=True)
    (scale_dir / "gate_report.json").write_text(
        json.dumps({"run_id": 3, "gates": []}), encoding="utf-8"
    )
    assert files.read_gate_report(tmp_path, "neutral")["run_id"] == 3

    (scale_dir / "gate_report.json").write_text("{nope", encoding="utf-8")
    assert files.read_gate_report(tmp_path, "neutral") == {}


def test_neutral_draws_summary_has_per_side_moments(tmp_path: Path) -> None:
    directory = tmp_path / "processed" / "reconcile"
    directory.mkdir(parents=True)
    draws = np.tile(np.array([[1.0, 10.0]]), (100, 1)) + np.linspace(-1, 1, 100)[:, None]
    np.savez(
        directory / "neutral_draws_run7.npz",
        run_id=np.int64(7),
        side_id=np.array([11, 12]),
        battle_id=np.array([1, 1]),
        draws=draws,
        m0_mean=np.float64(0.0),
        s0_mean=np.float64(1.0),
    )

    (path,) = files.find_neutral_draws(tmp_path)
    summary = files.summarise_neutral_draws(path)

    assert list(summary["side_id"]) == [11, 12]
    assert np.allclose(summary["mean"], [1.0, 10.0])
    assert (summary["q025"] < summary["q50"]).all() and (summary["q50"] < summary["q975"]).all()


def test_malformed_npz_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "neutral_draws_run1.npz"
    path.write_bytes(b"not an npz")
    assert files.summarise_neutral_draws(path).empty


def test_redact_url_hides_the_password() -> None:
    redacted = redact_url("postgresql+psycopg://u:secret@host:5432/db")
    assert "secret" not in redacted
    assert "host:5432/db" in redacted
