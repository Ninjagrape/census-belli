"""File-side readers for the inspector.

Every reader takes a ``data_root`` (normally ``data``) and degrades to an empty
result when the file or directory is absent or unreadable: a half-run pipeline
is the normal case for a dashboard, not an error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from pipeline.inspector.frames import Frame, empty_frame, pd

logger = structlog.get_logger()

__all__ = [
    "find_neutral_draws",
    "list_imputed_scales",
    "raw_file_counts",
    "read_gate_report",
    "read_manifests",
    "read_resolution_log",
    "read_rounds",
    "summarise_neutral_draws",
]

_ROUND_DIR: Final = re.compile(r"round_(\d+)")
_NEUTRAL_FILE: Final = re.compile(r"neutral_draws_run(\d+)\.npz")
_QUANTILES: Final = (0.025, 0.5, 0.975)
_ROUND_COLUMNS: Final = ["side_id", "battle_id", "log_troops", "is_imputed", "scale", "round"]


def raw_file_counts(data_root: Path) -> Frame:
    """Count files and bytes under each ``data/raw`` subdirectory.

    Args:
        data_root: The data directory.

    Returns:
        Columns ``subdir``, ``n_files``, ``size_mb``; files directly under
        ``raw`` are reported as ``(root)``.
    """
    raw = data_root / "raw"
    columns = ["subdir", "n_files", "size_mb"]
    if not raw.is_dir():
        return empty_frame(columns)

    rows: list[dict[str, Any]] = []
    root_files = [p for p in raw.iterdir() if p.is_file()]
    if root_files:
        rows.append(_count_row("(root)", root_files))
    for child in sorted(p for p in raw.iterdir() if p.is_dir()):
        rows.append(_count_row(child.name, [p for p in child.rglob("*") if p.is_file()]))
    return pd.DataFrame(rows, columns=columns) if rows else empty_frame(columns)


def _count_row(name: str, files: list[Path]) -> dict[str, Any]:
    """Summarise a list of files as one row."""
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return {"subdir": name, "n_files": len(files), "size_mb": round(total / 1e6, 3)}


def read_resolution_log(data_root: Path) -> tuple[Frame, int]:
    """Read ``processed/resolution_log.jsonl``, skipping malformed lines.

    Args:
        data_root: The data directory.

    Returns:
        The rows as a DataFrame (nested values rendered as JSON text so the
        frame is displayable) and the number of lines that could not be parsed.
    """
    path = data_root / "processed" / "resolution_log.jsonl"
    if not path.is_file():
        return pd.DataFrame(), 0

    rows: list[dict[str, Any]] = []
    bad = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("resolution_log_unreadable", path=str(path), error=str(exc))
        return pd.DataFrame(), 0

    for line in lines:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if not isinstance(parsed, dict):
            bad += 1
            continue
        rows.append({k: _display_cell(v) for k, v in parsed.items()})
    return pd.DataFrame(rows), bad


def _display_cell(value: Any) -> Any:
    """Render nested JSON values as text so they sit in a table cell."""
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return value


def list_imputed_scales(data_root: Path) -> list[str]:
    """Scales that have a directory under ``data/imputed``.

    Args:
        data_root: The data directory.

    Returns:
        Scale names, sorted.
    """
    imputed = data_root / "imputed"
    if not imputed.is_dir():
        return []
    return sorted(p.name for p in imputed.iterdir() if p.is_dir())


def _round_dirs(data_root: Path, scale: str) -> list[tuple[int, Path]]:
    """The ``round_{n}`` directories of a scale, ordered by n."""
    scale_dir = data_root / "imputed" / scale
    if not scale_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in scale_dir.iterdir():
        match = _ROUND_DIR.fullmatch(child.name)
        if child.is_dir() and match:
            found.append((int(match.group(1)), child))
    return sorted(found)


def read_rounds(data_root: Path, scale: str) -> Frame:
    """Stack every round's ``sides.csv`` into one frame.

    Args:
        data_root: The data directory.
        scale: ``'neutral'`` or ``'published'``.

    Returns:
        The sides with an added ``round`` column; empty when nothing is on disk.
    """
    frames: list[Frame] = []
    for number, directory in _round_dirs(data_root, scale):
        csv_path = directory / "sides.csv"
        if not csv_path.is_file():
            continue
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logger.warning("imputed_round_unreadable", path=str(csv_path), error=str(exc))
            continue
        frame["round"] = number
        frames.append(frame)
    if not frames:
        return empty_frame(_ROUND_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def read_manifests(data_root: Path, scale: str) -> Frame:
    """One row per round manifest of a scale.

    Args:
        data_root: The data directory.
        scale: ``'neutral'`` or ``'published'``.

    Returns:
        The manifests, ordered by round; empty when none exist.
    """
    rows: list[dict[str, Any]] = []
    for _, directory in _round_dirs(data_root, scale):
        manifest = _read_json(directory / "manifest.json")
        if isinstance(manifest, dict):
            rows.append(manifest)
    return pd.DataFrame(rows)


def read_gate_report(data_root: Path, scale: str) -> dict[str, Any]:
    """Read ``imputed/{scale}/gate_report.json``.

    Args:
        data_root: The data directory.
        scale: ``'neutral'`` or ``'published'``.

    Returns:
        The report, or an empty dict when absent or malformed.
    """
    report = _read_json(data_root / "imputed" / scale / "gate_report.json")
    return report if isinstance(report, dict) else {}


def _read_json(path: Path) -> Any:
    """Parse a JSON file, returning None when it is absent or malformed."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("json_unreadable", path=str(path), error=str(exc))
        return None


def find_neutral_draws(data_root: Path) -> list[Path]:
    """Neutral-draws files from reconcile, newest run last.

    Args:
        data_root: The data directory.

    Returns:
        Paths under ``processed/reconcile``, ordered by run id.
    """
    directory = data_root / "processed" / "reconcile"
    if not directory.is_dir():
        return []
    found = []
    for child in directory.iterdir():
        match = _NEUTRAL_FILE.fullmatch(child.name)
        if match:
            found.append((int(match.group(1)), child))
    return [path for _, path in sorted(found)]


def summarise_neutral_draws(path: Path) -> Frame:
    """Per-side mean, sd and quantiles of a neutral draws file.

    Args:
        path: A ``neutral_draws_run{n}.npz`` file.

    Returns:
        Columns ``side_id``, ``battle_id``, ``mean``, ``sd``, ``q025``,
        ``q50``, ``q975``; empty when the file is absent or malformed.
    """
    columns = ["side_id", "battle_id", "mean", "sd", "q025", "q50", "q975"]
    if not path.is_file():
        return empty_frame(columns)
    try:
        with np.load(path) as archive:
            side_id = np.asarray(archive["side_id"])
            battle_id = np.asarray(archive["battle_id"])
            draws = np.asarray(archive["draws"], dtype=float)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("neutral_draws_unreadable", path=str(path), error=str(exc))
        return empty_frame(columns)

    if draws.ndim != 2 or draws.shape[1] != side_id.shape[0]:
        logger.warning("neutral_draws_shape_mismatch", path=str(path), shape=draws.shape)
        return empty_frame(columns)

    lo, mid, hi = np.quantile(draws, _QUANTILES, axis=0)
    return pd.DataFrame(
        {
            "side_id": side_id,
            "battle_id": battle_id,
            "mean": draws.mean(axis=0),
            "sd": draws.std(axis=0),
            "q025": lo,
            "q50": mid,
            "q975": hi,
        }
    )
