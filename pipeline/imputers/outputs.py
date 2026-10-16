"""Writing the imputed datasets, manifests and gate report to disk.

Layout: ``{root}/{scale}/round_{n}/sides.csv`` and ``manifest.json`` per round,
``{root}/{scale}/gate_report.json``. There is deliberately no scale-less copy:
a sensitivity run must not overwrite the primary's report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from pipeline.imputers.conditional import RoundSet
from pipeline.imputers.design import ImputeDesign
from pipeline.imputers.gates import GateResult

__all__ = [
    "GATE_REPORT_NAME",
    "config_hash",
    "write_gate_report",
    "write_rounds",
]

logger = structlog.get_logger()

GATE_REPORT_NAME = "gate_report.json"
_ROUND_DIR = re.compile(r"round_(\d+)$")


def config_hash(config: dict[str, Any]) -> str:
    """A short stable hash of the run configuration.

    Args:
        config: A JSON-serialisable mapping.

    Returns:
        The first 16 hex characters of its sha256.
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_rounds(
    root: Path,
    *,
    scale: str,
    rounds: RoundSet,
    design: ImputeDesign,
    run_id: int,
    seed: int,
    reconcile_run_id: int | None,
    config_digest: str,
) -> list[Path]:
    """Write one directory per imputation round.

    Stale ``round_{n}`` directories left by an earlier run with more rounds are
    removed, so the directory always describes exactly this run.

    Args:
        root: The imputed-data root, normally ``data/imputed``.
        scale: ``'neutral'`` or ``'published'``.
        rounds: The rounds.
        design: The design (side and battle ids).
        run_id: This invocation's ``model_runs`` id.
        seed: Base seed used for the conditional draws.
        reconcile_run_id: The reconcile run behind the likelihood.
        config_digest: From :func:`config_hash`.

    Returns:
        The round directories written.
    """
    scale_dir = root / scale
    scale_dir.mkdir(parents=True, exist_ok=True)
    n_rounds = rounds.values.shape[0]
    for stale in scale_dir.iterdir():
        match = _ROUND_DIR.fullmatch(stale.name)
        if stale.is_dir() and match and int(match.group(1)) >= n_rounds:
            shutil.rmtree(stale)

    written: list[Path] = []
    for n in range(n_rounds):
        directory = scale_dir / f"round_{n}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "sides.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["side_id", "battle_id", "log_troops", "is_imputed", "scale"])
            for i in range(design.n_sides):
                writer.writerow(
                    [
                        int(design.side_ids[i]),
                        int(design.battle_ids[i]),
                        repr(float(rounds.values[n, i])),
                        bool(rounds.is_imputed[i]),
                        scale,
                    ]
                )
        manifest = {
            "run_id": run_id,
            "scale": scale,
            "seed": seed,
            "round": n,
            "draw_index": rounds.draw_index[n],
            "reconcile_run_id": reconcile_run_id,
            "config_hash": config_digest,
            "n_sides": design.n_sides,
            "n_imputed": int(rounds.is_imputed.sum()),
            "shared_error_propagated": rounds.theta_index is not None,
            "theta_draw_index": None if rounds.theta_index is None else rounds.theta_index[n],
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        written.append(directory)
    return written


def write_gate_report(
    root: Path,
    *,
    scale: str,
    run_id: int,
    written: bool,
    gates: Sequence[GateResult],
    shared_error: dict[str, object] | None = None,
) -> Path:
    """Write the gate report under the scale directory.

    Args:
        root: The imputed-data root.
        scale: ``'neutral'`` or ``'published'``.
        run_id: This invocation's ``model_runs`` id.
        written: Whether the rounds were written (False when the fit did not
            converge, in which case the gates describe nothing on disk).
        gates: The dataset-level gate results.
        shared_error: From ``shared_error_summary``: whether the shared
            reconcile error was propagated, with the diagnostics.

    Returns:
        The scale-level report path.
    """
    report = {
        "run_id": run_id,
        "scale": scale,
        "rounds_written": written,
        "gates": [g.as_dict() for g in gates],
    }
    if shared_error is not None:
        report["shared_error"] = shared_error
    text = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    scale_dir = root / scale
    scale_dir.mkdir(parents=True, exist_ok=True)
    path = scale_dir / GATE_REPORT_NAME
    path.write_text(text, encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    """Serialise numpy scalars that sneak into a gate's text."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
