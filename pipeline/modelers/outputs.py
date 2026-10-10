"""Files the model stage leaves beside the database rows.

Layout: ``{root}/run{pooled_run_id}/`` with ``rounds_summary.json``,
``rubin_per_general.csv``, ``components.json``, ``gate_report.json`` and
``pooled_draws.npz``. The run directory is keyed on the pooled run so a rerun
never overwrites an earlier ranking's evidence.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "GATE_REPORT_NAME",
    "run_directory",
    "to_jsonable",
    "write_columns_csv",
    "write_draws_npz",
    "write_gate_report",
    "write_json",
]

GATE_REPORT_NAME: Final[str] = "gate_report.json"
RUN_DIR_PREFIX: Final[str] = "run"


def run_directory(root: Path, run_id: int) -> Path:
    """The directory for one pooled run.

    Args:
        root: The diagnostics root.
        run_id: The pooled run's ``model_runs`` id.

    Returns:
        ``{root}/run{run_id}`` (not created).
    """
    return root / f"{RUN_DIR_PREFIX}{run_id}"


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays, and non-finite floats, for JSON.

    Args:
        value: Any nesting of mappings, sequences, numpy values and scalars.

    Returns:
        A structure ``json.dumps`` accepts; NaN and infinity become None.
    """
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, list | tuple | set):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> Path:
    """Write a JSON file, creating parents.

    Args:
        path: Destination.
        payload: Any structure :func:`to_jsonable` accepts.

    Returns:
        The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_gate_report(
    directory: Path,
    *,
    run_id: int,
    estimates_written: bool,
    gates: Sequence[Mapping[str, Any]],
) -> Path:
    """Write ``gate_report.json``.

    Args:
        directory: The run directory.
        run_id: The pooled run id.
        estimates_written: Whether estimates reached the database; False when
            a round failed, in which case ``gates`` is empty.
        gates: Gate dicts from :func:`pipeline.modelers.gates.gate_dict`.

    Returns:
        The report path.
    """
    return write_json(
        directory / GATE_REPORT_NAME,
        {"run_id": run_id, "estimates_written": estimates_written, "gates": list(gates)},
    )


def write_columns_csv(path: Path, columns: Mapping[str, Sequence[Any] | np.ndarray]) -> Path:
    """Write equal-length columns as a CSV with a header row.

    Args:
        path: Destination.
        columns: Column name to values; all the same length.

    Returns:
        The path.

    Raises:
        ValueError: If the columns differ in length.
    """
    lengths = {len(v) for v in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"columns differ in length: {sorted(lengths)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        for row in zip(*columns.values(), strict=True):
            writer.writerow(
                [repr(float(v)) if isinstance(v, float | np.floating) else v for v in row]
            )
    return path


def write_draws_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write the pooled draws and entry means for evaluate.

    Args:
        path: Destination.
        arrays: Named arrays.

    Returns:
        The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(arrays)
    np.savez_compressed(path, **payload)
    return path
