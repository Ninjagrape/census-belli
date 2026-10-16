"""Reading the model stage's inputs: the database and the imputed rounds.

Years come from ``battles.year_astronomical``, never ``date_start``: a BC date
cannot be decoded in Python (docs/handover-log.md 5.1).

The imputed rounds are files, so they can outlive the database they were made
against. Every round's manifest carries the ``run_id`` of the impute run that
wrote it, and the stage refuses rounds unless that is the latest completed
neutral-scale imputation run whose diagnostics say it wrote rounds. The pure
checks (:func:`check_round_provenance`) take plain values so they run without a
database.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text

from pipeline.modelers.records import CommanderInput, SideInput

__all__ = [
    "ImputedRound",
    "check_round_provenance",
    "fill_camp_log_troops",
    "has_multi_side_battle",
    "latest_written_imputation_run_id",
    "load_battle_quality",
    "load_battle_types",
    "load_general_names",
    "load_log_troop_measurements",
    "load_side_inputs",
    "read_rounds",
    "subset_sides",
]

_ROUND_DIR = re.compile(r"round_(\d+)$")
_UNKNOWN_METHOD = "unknown"

_LOAD_SIDES = text(
    """
    SELECT bs.side_id, bs.battle_id, b.year_astronomical,
           CAST(bs.outcome AS TEXT), bs.is_defender,
           CAST(b.battle_type AS TEXT), bs.est_troops_neutral_log_mean
    FROM battle_sides bs
    JOIN battles b ON b.battle_id = bs.battle_id
    ORDER BY bs.side_id
    """
)

_LOAD_COMMANDERS = text(
    """
    SELECT side_id, general_id, attribution_weight, attribution_method,
           CAST(command_role AS TEXT)
    FROM battle_commanders
    ORDER BY bc_id
    """
)

_LOAD_MEASUREMENTS = text(
    """
    SELECT side_id, est_troops_neutral_log_mean, est_troops_neutral_log_sd
    FROM battle_sides
    WHERE est_troops_neutral_log_mean IS NOT NULL
    """
)

_LOAD_TYPES = text("SELECT battle_id, CAST(battle_type AS TEXT) FROM battles")
_LOAD_QUALITY = text(
    "SELECT battle_id, data_quality_score FROM battles WHERE data_quality_score IS NOT NULL"
)
_LOAD_NAMES = text("SELECT general_id, canonical_name FROM generals")

# diagnostics.written is a JSON boolean; ->> renders it as 'true'.
_IMPUTE_RUN = text(
    """
    SELECT run_id FROM model_runs
    WHERE model_type = 'imputation'
      AND completed_at IS NOT NULL
      AND diagnostics ->> 'scale' = :scale
      AND diagnostics ->> 'written' = 'true'
    ORDER BY run_id DESC
    LIMIT 1
    """
)


@dataclass(frozen=True)
class ImputedRound:
    """One imputed dataset read from disk.

    Attributes:
        index: The ``n`` of ``round_{n}``.
        run_id: The impute run that wrote it (from the manifest).
        scale: The scale recorded in the manifest.
        log_troops: Log troop count per ``side_id``.
    """

    index: int
    run_id: int
    scale: str
    log_troops: dict[int, float]


def load_side_inputs(conn: Any) -> list[SideInput]:
    """Load every side with its commanders.

    A NULL ``attribution_weight`` becomes NaN, which the design drops like a
    zero weight; a NULL ``attribution_method`` becomes ``'unknown'``.

    Args:
        conn: An open database connection.

    Returns:
        One :class:`SideInput` per ``battle_sides`` row, ordered by side id.
    """
    by_side: dict[int, list[CommanderInput]] = defaultdict(list)
    for side_id, general_id, weight, method, role in conn.execute(_LOAD_COMMANDERS).fetchall():
        by_side[int(side_id)].append(
            CommanderInput(
                general_id=int(general_id),
                attribution_weight=float("nan") if weight is None else float(weight),
                attribution_method=_UNKNOWN_METHOD if method is None else str(method),
                command_role=None if role is None else str(role),
            )
        )
    return [
        SideInput(
            side_id=int(side_id),
            battle_id=int(battle_id),
            year=None if year is None else int(year),
            outcome=outcome,
            is_defender=None if is_defender is None else bool(is_defender),
            commanders=tuple(by_side.get(int(side_id), ())),
            battle_type=battle_type,
            log_troops=None if log_troops is None else float(log_troops),
        )
        for (
            side_id,
            battle_id,
            year,
            outcome,
            is_defender,
            battle_type,
            log_troops,
        ) in conn.execute(_LOAD_SIDES).fetchall()
    ]


def has_multi_side_battle(sides: Sequence[SideInput]) -> bool:
    """Whether any battle has three or more sides (so a camp may be merged)."""
    counts: dict[int, int] = defaultdict(int)
    for side in sides:
        counts[side.battle_id] += 1
    return any(n >= 3 for n in counts.values())


def fill_camp_log_troops(
    sides: Sequence[SideInput], rounds: Sequence[ImputedRound]
) -> list[SideInput]:
    """Give each unmeasured side of a 3+ side battle its mean imputed log troops.

    A merged camp splits its credit among member sides by troops. A member
    with no reconcile estimate would otherwise force equal shares, so it uses
    its mean over the imputed rounds instead. One value is used for every
    round, which keeps a single design (and so one set of commander weights)
    across the fits.

    Args:
        sides: All sides.
        rounds: The imputed rounds.

    Returns:
        The sides, with ``log_troops`` filled where it was None, a round had a
        value, and the side belongs to a battle with three or more sides.
    """
    if not rounds or not has_multi_side_battle(sides):
        return list(sides)
    sizes: dict[int, int] = defaultdict(int)
    for side in sides:
        sizes[side.battle_id] += 1
    out: list[SideInput] = []
    for side in sides:
        if side.log_troops is None and sizes[side.battle_id] >= 3:
            values = [r.log_troops[side.side_id] for r in rounds if side.side_id in r.log_troops]
            if values:
                side = replace(side, log_troops=float(np.mean(values)))
        out.append(side)
    return out


def subset_sides(sides: Sequence[SideInput], limit: int | None) -> list[SideInput]:
    """Keep every side of the first ``limit`` battles, for smoke runs.

    Args:
        sides: All sides.
        limit: Maximum number of battles, or None for all.

    Returns:
        The sides of the first ``limit`` battles by id.
    """
    if limit is None:
        return list(sides)
    keep = set(sorted({s.battle_id for s in sides})[:limit])
    return [s for s in sides if s.battle_id in keep]


def load_log_troop_measurements(conn: Any) -> dict[int, tuple[float, float]]:
    """Reconcile's neutral log-troop estimate per side, where it wrote one.

    Args:
        conn: An open database connection.

    Returns:
        ``side_id -> (log_mean, log_sd)``; sides with a NULL mean are absent
        and a NULL sd comes back as NaN (the joint model treats it as missing).
    """
    return {
        int(side_id): (float(mean), float("nan") if sd is None else float(sd))
        for side_id, mean, sd in conn.execute(_LOAD_MEASUREMENTS).fetchall()
    }


def load_battle_types(conn: Any) -> dict[int, str | None]:
    """Battle type per battle, for the included-battle counts in diagnostics.

    Args:
        conn: An open database connection.

    Returns:
        ``battle_id -> battle_type`` text (None when unset).
    """
    return {int(b): None if t is None else str(t) for b, t in conn.execute(_LOAD_TYPES).fetchall()}


def load_battle_quality(conn: Any) -> dict[int, float]:
    """Data quality score per battle, for stratifying held-out accuracy.

    Args:
        conn: An open database connection.

    Returns:
        ``battle_id -> data_quality_score``; battles with a NULL score are absent.
    """
    return {int(b): float(q) for b, q in conn.execute(_LOAD_QUALITY).fetchall()}


def load_general_names(conn: Any) -> dict[int, str]:
    """Canonical names, for logs and the per-general CSV.

    Args:
        conn: An open database connection.

    Returns:
        ``general_id -> canonical_name``.
    """
    return {int(gid): str(name) for gid, name in conn.execute(_LOAD_NAMES).fetchall()}


def latest_written_imputation_run_id(conn: Any, scale: str = "neutral") -> int | None:
    """The newest completed impute run of a scale whose fit was written.

    Args:
        conn: An open database connection.
        scale: The imputation scale.

    Returns:
        The run id, or None when no such run exists.
    """
    value = conn.execute(_IMPUTE_RUN, {"scale": scale}).scalar()
    return None if value is None else int(value)


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"cannot read {path}: {type(exc).__name__}") from exc
    if not isinstance(manifest, dict) or "run_id" not in manifest:
        raise ValueError(f"{path} has no run_id")
    return manifest


def _read_sides(directory: Path) -> dict[int, float]:
    path = directory / "sides.csv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {path}: {type(exc).__name__}") from exc
    values = {int(row["side_id"]): float(row["log_troops"]) for row in rows}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError(f"{path} has a non-finite log_troops value")
    return values


def read_rounds(
    root: Path, scale: str = "neutral", max_rounds: int | None = None
) -> list[ImputedRound]:
    """Read ``{root}/{scale}/round_{n}/`` for every round, in numeric order.

    Args:
        root: The imputed-data root.
        scale: The scale directory to read.
        max_rounds: Read at most this many rounds, or None for all.

    Returns:
        The rounds, ordered by index.

    Raises:
        FileNotFoundError: If the scale directory has no rounds, or a round
            lacks its manifest or sides file.
        ValueError: If a manifest's scale is not the requested one.
    """
    scale_dir = root / scale
    found: list[tuple[int, Path]] = []
    if scale_dir.is_dir():
        for entry in scale_dir.iterdir():
            match = _ROUND_DIR.fullmatch(entry.name)
            if entry.is_dir() and match:
                found.append((int(match.group(1)), entry))
    if not found:
        raise FileNotFoundError(
            f"no round_* directories under {scale_dir}: run the impute stage first "
            f"(--stages impute) with params.scale={scale}"
        )
    found.sort()
    if max_rounds is not None:
        found = found[:max_rounds]
    rounds: list[ImputedRound] = []
    for index, directory in found:
        manifest = _read_manifest(directory)
        if str(manifest.get("scale")) != scale:
            raise ValueError(
                f"{directory / 'manifest.json'} is scale {manifest.get('scale')!r}, "
                f"expected {scale!r}"
            )
        rounds.append(
            ImputedRound(
                index=index,
                run_id=int(manifest["run_id"]),
                scale=scale,
                log_troops=_read_sides(directory),
            )
        )
    return rounds


def check_round_provenance(rounds: Sequence[ImputedRound], expected_run_id: int | None) -> int:
    """Refuse rounds that are mixed or stale.

    Args:
        rounds: The rounds read from disk.
        expected_run_id: The latest written imputation run of the rounds' scale in the
            database, or None if there is none.

    Returns:
        The imputation run id every round shares.

    Raises:
        ValueError: If the rounds come from more than one impute run, or from
            a run that is not ``expected_run_id``.
    """
    run_ids = sorted({r.run_id for r in rounds})
    if len(run_ids) != 1:
        raise ValueError(
            f"imputed rounds come from several impute runs {run_ids}: a directory that was "
            "partly overwritten. Re-run the impute stage so every round has one origin."
        )
    run_id = run_ids[0]
    if expected_run_id is None or run_id != expected_run_id:
        raise ValueError(
            f"imputed rounds come from impute run {run_id}, but the latest impute run of "
            f"this scale that wrote rounds is {expected_run_id}: the files are stale or from "
            "a different database. Re-run impute, or point params.imputed_root at the "
            "matching directory."
        )
    return run_id
