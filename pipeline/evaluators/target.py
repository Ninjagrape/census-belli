"""The run the evaluate stage validates, and the inputs rebuilt from it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import text

from pipeline.modelers.design import x_difference
from pipeline.modelers.load import ImputedRound, load_log_troop_measurements
from pipeline.modelers.records import ModelDesign
from pipeline.quality import EVALUABLE_RUN_PREDICATE

__all__ = [
    "PublishedRun",
    "load_target_run",
    "measured_log_troops",
    "round_x_diffs",
]

_TARGET_RUN = text(
    f"SELECT run_id, config FROM model_runs WHERE {EVALUABLE_RUN_PREDICATE} "
    "ORDER BY run_id DESC LIMIT 1"
)
_ESTIMATES = text(
    """
    SELECT e.general_id, e.skill_mean, e.total_war, e.rank, e.n_battles,
           e.skill_ci_lo, e.skill_ci_hi, g.wikidata_id, g.canonical_name
    FROM general_skill_estimates e
    JOIN generals g ON g.general_id = e.general_id
    WHERE e.run_id = :run_id
    ORDER BY e.general_id
    """
)


@dataclass(frozen=True)
class PublishedRun:
    """The published run's per-general estimates.

    All arrays are aligned and sorted by ``general_id``. ``rank`` and
    ``total_war`` are NaN where the row has no value.

    Attributes:
        run_id: The pooled ``model_runs`` id.
        imputation_run_id: The impute run its rounds came from.
        general_ids: Raw general ids.
        names: Canonical names.
        wikidata_ids: Q-ids, None where a general has none.
        skill_mean: Pooled posterior mean skill.
        total_war: Total expected WAR.
        rank: Published rank (1 is best).
        n_battles: Included battles per general.
        ci_lo: Lower 95% skill bound.
        ci_hi: Upper 95% skill bound.
    """

    run_id: int
    imputation_run_id: int
    general_ids: np.ndarray
    names: list[str]
    wikidata_ids: list[str | None]
    skill_mean: np.ndarray
    total_war: np.ndarray
    rank: np.ndarray
    n_battles: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray

    @property
    def score(self) -> np.ndarray:
        """Higher-is-better score reproducing the published order."""
        return -self.rank

    def rank_by_qid(self) -> dict[str, int]:
        """Published rank per Wikidata Q-id, for generals that have one."""
        return {
            q: int(r)
            for q, r in zip(self.wikidata_ids, self.rank, strict=True)
            if q is not None and np.isfinite(r)
        }


def _nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _imputation_run_id(config: Any) -> int:
    parsed = json.loads(config) if isinstance(config, str) else config
    value = parsed.get("imputation_run_id") if isinstance(parsed, dict) else None
    if value is None:
        raise ValueError("the target run's config has no imputation_run_id")
    return int(value)


def load_target_run(conn: Any) -> PublishedRun | None:
    """Load the newest converged pooled run that wrote estimates.

    Args:
        conn: An open database connection.

    Returns:
        The run and its estimates, or None when no such run exists.

    Raises:
        ValueError: If the run's config names no imputation run.
    """
    row = conn.execute(_TARGET_RUN).fetchone()
    if row is None:
        return None
    run_id = int(row[0])
    imputation_run_id = _imputation_run_id(row[1])
    rows = conn.execute(_ESTIMATES, {"run_id": run_id}).fetchall()
    return PublishedRun(
        run_id=run_id,
        imputation_run_id=imputation_run_id,
        general_ids=np.array([int(r[0]) for r in rows], dtype=np.int64),
        names=[str(r[8]) for r in rows],
        wikidata_ids=[None if r[7] is None else str(r[7]) for r in rows],
        skill_mean=np.array([_nan(r[1]) for r in rows]),
        total_war=np.array([_nan(r[2]) for r in rows]),
        rank=np.array([_nan(r[3]) for r in rows]),
        n_battles=np.array([int(r[4]) for r in rows], dtype=np.int64),
        ci_lo=np.array([_nan(r[5]) for r in rows]),
        ci_hi=np.array([_nan(r[6]) for r in rows]),
    )


def round_x_diffs(design: ModelDesign, rounds: Sequence[ImputedRound]) -> list[np.ndarray]:
    """``x_A - x_B`` per round, naming the round when a side is missing.

    Args:
        design: The design.
        rounds: The imputed rounds.

    Returns:
        One vector per round.

    Raises:
        ValueError: If a round lacks a side of the design.
    """
    out: list[np.ndarray] = []
    for rnd in rounds:
        try:
            out.append(x_difference(design, rnd.log_troops))
        except ValueError as exc:
            raise ValueError(
                f"imputed round {rnd.index} is incomplete: {exc}. Rounds are complete by "
                "construction, so the files do not match this database; re-run impute."
            ) from exc
    return out


def measured_log_troops(conn: Any, design: ModelDesign) -> tuple[np.ndarray, np.ndarray]:
    """Measured log-troop mean and sd per side index (NaN where unmeasured).

    Args:
        conn: An open database connection.
        design: The design.

    Returns:
        ``(obs_mean, obs_sd)``, each of shape ``(n_sides,)``.
    """
    measured = load_log_troop_measurements(conn)
    obs_mean = np.full(design.n_sides, np.nan)
    obs_sd = np.full(design.n_sides, np.nan)
    for i, side_id in enumerate(design.side_ids):
        if int(side_id) in measured:
            obs_mean[i], obs_sd[i] = measured[int(side_id)]
    return obs_mean, obs_sd
