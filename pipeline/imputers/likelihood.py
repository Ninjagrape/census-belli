"""Turning reconcile's output into a likelihood for the imputation model.

## The likelihood comes from reconcile, not from dividing out its prior

Until 2026-10-22 this module recovered each side's likelihood by dividing
reconcile's side prior ``N(m0, s0^2)`` out of the neutral posterior. The prior
is a Student-t, so for a side far from ``m0`` the posterior is wide and
near-bimodal, the Gaussian division judged it weak (precision at most
``min_lik_precision``) and the largest and smallest armies were discarded: 75 of
341 sides, and a kept-pair log-ratio slope of 0.43
(``docs/handover-log/31-troop-numbers-diagnosis-2026-10-22.md``, Q2).

Reconcile's neutral fit now exports each side's likelihood directly
(:mod:`pipeline.reconcilers.likelihood_export`): for every side with at least one
point row, the mean and sd on the log scale from its point rows at the fitted
noise, combined by precision with the Student-t Fisher information, and loadings
on the shared parameters ``g_unlabelled``, ``a_era`` and ``offset_bar``. This
module reads those into one :class:`~pipeline.imputers.records.BattleLikelihood`
per battle. The sides within a battle are independent in the likelihood (the
shared part is the loadings), so the precision is diagonal.

A side is missing when it has no point row: bound-only, or no usable report at
all. A side with a point row is never weak. There is no precision threshold.

## Shared error

Reconcile's neutral posterior has error shared across battles (the unlabelled
offset, the era slope, the average of the other offsets). Each round picks one
reconcile draw of those parameters, stratified on the leading one, and shifts
every side's likelihood mean by its loading times the centred draw, so all
battles in a round share that draw. An imputed side co-moves with the observed
sides of its era (``conditional.era_loading_table``). Offsets outside the three
parameters stay unmodelled shared error. A file with no parameter varying, or a
run with ``propagate_shared_error: false``, treats battles as independent and
logs ``impute_shared_error_not_propagated``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from pipeline.imputers.records import BattleLikelihood, SideRow

__all__ = [
    "LIKELIHOOD_KEYS",
    "LikelihoodStats",
    "NeutralDraws",
    "SharedError",
    "StaleDrawsFileError",
    "likelihoods_from_draws",
    "likelihoods_from_published",
    "load_neutral_draws",
    "shared_error_summary",
]

logger = structlog.get_logger()

# The 95% interval on the log scale spans 2 * 1.96 sd.
_INTERVAL_TO_SD = 3.92
_MIN_PUBLISHED_SD = 0.05
# A shared parameter this flat across draws carries no error to propagate.
_THETA_VARIANCE_FLOOR = 1e-12
_MIN_THETA_DRAWS = 3

# Per-side arrays reconcile's neutral fit writes for impute, from the run that
# introduced the exported likelihood.
LIKELIHOOD_KEYS = (
    "lik_mean",
    "lik_sd",
    "lik_has_point",
    "lik_loadings",
    "lik_loading_names",
)
_THETA_KEYS = ("g_unlabelled", "a_era", "offset_bar")


class StaleDrawsFileError(ValueError):
    """The neutral draws file predates the exported likelihood."""


@dataclass(frozen=True)
class NeutralDraws:
    """The contents of reconcile's ``neutral_draws_run{run_id}.npz``.

    ``draws`` is the neutral posterior of ``mu_side`` with shape ``(n_draws,
    n_sides)``; the evaluate stage regresses it on the shared parameters. The
    impute likelihood is the ``lik_*`` arrays, one entry per side in ``side_id``
    order: ``lik_mean``/``lik_sd`` (NaN where ``lik_has_point`` is False),
    ``lik_loadings`` ``(n_sides, p)`` on the parameters named in
    ``lik_loading_names``. ``g_unlabelled``, ``a_era`` and ``offset_bar`` are
    ``(n_draws,)`` draws row-aligned with ``draws`` (and with the loadings'
    regression); ``side_era`` and ``side_unlabelled_share`` are ``(n_sides,)``.
    Arrays a file does not carry are None, so evaluate can still read a file
    that predates the likelihood; :func:`likelihoods_from_draws` requires it.
    """

    run_id: int
    side_id: np.ndarray
    battle_id: np.ndarray
    draws: np.ndarray
    m0_mean: float
    s0_mean: float
    database: str | None = None
    started_at: str | None = None
    g_unlabelled: np.ndarray | None = None
    a_era: np.ndarray | None = None
    offset_bar: np.ndarray | None = None
    side_era: np.ndarray | None = None
    side_unlabelled_share: np.ndarray | None = None
    m0_draws: np.ndarray | None = None
    lik_mean: np.ndarray | None = None
    lik_sd: np.ndarray | None = None
    lik_has_point: np.ndarray | None = None
    lik_loadings: np.ndarray | None = None
    lik_loading_names: tuple[str, ...] = ()

    @property
    def theta(self) -> dict[str, np.ndarray]:
        """The shared-parameter draws the file carries, by name."""
        found = {name: getattr(self, name) for name in _THETA_KEYS}
        return {name: values for name, values in found.items() if values is not None}


@dataclass(frozen=True)
class SharedError:
    """Reconcile's shared parameter error, as draws to be applied through loadings.

    Attributes:
        names: The shared parameters kept, in loading-column order.
        theta_centered: ``(n_draws, p)`` draws of the parameters minus their
            means, one row per reconcile draw. A round picks one row.
        mean_abs_loading: Mean ``|loading|`` over kept sides and components.
        explained_share: Share of the kept sides' likelihood variance
            (loading-driven variance plus the Fisher variance) that the shared
            parameters carry, in ``[0, 1]``.
    """

    names: tuple[str, ...]
    theta_centered: np.ndarray
    mean_abs_loading: float
    explained_share: float


@dataclass
class LikelihoodStats:
    """Counts from building likelihoods, for the run's log and counters.

    ``shared`` is None when the shared error is not propagated, and
    ``shared_reason`` then says why. ``bound_only_side_ids`` are the in-scope
    sides the file lists without a point row; every other side without a
    likelihood is absent from the file, and the stage tells ``no_reports`` from
    ``other`` using the database.
    """

    stale_sides: int = 0
    bound_only_side_ids: frozenset[int] = frozenset()
    shared: SharedError | None = None
    shared_reason: str = "disabled"


_OPTIONAL_ARRAYS = (*_THETA_KEYS, "side_era", "side_unlabelled_share", "m0_draws")


def load_neutral_draws(path: Path, *, require_likelihood: bool = True) -> NeutralDraws:
    """Read reconcile's neutral draws file and validate its shapes.

    Args:
        path: ``neutral_draws_run{run_id}.npz``.
        require_likelihood: Refuse a file without the exported likelihood. The
            impute stage needs it; the evaluate stage's regime variants do not.

    Returns:
        The arrays and scalars.

    Raises:
        StaleDrawsFileError: If ``require_likelihood`` and the file has no
            exported likelihood (written by a reconcile before 2026-10-22).
            Re-run reconcile; there is no fallback, because the replacement
            would be a different likelihood.
        ValueError: If the arrays disagree in length or are not finite where
            they must be.
    """
    with np.load(path) as data:
        missing = [key for key in LIKELIHOOD_KEYS if key not in data.files]
        if missing and require_likelihood:
            raise StaleDrawsFileError(
                f"{path.name} has no exported likelihood (missing {missing}): it was written by "
                "a reconcile that predates it. Re-run reconcile (python -m pipeline.orchestrator "
                "--stages reconcile) to write a new draws file; impute does not fall back to "
                "dividing out reconcile's side prior."
            )
        run_id = int(data["run_id"])
        side_id = np.asarray(data["side_id"], dtype=np.int64)
        battle_id = np.asarray(data["battle_id"], dtype=np.int64)
        draws = np.asarray(data["draws"], dtype=np.float64)
        m0 = float(data["m0_mean"])
        s0 = float(data["s0_mean"])
        database = str(data["database"]) if "database" in data.files else None
        started_at = str(data["started_at"]) if "started_at" in data.files else None
        optional: dict[str, Any] = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in _OPTIONAL_ARRAYS
            if key in data.files
        }
        likelihood: dict[str, Any] = {}
        if not missing:
            likelihood = {
                "lik_mean": np.asarray(data["lik_mean"], dtype=np.float64),
                "lik_sd": np.asarray(data["lik_sd"], dtype=np.float64),
                "lik_has_point": np.asarray(data["lik_has_point"], dtype=bool),
                "lik_loadings": np.asarray(data["lik_loadings"], dtype=np.float64),
                "lik_loading_names": tuple(
                    str(n) for n in np.asarray(data["lik_loading_names"]).reshape(-1)
                ),
            }
    n_draws, n_sides = draws.shape if draws.ndim == 2 else (0, -1)
    if draws.ndim != 2 or n_sides != len(side_id) or len(battle_id) != len(side_id):
        raise ValueError(
            f"{path}: draws {draws.shape} does not match {len(side_id)} sides / "
            f"{len(battle_id)} battles"
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError(f"{path}: draws contain non-finite values")
    if not s0 > 0:
        raise ValueError(f"{path}: s0_mean must be positive, got {s0}")
    expected = {
        "g_unlabelled": (n_draws,),
        "a_era": (n_draws,),
        "offset_bar": (n_draws,),
        "m0_draws": (n_draws,),
        "side_era": (n_sides,),
        "side_unlabelled_share": (n_sides,),
    }
    for key, array in optional.items():
        if array.shape != expected[key]:
            raise ValueError(f"{path}: {key} has shape {array.shape}, expected {expected[key]}")
    if likelihood:
        _validate_likelihood(path, likelihood, n_sides)
    return NeutralDraws(
        run_id,
        side_id,
        battle_id,
        draws,
        m0,
        s0,
        database,
        started_at,
        **optional,
        **likelihood,
    )


def _validate_likelihood(path: Path, likelihood: dict[str, Any], n_sides: int) -> None:
    """Check the shapes and finiteness of the exported likelihood arrays.

    Raises:
        ValueError: If an array has the wrong shape, a side with a point row has
            a non-finite mean or a non-positive sd, or a loading is not finite.
    """
    for key in ("lik_mean", "lik_sd", "lik_has_point"):
        if likelihood[key].shape != (n_sides,):
            raise ValueError(
                f"{path}: {key} has shape {likelihood[key].shape}, expected ({n_sides},)"
            )
    names = likelihood["lik_loading_names"]
    loadings = likelihood["lik_loadings"]
    if loadings.shape != (n_sides, len(names)):
        raise ValueError(
            f"{path}: lik_loadings has shape {loadings.shape}, expected ({n_sides}, {len(names)})"
        )
    has_point = likelihood["lik_has_point"]
    if not (
        np.all(np.isfinite(likelihood["lik_mean"][has_point]))
        and np.all(likelihood["lik_sd"][has_point] > 0)
    ):
        raise ValueError(
            f"{path}: a side with a point row has a non-finite mean or non-positive sd"
        )
    if not np.all(np.isfinite(loadings)):
        raise ValueError(f"{path}: lik_loadings contain non-finite values")


def _shared_error(neutral: NeutralDraws, kept_rows: np.ndarray) -> SharedError | str:
    """Build the shared-error draws from the file, or say why that is not possible.

    Args:
        neutral: The loaded file.
        kept_rows: Positions (in file order) of the in-scope sides with a likelihood.

    Returns:
        The shared error, or a string naming why it is not available.
    """
    names = neutral.lik_loading_names
    if neutral.lik_loadings is None or neutral.lik_sd is None:
        return "shared_arrays_absent"
    if not names or kept_rows.size == 0:
        return "theta_has_no_variance"
    absent = [n for n in names if n not in neutral.theta]
    if absent:
        return "shared_arrays_absent"
    theta = np.column_stack([neutral.theta[n] for n in names])
    if theta.shape[0] < _MIN_THETA_DRAWS:
        return "too_few_draws"
    if not np.any(np.var(theta, axis=0) > _THETA_VARIANCE_FLOOR):
        return "theta_has_no_variance"
    centred = theta - theta.mean(axis=0)
    loadings = neutral.lik_loadings[kept_rows]
    shift_var = np.var(centred @ loadings.T, axis=0)
    fisher_var = neutral.lik_sd[kept_rows] ** 2
    total = float(np.sum(shift_var + fisher_var))
    explained = 0.0 if total <= 0 else float(np.sum(shift_var)) / total
    return SharedError(
        names=names,
        theta_centered=centred,
        mean_abs_loading=float(np.abs(loadings).mean()),
        explained_share=min(max(explained, 0.0), 1.0),
    )


def shared_error_summary(
    shared: SharedError | None, reason: str, theta_index: Sequence[int] | None = None
) -> dict[str, object]:
    """Describe whether and how the shared error was propagated, for the reports.

    Args:
        shared: The fitted shared error, or None.
        reason: Why it is None (ignored otherwise).
        theta_index: The reconcile draw each round used.

    Returns:
        A JSON-serialisable mapping.
    """
    if shared is None:
        return {"propagated": False, "reason": reason}
    return {
        "propagated": True,
        "theta": list(shared.names),
        "mean_abs_loading": shared.mean_abs_loading,
        "explained_variance_share": shared.explained_share,
        "theta_draw_index": None if theta_index is None else [int(i) for i in theta_index],
    }


def likelihoods_from_draws(
    neutral: NeutralDraws,
    in_scope_side_ids: set[int],
    *,
    propagate_shared: bool = True,
) -> tuple[list[BattleLikelihood], LikelihoodStats]:
    """Build one likelihood per battle from the neutral fit's exported likelihood.

    Args:
        neutral: The loaded draws file.
        in_scope_side_ids: Sides that exist in the database now. A side in the
            file but not here (deleted since the fit) is ignored.
        propagate_shared: Attach each side's loadings so a round can shift the
            likelihood by its reconcile draw (see :class:`SharedError`). Falls
            back to independent battles, logging
            ``impute_shared_error_not_propagated``, when the file's parameters do
            not vary.

    Returns:
        The likelihoods and the counts; ``stats.bound_only_side_ids`` has the
        sides the file lists without a point row.
    """
    if (
        neutral.lik_mean is None
        or neutral.lik_sd is None
        or neutral.lik_has_point is None
        or neutral.lik_loadings is None
    ):
        raise StaleDrawsFileError(
            f"neutral draws of run {neutral.run_id} carry no exported likelihood: re-run reconcile"
        )
    stats = LikelihoodStats()
    in_scope = np.array([int(s) in in_scope_side_ids for s in neutral.side_id], dtype=bool)
    stats.stale_sides = int((~in_scope).sum())
    kept = in_scope & neutral.lik_has_point
    stats.bound_only_side_ids = frozenset(
        int(s) for s in neutral.side_id[in_scope & ~neutral.lik_has_point]
    )

    kept_rows = np.flatnonzero(kept)
    if propagate_shared:
        fit = _shared_error(neutral, kept_rows)
        if isinstance(fit, str):
            stats.shared_reason = fit
        else:
            stats.shared = fit
    if stats.shared is None:
        logger.warning("impute_shared_error_not_propagated", reason=stats.shared_reason)

    columns_by_battle: dict[int, list[int]] = defaultdict(list)
    for column in kept_rows:
        columns_by_battle[int(neutral.battle_id[column])].append(int(column))

    out: list[BattleLikelihood] = []
    for battle_id in sorted(columns_by_battle):
        columns = columns_by_battle[battle_id]
        sd = neutral.lik_sd[columns]
        shift = neutral.lik_loadings[columns] if stats.shared is not None else None
        out.append(
            BattleLikelihood(
                battle_id,
                tuple(int(neutral.side_id[c]) for c in columns),
                neutral.lik_mean[columns].copy(),
                np.diag(1.0 / sd**2),
                "reconcile",
                shift,
            )
        )
    return out, stats


def likelihoods_from_published(
    sides: Sequence[SideRow], *, default_sd: float
) -> list[BattleLikelihood]:
    """Build diagonal likelihoods from the published estimates.

    Documented limitation of the sensitivity variant: sides within a battle
    share a regime mix, so their published errors are correlated, and the
    published interval already contains reconcile's side prior. Neither is
    corrected here.

    Args:
        sides: Every in-scope side; those without a published estimate are
            skipped.
        default_sd: Log sd to use when an estimate has no usable interval.

    Returns:
        One likelihood per battle with at least one published side.
    """
    by_battle: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for side in sides:
        if side.published is None:
            continue
        estimate, lo, hi = side.published
        if estimate <= 0:
            continue
        if lo is not None and hi is not None and lo > 0 and hi > lo:
            sd = max(float(np.log(hi / lo)) / _INTERVAL_TO_SD, _MIN_PUBLISHED_SD)
        else:
            sd = default_sd
        by_battle[side.battle_id].append((side.side_id, float(np.log(estimate)), sd))

    out: list[BattleLikelihood] = []
    for battle_id in sorted(by_battle):
        rows = by_battle[battle_id]
        out.append(
            BattleLikelihood(
                battle_id,
                tuple(r[0] for r in rows),
                np.array([r[1] for r in rows]),
                np.diag([1.0 / r[2] ** 2 for r in rows]),
                "published",
            )
        )
    return out
