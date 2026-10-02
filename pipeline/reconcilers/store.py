"""Writing reconciled estimates and source biases into the database.

This is the module that finally writes an ``est_troops_total`` -- everything
upstream of it (:mod:`pipeline.reconcilers.load`, ``design.py``, ``model.py``,
``summarise.py``) computes a :class:`~pipeline.reconcilers.records.SideEstimate`
or :class:`~pipeline.reconcilers.records.SourceBias`, but nothing has ever
persisted one. See ``pipeline/stages/reconcile.py`` for the run sequence this
module's three groups of statements are called in.

One schema detail is load-bearing here: ``config/schema.sql`` gives troops
and casualties asymmetric column families on ``battle_sides`` rather than
one column family per quantity. Troops write ``est_troops_total`` / ``_lo``
/ ``_hi`` / ``_run_id`` / ``_n_reports`` / ``_n_sources`` / ``_method`` /
``_updated_at``; casualties write the same shape under ``est_casualties``
(no ``_total`` suffix on the point estimate, unlike troops). The column
names differ per quantity, so they cannot be parameterised into one UPDATE
the way the row values can -- SQL does not bind identifiers -- and this
module carries one statement per quantity instead.

``sources`` has one ``bias_run_id`` and one ``bias_updated_at``, not one per
quantity, alongside ``bias_n_troop_reports`` and ``bias_n_casualty_reports``,
which *are* split. That single shared column is not a provenance gap in
practice: ``pipeline/stages/reconcile.py`` fits both quantities under one
``model_runs`` row per stage invocation (see that module's docstring for
why), so a source fitted for both troops and casualties in the same run
still ends up with ``bias_run_id`` pointing at the one run both
``troop_bias_mu`` and ``casualty_bias_mu`` actually came from.
"""

from __future__ import annotations

import json
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.reconcilers.records import Quantity, SideEstimate, SourceBias

__all__ = [
    "complete_model_run",
    "start_model_run",
    "write_side_estimates",
    "write_source_biases",
]

logger = structlog.get_logger()

_INSERT_MODEL_RUN = text(
    """
    INSERT INTO model_runs (run_name, model_type, config, started_at)
    VALUES (:run_name, :model_type, CAST(:config AS JSONB), now())
    RETURNING run_id
    """
)

_COMPLETE_MODEL_RUN = text(
    """
    UPDATE model_runs
    SET completed_at = now(), diagnostics = CAST(:diagnostics AS JSONB)
    WHERE run_id = :run_id
    """
)

_UPDATE_SIDE_TROOPS = text(
    """
    UPDATE battle_sides SET
        est_troops_total      = :value,
        est_troops_total_lo   = :lo,
        est_troops_total_hi   = :hi,
        est_troops_run_id     = :run_id,
        est_troops_n_reports  = :n_reports,
        est_troops_n_sources  = :n_sources,
        est_troops_method     = :method,
        est_troops_updated_at = :updated_at
    WHERE side_id = :side_id
    """
)

_UPDATE_SIDE_CASUALTIES = text(
    """
    UPDATE battle_sides SET
        est_casualties             = :value,
        est_casualties_lo          = :lo,
        est_casualties_hi          = :hi,
        est_casualties_run_id      = :run_id,
        est_casualties_n_reports   = :n_reports,
        est_casualties_n_sources   = :n_sources,
        est_casualties_method      = :method,
        est_casualties_updated_at  = :updated_at
    WHERE side_id = :side_id
    """
)

_UPDATE_SOURCE_TROOPS = text(
    """
    UPDATE sources SET
        troop_bias_mu        = :bias_mu,
        troop_bias_sd        = :bias_sd,
        troop_sigma_mu       = :sigma_mu,
        troop_sigma_sd       = :sigma_sd,
        bias_run_id          = :run_id,
        bias_n_troop_reports = :n_reports,
        bias_updated_at      = :updated_at
    WHERE source_id = :source_id
    """
)

_UPDATE_SOURCE_CASUALTIES = text(
    """
    UPDATE sources SET
        casualty_bias_mu        = :bias_mu,
        casualty_bias_sd        = :bias_sd,
        casualty_sigma_mu       = :sigma_mu,
        casualty_sigma_sd       = :sigma_sd,
        bias_run_id             = :run_id,
        bias_n_casualty_reports = :n_reports,
        bias_updated_at         = :updated_at
    WHERE source_id = :source_id
    """
)

_SIDE_STATEMENT_BY_QUANTITY: Final[dict[Quantity, Any]] = {
    "troops": _UPDATE_SIDE_TROOPS,
    "casualties": _UPDATE_SIDE_CASUALTIES,
}

_SOURCE_STATEMENT_BY_QUANTITY: Final[dict[Quantity, Any]] = {
    "troops": _UPDATE_SOURCE_TROOPS,
    "casualties": _UPDATE_SOURCE_CASUALTIES,
}


def start_model_run(conn: Any, *, model_type: str, run_name: str, config: dict[str, Any]) -> int:
    """Insert a new ``model_runs`` row and return its id.

    Called before the model is sampled, not after: the run_id is what
    :func:`~pipeline.reconcilers.summarise.summarise_fit` stamps every record
    with, so it has to exist first. The caller is responsible for making this
    insert durable (via a connection-level commit) before starting a
    long-running fit, if it wants a failed fit to leave an inspectable
    incomplete row rather than lose the attempt entirely -- see
    ``pipeline/stages/reconcile.py`` for how the stage runner does that on a
    connection it owns.

    Args:
        conn: An open database connection.
        model_type: The ``model_runs.model_type`` value, e.g.
            ``"source_disagreement"``.
        run_name: A human-readable label for this run. ``model_runs.run_name``
            is NOT NULL; there is no sensible default, so the caller supplies
            one that says at least which quantity this run is for.
        config: The run's configuration -- priors, sampler settings, the
            inflation prior's provenance -- as a JSON-serialisable dict.

    Returns:
        The new ``run_id``.
    """
    row = conn.execute(
        _INSERT_MODEL_RUN,
        {
            "run_name": run_name,
            "model_type": model_type,
            # default=str: config/inflation_priors.yaml's provenance block
            # carries a plain datetime.date (fitted_on), which json can only
            # serialise via a fallback, not natively.
            "config": json.dumps(config, default=str),
        },
    ).fetchone()
    run_id = int(row[0])
    logger.info(
        "reconcile_model_run_started", run_id=run_id, run_name=run_name, model_type=model_type
    )
    return run_id


def complete_model_run(conn: Any, run_id: int, diagnostics: dict[str, Any]) -> None:
    """Stamp a ``model_runs`` row complete, with its sampler diagnostics.

    Only called after a fit has actually finished: ``completed_at`` and
    ``diagnostics`` are what ``agents/reconcile.yaml``'s ``model_run_recorded``
    and ``model_convergence`` gates read, so a run that never reaches this
    call stays correctly invisible to both.

    Args:
        conn: An open database connection.
        run_id: The row to complete, from :func:`start_model_run`.
        diagnostics: The JSON-serialisable diagnostics payload from
            :func:`~pipeline.reconcilers.summarise.summarise_fit`.
    """
    conn.execute(
        _COMPLETE_MODEL_RUN,
        {"run_id": run_id, "diagnostics": json.dumps(diagnostics, default=str)},
    )
    logger.info("reconcile_model_run_completed", run_id=run_id)


def write_side_estimates(conn: Any, estimates: list[SideEstimate]) -> int:
    """Write one quantity's fitted estimates onto ``battle_sides``.

    Every side named in ``estimates`` is updated unconditionally -- there is
    no upsert to reason about, because ``battle_sides`` rows already exist
    (extract wrote them) and this only ever UPDATEs. Re-running with the same
    seed and the same data overwrites each side with the same values, which
    is what makes the stage idempotent rather than accumulating drift.

    Args:
        conn: An open database connection.
        estimates: One quantity's estimates, from
            :func:`~pipeline.reconcilers.summarise.summarise_fit`. Mixed
            quantities are fine; each row is routed to the right column
            family by its own ``quantity``.

    Returns:
        How many rows were written.
    """
    written = 0
    for estimate in estimates:
        statement = _SIDE_STATEMENT_BY_QUANTITY[estimate.quantity]
        conn.execute(
            statement,
            {
                "side_id": estimate.side_id,
                "value": estimate.value,
                "lo": estimate.lo,
                "hi": estimate.hi,
                "run_id": estimate.run_id,
                "n_reports": estimate.n_reports,
                "n_sources": estimate.n_sources,
                "method": estimate.method,
                "updated_at": estimate.updated_at,
            },
        )
        written += 1
    return written


def write_source_biases(conn: Any, biases: list[SourceBias]) -> int:
    """Write one quantity's fitted source biases onto ``sources``.

    A source with no reports of this quantity never appears in ``biases`` --
    see :func:`~pipeline.reconcilers.summarise.summarise_fit`'s
    ``_source_biases`` -- so it is never touched here, and keeps whatever the
    ``troop_bias_*`` / ``casualty_bias_*`` columns already held (the schema
    default, i.e. the prior, on a source that has never been fitted).

    Args:
        conn: An open database connection.
        biases: One quantity's biases.

    Returns:
        How many rows were written.
    """
    written = 0
    for bias in biases:
        statement = _SOURCE_STATEMENT_BY_QUANTITY[bias.quantity]
        conn.execute(
            statement,
            {
                "source_id": bias.source_id,
                "bias_mu": bias.bias_mu,
                "bias_sd": bias.bias_sd,
                "sigma_mu": bias.sigma_mu,
                "sigma_sd": bias.sigma_sd,
                "run_id": bias.run_id,
                "n_reports": bias.n_reports,
                "updated_at": bias.updated_at,
            },
        )
        written += 1
    return written
