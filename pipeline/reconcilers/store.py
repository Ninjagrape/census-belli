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
from datetime import UTC
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.classifiers.records import NO_ESTIMATE_NOTE_PREFIX
from pipeline.reconcilers.records import (
    NeutralSideEstimate,
    Quantity,
    SideEstimate,
    SourceBias,
)

__all__ = [
    "clear_unmodelled",
    "clear_unmodelled_neutral",
    "complete_model_run",
    "run_fingerprint",
    "start_model_run",
    "write_neutral_estimates",
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

_RUN_FINGERPRINT = text(
    "SELECT current_database(), started_at FROM model_runs WHERE run_id = :run_id"
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

# The neutral family has its own run id: it is written independently of the
# published fit, so the two can come from different runs.
_UPDATE_SIDE_NEUTRAL = text(
    """
    UPDATE battle_sides SET
        est_troops_neutral          = :value,
        est_troops_neutral_lo       = :lo,
        est_troops_neutral_hi       = :hi,
        est_troops_neutral_log_mean = :log_mean,
        est_troops_neutral_log_sd   = :log_sd,
        est_troops_neutral_run_id   = :run_id
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

# Only rows a fit wrote (run_id set) are ever cleared: a manually entered
# estimate has no run id and is not this stage's to remove.
_CLEAR_SIDE_TROOPS = text(
    """
    UPDATE battle_sides SET
        est_troops_total = NULL, est_troops_total_lo = NULL, est_troops_total_hi = NULL,
        est_troops_run_id = NULL, est_troops_n_reports = NULL, est_troops_n_sources = NULL,
        est_troops_method = NULL, est_troops_updated_at = NULL
    WHERE est_troops_run_id IS NOT NULL AND side_id <> ALL(CAST(:keep AS INT[]))
    """
)

_CLEAR_SIDE_CASUALTIES = text(
    """
    UPDATE battle_sides SET
        est_casualties = NULL, est_casualties_lo = NULL, est_casualties_hi = NULL,
        est_casualties_run_id = NULL, est_casualties_n_reports = NULL,
        est_casualties_n_sources = NULL, est_casualties_method = NULL,
        est_casualties_updated_at = NULL
    WHERE est_casualties_run_id IS NOT NULL AND side_id <> ALL(CAST(:keep AS INT[]))
    """
)

_CLEAR_SIDE_NEUTRAL = text(
    """
    UPDATE battle_sides SET
        est_troops_neutral = NULL, est_troops_neutral_lo = NULL, est_troops_neutral_hi = NULL,
        est_troops_neutral_log_mean = NULL, est_troops_neutral_log_sd = NULL,
        est_troops_neutral_run_id = NULL
    WHERE est_troops_neutral_run_id IS NOT NULL AND side_id <> ALL(CAST(:keep AS INT[]))
    """
)

_CLEAR_SOURCE_TROOPS = text(
    """
    UPDATE sources SET
        troop_bias_mu = NULL, troop_bias_sd = NULL,
        troop_sigma_mu = NULL, troop_sigma_sd = NULL, bias_n_troop_reports = 0
    WHERE (troop_sigma_mu IS NOT NULL OR COALESCE(bias_n_troop_reports, 0) > 0)
      AND source_id <> ALL(CAST(:keep AS INT[]))
    """
)

_CLEAR_SOURCE_CASUALTIES = text(
    """
    UPDATE sources SET
        casualty_bias_mu = NULL, casualty_bias_sd = NULL,
        casualty_sigma_mu = NULL, casualty_sigma_sd = NULL, bias_n_casualty_reports = 0
    WHERE (casualty_sigma_mu IS NOT NULL OR COALESCE(bias_n_casualty_reports, 0) > 0)
      AND source_id <> ALL(CAST(:keep AS INT[]))
    """
)

# A side the fit covered but whose estimate was not written (the fit missed the
# write thresholds) must reach missing_data_log, or downstream missingness reads
# it as a side nobody tried to fill. Only sides left without an estimate are
# logged: one that still carries an earlier run's value is a staleness matter,
# which estimates_are_not_stale reports. The notes prefix is the one load.py's
# write_unfillable clears at the start of each run, so a re-run does not stack
# rows, and the "estimate withheld" wording lets estimates_populated keep
# counting these sides as failures instead of as unfillable.
WITHHELD_NOTE_PREFIX: Final[str] = "reconcile: estimate withheld"

_LOG_WITHHELD_TROOPS = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, notes)
    SELECT bs.battle_id, bs.side_id, 'troop_total', 'unclassified', :notes
    FROM battle_sides bs
    WHERE bs.side_id = ANY(CAST(:side_ids AS INT[])) AND bs.est_troops_total IS NULL
      AND NOT EXISTS (
        SELECT 1 FROM missing_data_log m
        WHERE m.side_id = bs.side_id AND m.field_name = 'troop_total'
          AND m.notes LIKE :prefix || '%'
      )
    """
)

_LOG_WITHHELD_CASUALTIES = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, notes)
    SELECT bs.battle_id, bs.side_id, 'casualties', 'unclassified', :notes
    FROM battle_sides bs
    WHERE bs.side_id = ANY(CAST(:side_ids AS INT[])) AND bs.est_casualties IS NULL
      AND NOT EXISTS (
        SELECT 1 FROM missing_data_log m
        WHERE m.side_id = bs.side_id AND m.field_name = 'casualties'
          AND m.notes LIKE :prefix || '%'
      )
    """
)

# An estimate written by a later run supersedes an earlier "estimate withheld"
# row for the same side and quantity; left behind it would contradict the value.
_DELETE_STALE_WITHHELD = text(
    """
    DELETE FROM missing_data_log
    WHERE side_id = ANY(CAST(:side_ids AS INT[]))
      AND field_name = :field_name
      AND notes LIKE :prefix || '%'
    """
)

# Classify runs after reconcile but can see a side before reconcile has re-run
# (extract recreated it), and logs it as having no estimate. Once reconcile writes
# one, that row contradicts the value and impute would still target it. Only
# classify's own un-imputed gap rows go; every other note is another stage's.
_DELETE_STALE_CLASSIFY_GAP = text(
    """
    DELETE FROM missing_data_log
    WHERE side_id = ANY(CAST(:side_ids AS INT[]))
      AND field_name = 'troop_total'
      AND was_imputed = FALSE
      AND notes LIKE :prefix || '%'
    """
)

_FIELD_BY_QUANTITY: Final[dict[Quantity, str]] = {
    "troops": "troop_total",
    "casualties": "casualties",
}

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
    for quantity in {e.quantity for e in estimates}:
        conn.execute(
            _DELETE_STALE_WITHHELD,
            {
                "side_ids": sorted(e.side_id for e in estimates if e.quantity == quantity),
                "field_name": _FIELD_BY_QUANTITY[quantity],
                "prefix": WITHHELD_NOTE_PREFIX,
            },
        )
    troop_side_ids = sorted(e.side_id for e in estimates if e.quantity == "troops")
    if troop_side_ids:
        deleted = conn.execute(
            _DELETE_STALE_CLASSIFY_GAP,
            {"side_ids": troop_side_ids, "prefix": NO_ESTIMATE_NOTE_PREFIX},
        ).rowcount
        logger.info("reconcile_stale_classify_gap_deleted", rows=int(deleted or 0))
    return written


def clear_unmodelled(
    conn: Any, quantity: Quantity, *, side_ids: set[int], source_ids: set[int]
) -> tuple[int, int]:
    """Null the fitted values of sides and sources that are not in this fit.

    A re-run that newly excludes a side's rows (a plausibility rule, say) would
    otherwise leave the estimate an earlier fit gave it, stamped with an old run
    id, indistinguishable from a current one. Run only after the fit was
    written, so an unconverged run leaves the previous estimates standing.

    Args:
        conn: An open database connection.
        quantity: ``"troops"`` or ``"casualties"``.
        side_ids: Every side in this quantity's design.
        source_ids: Every ``source_id`` with a row in this quantity's design.

    Returns:
        ``(sides cleared, sources cleared)``.
    """
    side_sql = _CLEAR_SIDE_TROOPS if quantity == "troops" else _CLEAR_SIDE_CASUALTIES
    source_sql = _CLEAR_SOURCE_TROOPS if quantity == "troops" else _CLEAR_SOURCE_CASUALTIES
    sides = conn.execute(side_sql, {"keep": sorted(side_ids)}).rowcount
    sources = conn.execute(source_sql, {"keep": sorted(source_ids)}).rowcount
    logger.info("reconcile_stale_cleared", quantity=quantity, sides=sides, sources=sources)
    return int(sides), int(sources)


def log_withheld_sides(
    conn: Any,
    quantity: Quantity,
    *,
    side_ids: set[int],
    run_id: int,
    reason: str,
) -> int:
    """Log the sides of an unwritten fit that are left without an estimate.

    Args:
        conn: An open database connection.
        quantity: ``"troops"`` or ``"casualties"``.
        side_ids: Every side in the unwritten fit's design.
        run_id: The run whose fit was withheld, named in the log notes.
        reason: Short human-readable cause, e.g. which threshold failed.

    Returns:
        How many sides were logged.
    """
    statement = _LOG_WITHHELD_TROOPS if quantity == "troops" else _LOG_WITHHELD_CASUALTIES
    notes = f"{WITHHELD_NOTE_PREFIX}, fit did not meet write thresholds (run {run_id}): {reason}"
    logged = int(
        conn.execute(
            statement,
            {"side_ids": sorted(side_ids), "notes": notes, "prefix": WITHHELD_NOTE_PREFIX},
        ).rowcount
    )
    logger.warning(
        "reconcile_withheld_sides_logged", quantity=quantity, run_id=run_id, sides=logged
    )
    return logged


def run_fingerprint(conn: Any, run_id: int) -> tuple[str, str]:
    """Identify a ``model_runs`` row across databases.

    Run ids are per database, so the id alone cannot tell a file from this
    database's run from one written by another database that also reached it.

    Args:
        conn: An open database connection.
        run_id: The ``model_runs.run_id``.

    Returns:
        ``(current_database(), started_at as a UTC ISO string)``.

    Raises:
        LookupError: If the run does not exist.
    """
    row = conn.execute(_RUN_FINGERPRINT, {"run_id": run_id}).one_or_none()
    if row is None or row[1] is None:
        raise LookupError(f"model_runs row {run_id} has no started_at")
    return str(row[0]), row[1].astimezone(UTC).isoformat()


def clear_unmodelled_neutral(conn: Any, *, side_ids: set[int]) -> int:
    """Null the neutral troop estimate of sides that are not in the neutral fit.

    Args:
        conn: An open database connection.
        side_ids: Every side in the neutral fit's design.

    Returns:
        How many sides were cleared.
    """
    return int(conn.execute(_CLEAR_SIDE_NEUTRAL, {"keep": sorted(side_ids)}).rowcount)


def write_neutral_estimates(conn: Any, estimates: list[NeutralSideEstimate]) -> int:
    """Write the regime-neutral troop estimates onto ``battle_sides``.

    Args:
        conn: An open database connection.
        estimates: The neutral fit's per-side estimates.

    Returns:
        How many rows were written.
    """
    for estimate in estimates:
        conn.execute(
            _UPDATE_SIDE_NEUTRAL,
            {
                "side_id": estimate.side_id,
                "value": estimate.value,
                "lo": estimate.lo,
                "hi": estimate.hi,
                "log_mean": estimate.log_mean,
                "log_sd": estimate.log_sd,
                "run_id": estimate.run_id,
            },
        )
    return len(estimates)


def write_source_biases(conn: Any, biases: list[SourceBias]) -> int:
    """Write one quantity's fitted source biases onto ``sources``.

    A source with no reports of this quantity never appears in ``biases``, so
    it is never touched here; :func:`clear_unmodelled` nulls the ones an earlier
    fit had written. A ``bias_mu`` of None is stored as NULL: the key is pooled
    and its bias is identically zero, which is not a measurement. Such a row is
    recognisable as ``bias_n_*_reports > 0`` with a NULL bias.

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
