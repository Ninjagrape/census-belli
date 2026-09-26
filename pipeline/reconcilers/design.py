"""Turn reports into the index vectors and censoring partitions a fit needs.

Pure numpy. No database, no pymc, so this module is testable wherever the
package imports, and the one piece of arithmetic that is easy to get silently
wrong lives here alone rather than being spelled out again inside the model.

That piece is the censoring. ``troop_reports`` records a stated bound with
``is_upper_bound`` ("up to 5,000") and ``is_lower_bound`` ("at least 5,000"),
and ``pm.Censored`` is parameterised the other way round: it *clamps*, so an
observation sitting at ``lower`` contributes ``log CDF(lower)``, the mass of
everything at or below it. A source saying "up to X" is therefore expressed as
``lower=log X``, and "at least X" as ``upper=log X``.

Getting that backwards samples cleanly, reports rhat 1.00, and emits a
plausible ranking, so nothing downstream would catch it. The partitions are
consequently named for **what PyMC does with them** -- ``censor_lower_at``,
``censor_upper_at`` -- rather than for the flag that produced them, and the
translation happens once, here, under a test that pins the direction.

Two further decisions worth knowing before changing anything:

**The era covariate is centred.** In a heavily ancient corpus an uncentred
ancient/modern indicator is a column of mostly ones, which makes the era term
a second intercept confounded with the corpus mean: the sampler funnels, or
every estimate shifts by a constant nobody can see. Centring costs nothing and
removes the failure. A corpus that is entirely ancient degenerates gracefully
to an identically zero column, which is the correct answer -- with no contrast
in the data there is nothing for an era term to estimate.

**A report with no readable era sits at the corpus average**, not at modern.
``era_flag`` returns None for an undated battle and that None becomes 0.0
after centring, meaning no evidence either way, rather than a positive claim
that the battle is recent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import structlog

from pipeline.extractors.records import CASUALTY_TYPES, SCOPES
from pipeline.reconcilers.claims import era_flag
from pipeline.reconcilers.records import CLAIM_REGIMES, ReconcileCounts, Report, SourceKey

__all__ = [
    "CASUALTY_TYPE_ORDER",
    "SCOPE_ORDER",
    "Design",
    "build_design",
]

logger = structlog.get_logger()

# 'engaged' is the reference level and must be index 0: the model pins its
# offset to exactly zero and builds the others as non-negative increments from
# it, which encodes the containment hierarchy (engaged within available within
# a theatre roster) instead of hoping the data discovers it.
SCOPE_ORDER: Final[tuple[str, ...]] = (
    "engaged",
    "available",
    "theatre_strength",
    "on_paper",
    "unknown",
)

# 'total' is the reference level for the same reason, and must be index 0.
CASUALTY_TYPE_ORDER: Final[tuple[str, ...]] = (
    "total",
    "killed",
    "wounded",
    "captured",
    "missing",
)

_TROOPS: Final[str] = "troops"


@dataclass(frozen=True)
class Design:
    """Everything a fit needs about one quantity, as aligned arrays.

    Every per-observation array has length ``n_obs`` and is aligned with every
    other, so a model can index them together without re-deriving anything.

    Attributes:
        quantity: ``"troops"`` or ``"casualties"``.
        log_y: Natural log of each reported value.
        side_index: Row's side, as an index into ``side_ids``.
        source_index: Row's source *key*, as an index into ``source_keys``.
        source_type_index: Row's source type, into ``source_types``.
        regime_index: Row's claim regime, into ``regimes``.
        level_index: Row's scope (troops) or casualty type (casualties), into
            ``levels``. Index 0 is the reference level in both cases.
        lineage_index: Row's claim lineage, into ``lineage_ids``. Reports
            repeating one claim share an index so they contribute roughly one
            observation's worth of precision between them rather than one
            each.
        era: The centred era covariate. Zero means "no information", which is
            what both an undated battle and a corpus with no era contrast
            should contribute.
        era_mean: The report-weighted fraction of ancient reports that was
            subtracted. Recorded so a run's anchoring is reconstructable.
        roundness: 0..1 per row; conventional figures get more observation
            variance because they are coarsened, not merely noisy.
        is_estimate: Whether the source hedged the figure.
        point_rows: Indices of rows observed exactly, with no stated bound.
        censor_lower_at: Indices whose latent value lies **below** the stated
            figure, from ``is_upper_bound``. Named for the ``pm.Censored``
            argument they populate; see the module docstring.
        censor_upper_at: Indices whose latent value lies **above** the stated
            figure, from ``is_lower_bound``.
        zero_rows: Casualty rows reporting exactly zero. Kept, because zero
            casualties is a fact rather than a missing value, and handled as a
            censoring rather than by shifting every other row's scale with a
            log1p.
        side_ids: Labels ``side_index`` points into.
        source_keys: Labels ``source_index`` points into.
        source_types: Labels ``source_type_index`` points into.
        regimes: Labels ``regime_index`` points into.
        levels: Labels ``level_index`` points into.
        lineage_ids: Labels ``lineage_index`` points into.
    """

    quantity: str
    log_y: np.ndarray
    side_index: np.ndarray
    source_index: np.ndarray
    source_type_index: np.ndarray
    regime_index: np.ndarray
    level_index: np.ndarray
    lineage_index: np.ndarray
    era: np.ndarray
    era_mean: float
    roundness: np.ndarray
    is_estimate: np.ndarray
    point_rows: np.ndarray
    censor_lower_at: np.ndarray
    censor_upper_at: np.ndarray
    zero_rows: np.ndarray
    side_ids: tuple[int, ...]
    source_keys: tuple[SourceKey, ...]
    source_types: tuple[str, ...]
    regimes: tuple[str, ...]
    levels: tuple[str, ...]
    lineage_ids: tuple[int, ...]

    @property
    def n_obs(self) -> int:
        """Number of observations in this design."""
        return int(self.log_y.shape[0])

    @property
    def n_sides(self) -> int:
        """Number of distinct sides."""
        return len(self.side_ids)

    @property
    def n_sources(self) -> int:
        """Number of distinct source keys."""
        return len(self.source_keys)

    def coords(self) -> dict[str, list[str]]:
        """Render the coordinate labels for a PyMC model.

        Returns:
            A mapping of dimension name to label list, ready to pass as a
            model's ``coords``.
        """
        return {
            "side": [str(s) for s in self.side_ids],
            "source_key": [f"{k.source_type}|{k.url}" for k in self.source_keys],
            "source_type": list(self.source_types),
            "regime": list(self.regimes),
            "level": list(self.levels),
            "lineage": [str(x) for x in self.lineage_ids],
        }


def _index_map(values: list[Any], order: tuple[str, ...] | None = None) -> dict[Any, int]:
    """Build a stable label-to-index mapping.

    Args:
        values: The labels appearing in the data.
        order: A fixed ordering to honour, for vocabularies whose index 0 is a
            reference level the model pins to zero.

    Returns:
        A mapping from label to index.
    """
    if order is not None:
        present = [v for v in order if v in set(values)]
        # The reference level must exist even when no row uses it, or index 0
        # silently becomes some other level and the pinned-to-zero offset
        # would apply to the wrong thing.
        if order[0] not in present:
            present = [order[0], *present]
        return {label: i for i, label in enumerate(present)}
    return {label: i for i, label in enumerate(sorted(set(values), key=str))}


def build_design(
    reports: list[Report],
    *,
    quantity: str,
    ancient_cutoff_year: int,
    counts: ReconcileCounts | None = None,
) -> Design:
    """Assemble the aligned arrays for one quantity's model.

    Args:
        reports: Usable reports, already filtered and lineage-assigned by
            :mod:`pipeline.reconcilers.load`.
        quantity: ``"troops"`` or ``"casualties"``; selects which reports are
            used and whether the level vocabulary is scope or casualty type.
        ancient_cutoff_year: Astronomical year below which a battle counts as
            ancient, for the era fallback.
        counts: Optional counters to accumulate censoring tallies into.

    Returns:
        The design.

    Raises:
        ValueError: If no report matches the quantity. An empty design would
            otherwise produce a model with no observations, which samples its
            priors and reports clean convergence on no evidence.
    """
    rows = [r for r in reports if r.quantity == quantity]
    if not rows:
        raise ValueError(
            f"No usable {quantity} reports. Fitting would sample the priors and "
            "report clean convergence on no evidence."
        )

    is_troops = quantity == _TROOPS
    level_order = SCOPE_ORDER if is_troops else CASUALTY_TYPE_ORDER

    def level_of(report: Report) -> str:
        return report.scope if is_troops else report.casualty_type

    side_ix = _index_map([r.side_id for r in rows])
    key_ix = _index_map([f"{r.source_key.source_type}|{r.source_key.url}" for r in rows])
    type_ix = _index_map([r.source_type for r in rows])
    regime_ix = _index_map([r.claim_regime for r in rows], order=CLAIM_REGIMES)
    level_ix = _index_map([level_of(r) for r in rows], order=level_order)
    lineage_ix = _index_map([r.lineage_id for r in rows])

    # A casualty report of exactly zero is meaningful and is kept. It cannot be
    # logged, so it is held out as a censoring at log(1): "one casualty or
    # fewer". The alternative, log1p on every row, would shift the scale of
    # every other observation to accommodate a handful.
    raw = np.array([r.reported_value for r in rows], dtype=float)
    zero_mask = raw <= 0.0
    log_y = np.log(np.where(zero_mask, 1.0, raw))

    era_values: list[float] = []
    for report in rows:
        flag = era_flag(report.year_astronomical, ancient_cutoff_year)
        era_values.append(np.nan if flag is None else flag)
    era_raw = np.array(era_values, dtype=float)
    known = ~np.isnan(era_raw)
    era_mean = float(era_raw[known].mean()) if bool(known.any()) else 0.0
    era = np.where(known, era_raw - era_mean, 0.0)

    upper_flag = np.array([r.is_upper_bound for r in rows], dtype=bool)
    lower_flag = np.array([r.is_lower_bound for r in rows], dtype=bool)
    # A row asserting both bounds is asserting an interval whose ends it does
    # not distinguish, which is not a censoring this model can express.
    # Treated as a point observation and counted, so the compromise is visible
    # in the diagnostics rather than buried.
    both = upper_flag & lower_flag
    censor_lower = upper_flag & ~both & ~zero_mask
    censor_upper = lower_flag & ~both & ~zero_mask
    point = ~censor_lower & ~censor_upper & ~zero_mask

    if counts is not None:
        # Counted by the flag that produced them, not by the PyMC argument, so
        # the diagnostics read the way a historian would expect.
        counts.censored_upper += int(censor_lower.sum())
        counts.censored_lower += int(censor_upper.sum())
        counts.both_bounds += int(both.sum())

    design = Design(
        quantity=quantity,
        log_y=log_y,
        side_index=np.array([side_ix[r.side_id] for r in rows], dtype=int),
        source_index=np.array(
            [key_ix[f"{r.source_key.source_type}|{r.source_key.url}"] for r in rows], dtype=int
        ),
        source_type_index=np.array([type_ix[r.source_type] for r in rows], dtype=int),
        regime_index=np.array([regime_ix[r.claim_regime] for r in rows], dtype=int),
        level_index=np.array([level_ix[level_of(r)] for r in rows], dtype=int),
        lineage_index=np.array([lineage_ix[r.lineage_id] for r in rows], dtype=int),
        era=era,
        era_mean=era_mean,
        roundness=np.array([r.roundness for r in rows], dtype=float),
        is_estimate=np.array([r.is_estimate for r in rows], dtype=bool),
        point_rows=np.flatnonzero(point),
        censor_lower_at=np.flatnonzero(censor_lower),
        censor_upper_at=np.flatnonzero(censor_upper),
        zero_rows=np.flatnonzero(zero_mask),
        side_ids=tuple(sorted(side_ix, key=lambda k: side_ix[k])),
        source_keys=tuple(
            SourceKey(*str(label).split("|", 1))
            for label in sorted(key_ix, key=lambda k: key_ix[k])
        ),
        source_types=tuple(str(x) for x in sorted(type_ix, key=lambda k: type_ix[k])),
        regimes=tuple(str(x) for x in sorted(regime_ix, key=lambda k: regime_ix[k])),
        levels=tuple(str(x) for x in sorted(level_ix, key=lambda k: level_ix[k])),
        lineage_ids=tuple(sorted(lineage_ix, key=lambda k: lineage_ix[k])),
    )

    logger.info(
        "reconcile_design_built",
        quantity=quantity,
        n_obs=design.n_obs,
        n_sides=design.n_sides,
        n_sources=design.n_sources,
        n_lineages=len(design.lineage_ids),
        era_mean=round(era_mean, 4),
        censor_lower_at=int(censor_lower.sum()),
        censor_upper_at=int(censor_upper.sum()),
        both_bounds=int(both.sum()),
        zero_rows=int(zero_mask.sum()),
    )
    return design


def vocabularies_agree() -> bool:
    """Check the level orderings are subsets of the extractor's vocabularies.

    ``SCOPE_ORDER`` and ``CASUALTY_TYPE_ORDER`` fix an ordering the model
    depends on, so they are written out here rather than imported. That makes
    them able to drift from the vocabularies the extract stage actually
    writes, which is what this checks; a unit test calls it.

    Returns:
        True when both orderings are subsets of their source vocabulary.
    """
    return set(SCOPE_ORDER) <= set(SCOPES) and set(CASUALTY_TYPE_ORDER) <= set(CASUALTY_TYPES)
