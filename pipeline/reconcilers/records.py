"""
The records the reconcile stage passes between its steps.

The stage turns conflicting troop and casualty reports into single estimates
with credible intervals, by fitting a hierarchical model of *why* sources
disagree rather than averaging them. Every deterministic step upstream of
that model -- grouping reports by document identity, classifying the kind of
claim a report makes, scoring how coarse a number looks -- needs the same
shapes to hand off, which is what this module defines.

Nothing here fits anything. In particular a :class:`Report` carrying
``claim_regime="unlabelled"`` is not a claim that the report's provenance is
unknowable, only that :func:`pipeline.reconcilers.claims.classify_regime`
found no textual marker for it; most reports never carry one; and that is
what makes the anchor decision below load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

from pipeline.extractors.records import BRANCHES, CASUALTY_TYPES, SCOPES

__all__ = [
    "CLAIM_REGIMES",
    "ReconcileCounts",
    "Report",
    "SideEstimate",
    "SourceBias",
    "SourceKey",
]

Quantity = Literal["troops", "casualties"]

# The vocabulary a report's surrounding text can be classified into. Order
# here is also the priority order classify_regime checks in, so a string
# naming both a modern estimate and an ancient source classifies as the
# former -- ambiguous provenance in one string is rarer than deliberate,
# side-by-side contrast ("52,930 (modern estimates) 250,000 (ancient
# sources)"), and the model needs one label per report, not a set.
#
# "modern_scholarly" is the model's anchor: it is pinned to exactly zero bias
# in the reconcile model, not fitted like every other regime. Without a fixed
# reference point, "bias" has no meaning -- a source can only be biased
# relative to something, and pinning the regime a modern secondary source
# writes in is what lets an estimate mean "what a modern scholar would say"
# rather than "the average of whatever this corpus happened to collect".
CLAIM_REGIMES: Final[tuple[str, ...]] = (
    "modern_scholarly",
    "ancient_claim",
    "chronicle",
    "administrative_partisan",
    "staff_return",
    "unlabelled",
)


@dataclass(frozen=True)
class SourceKey:
    """The canonical identity of a source *document*.

    ``sources`` has no unique constraint, and ``pipeline/extractors/store.py``
    does SELECT-then-INSERT rather than upsert, so one physical document (one
    URL, one book) can legitimately occupy several ``source_id`` rows -- once
    per extraction pass that happened to race the lookup, or that ran before
    and after a title changed. Indexing the reconcile model's per-source bias
    parameter by ``source_id`` would then split one document's evidence
    across two parameters that each see fewer reports and so each shrink
    harder toward the population mean. That understates the bias of exactly
    the sources that appear most often in the corpus -- the ones with the
    most duplicate rows to begin with -- which is the opposite of what a
    bias model is for. Grouping by ``(source_type, url)`` instead is stable
    across however many ``source_id`` rows the same document was inserted as.

    Attributes:
        source_type: A ``source_type`` enum value, e.g. ``"wikipedia_infobox"``.
        url: The document's URL, or ``""`` when the source has none (a book
            cited by title and author only, say). Never ``None``, so a
            :class:`SourceKey` can be hashed and used as a mapping key without
            a null check at every call site.
    """

    source_type: str
    url: str

    @classmethod
    def from_row(cls, source_type: str, url: str | None) -> SourceKey:
        """Build a key from a ``sources`` row, normalising a NULL url.

        Args:
            source_type: The row's ``source_type`` column.
            url: The row's ``url`` column, possibly NULL.

        Returns:
            A :class:`SourceKey` with ``url`` normalised to ``""`` when the
            row's url is NULL or empty.
        """
        return cls(source_type=source_type, url=url or "")


@dataclass(frozen=True)
class Report:
    """One source's statement of one quantity for one battle side.

    A row of ``troop_reports`` or ``casualty_reports``, joined against
    ``sources`` for the fields the reconcile model needs about where the
    number came from. ``claim_regime``, ``lineage_id`` and ``roundness`` are
    not columns on either table; they are computed by this package's
    functions and carried alongside the raw report so the model-building
    step can read one record instead of re-deriving them.

    Attributes:
        report_id: The ``troop_reports.report_id`` or
            ``casualty_reports.report_id`` this was built from. Unique within
            one quantity, and used as the key for lineage assignment.
        side_id: The ``battle_sides.side_id`` this report is about.
        battle_id: The battle the side belongs to, denormalised here because
            grouping and diagnostics both want it without a join.
        source_id: The ``sources.source_id`` FK on the report row.
        source_key: This source's document identity; see :class:`SourceKey`
            for why it is not just ``source_id``.
        source_type: A ``source_type`` enum value, denormalised from
            ``sources`` for the same reason as ``battle_id``.
        quantity: ``"troops"`` or ``"casualties"`` -- which table this came
            from, since the two are modelled on different report subsets.
        branch: A ``troop_branch`` enum value. Meaningless for a casualty
            report, where it stays at its default.
        casualty_type: A ``casualty_type`` value from
            :data:`pipeline.extractors.records.CASUALTY_TYPES`. Meaningless
            for a troop report, where it stays at its default.
        reported_value: The number the source gives, unmodified.
        scope: A ``scope`` value from
            :data:`pipeline.extractors.records.SCOPES` -- whether this counts
            troops engaged, available, on paper, or the scope could not be
            read. Meaningless for a casualty report.
        is_estimate: Whether the source itself hedges the number.
        is_upper_bound: Whether the source says "up to" this value.
        is_lower_bound: Whether the source says "at least" this value.
        extracted_context: The sentence or field the number came from, which
            is what :func:`pipeline.reconcilers.claims.classify_regime` reads.
        year_astronomical: The battle's astronomical year, or None when
            undated. See :func:`pipeline.reconcilers.claims.era_flag` for why
            this must never be defaulted to 0.
        claim_regime: A member of :data:`CLAIM_REGIMES`, defaulting to
            ``"unlabelled"`` until classified.
        lineage_id: Which group of reports repeating one underlying claim this
            belongs to, from
            :func:`pipeline.reconcilers.claims.assign_lineages`, or None
            before lineage assignment has run.
        roundness: How conventional this figure looks, from
            :func:`pipeline.reconcilers.claims.roundness`, or its default
            ``0.0`` before that has run.
    """

    report_id: int
    side_id: int
    battle_id: int
    source_id: int
    source_key: SourceKey
    source_type: str
    quantity: Quantity
    reported_value: float
    branch: str = "total"
    casualty_type: str = "total"
    scope: str = "unknown"
    is_estimate: bool = False
    is_upper_bound: bool = False
    is_lower_bound: bool = False
    extracted_context: str = ""
    year_astronomical: int | None = None
    claim_regime: str = "unlabelled"
    lineage_id: int | None = None
    roundness: float = 0.0

    def __post_init__(self) -> None:
        """Validate every vocabulary field at construction, once, at the boundary.

        A :class:`Report` is built once per row and read many times
        downstream by grouping and modelling code that has no reason to
        re-check its inputs. Reusing
        :data:`pipeline.extractors.records.BRANCHES`,
        :data:`~pipeline.extractors.records.CASUALTY_TYPES` and
        :data:`~pipeline.extractors.records.SCOPES` here, rather than
        re-declaring the same vocabularies, is what keeps this check from
        silently drifting out of sync with what the extract stage actually
        writes.

        Raises:
            ValueError: If any vocabulary field holds a value outside its
                allowed set.
        """
        if self.quantity not in ("troops", "casualties"):
            raise ValueError(f"quantity must be 'troops' or 'casualties', got {self.quantity!r}")
        if self.branch not in BRANCHES:
            raise ValueError(f"branch {self.branch!r} is not in extractors.records.BRANCHES")
        if self.casualty_type not in CASUALTY_TYPES:
            raise ValueError(
                f"casualty_type {self.casualty_type!r} is not in extractors.records.CASUALTY_TYPES"
            )
        if self.scope not in SCOPES:
            raise ValueError(f"scope {self.scope!r} is not in extractors.records.SCOPES")
        if self.claim_regime not in CLAIM_REGIMES:
            raise ValueError(f"claim_regime {self.claim_regime!r} is not in CLAIM_REGIMES")


@dataclass
class ReconcileCounts:
    """Row and decision counts for one reconcile run, for the summary log.

    Mutable, like :class:`pipeline.resolvers.records.ResolveCounts`: a run
    accumulates these one report and one side at a time, and there is no
    later step that needs an immutable snapshot of a partial count.
    """

    reports_read: int = 0
    troop_reports_used: int = 0
    casualty_reports_used: int = 0
    excluded_non_total_branch: int = 0
    excluded_naval: int = 0
    excluded_non_positive: int = 0
    censored_upper: int = 0
    censored_lower: int = 0
    both_bounds: int = 0
    sides_estimated: int = 0
    sides_unfillable: int = 0
    sources_updated: int = 0
    lineages: int = 0
    regime_unlabelled: int = 0

    def as_dict(self) -> dict[str, int]:
        """Render the counts for a structlog call.

        Returns:
            A flat mapping of counter name to value.
        """
        return {
            "reports_read": self.reports_read,
            "troop_reports_used": self.troop_reports_used,
            "casualty_reports_used": self.casualty_reports_used,
            "excluded_non_total_branch": self.excluded_non_total_branch,
            "excluded_naval": self.excluded_naval,
            "excluded_non_positive": self.excluded_non_positive,
            "censored_upper": self.censored_upper,
            "censored_lower": self.censored_lower,
            "both_bounds": self.both_bounds,
            "sides_estimated": self.sides_estimated,
            "sides_unfillable": self.sides_unfillable,
            "sources_updated": self.sources_updated,
            "lineages": self.lineages,
            "regime_unlabelled": self.regime_unlabelled,
        }


@dataclass(frozen=True)
class SideEstimate:
    """One battle side's reconciled troop or casualty total.

    Mirrors the ``est_<quantity>_*`` column family on ``battle_sides``. Troops
    and casualties are estimated independently, from different report
    subsets, so one record per quantity per side keeps that separation
    instead of forcing a false pairing into a single row.

    Attributes:
        side_id: The ``battle_sides.side_id`` this estimate belongs to.
        quantity: ``"troops"`` or ``"casualties"``.
        value: The point estimate, or None when the side could not be
            estimated at all (``ReconcileCounts.sides_unfillable``).
        lo: The 95% credible interval lower bound, or None alongside `value`.
        hi: The 95% credible interval upper bound, or None alongside `value`.
        run_id: The ``model_runs.run_id`` that produced this estimate.
        n_reports: How many reports fed the estimate.
        n_sources: How many distinct :class:`SourceKey` values fed it -- not
            a count of ``source_id`` rows; see that class for why.
        method: ``"source_disagreement"`` when two or more independent claim
            lineages were available, or ``"single_report_debiased"`` when
            only one was, and the interval is necessarily wide.
        updated_at: When this estimate was computed.
    """

    side_id: int
    quantity: Quantity
    value: float | None
    lo: float | None
    hi: float | None
    run_id: int
    n_reports: int
    n_sources: int
    method: str
    updated_at: datetime


@dataclass(frozen=True)
class SourceBias:
    """One source's fitted log-scale bias for troop or casualty reports.

    Mirrors the ``<quantity>_bias_mu/sd`` and ``<quantity>_sigma_mu/sd``
    column family on ``sources``.

    Attributes:
        source_id: The ``sources.source_id`` row this bias is written to. A
            single :class:`SourceKey` can map to several ``source_id`` rows,
            so the caller applies one fitted bias to every row sharing its
            key rather than fitting one per row.
        quantity: ``"troops"`` or ``"casualties"``.
        bias_mu: Posterior mean log-scale bias, relative to the
            ``modern_scholarly`` anchor.
        bias_sd: Posterior SD of that bias.
        sigma_mu: Posterior mean log-scale observation SD, or None when not
            estimated for this quantity.
        sigma_sd: Posterior SD of that, or None alongside `sigma_mu`.
        run_id: The ``model_runs.run_id`` that produced this fit.
        n_reports: How many reports of this quantity this source contributed.
            ``0`` means the bias columns still hold the prior, untouched, and
            must stay distinguishable from a fitted ``0.0``.
        updated_at: When this bias was computed.
    """

    source_id: int
    quantity: Quantity
    bias_mu: float
    bias_sd: float
    sigma_mu: float | None
    sigma_sd: float | None
    run_id: int
    n_reports: int
    updated_at: datetime
