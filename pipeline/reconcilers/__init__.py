"""
Source-disagreement reconciliation for the reconcile stage.

The stage turns conflicting troop and casualty reports into single estimates
with credible intervals, by modelling *why* sources disagree rather than
averaging them away. Each report's log-scale bias decomposes into a
source_type mean, a per-source deviation, and a claim-regime term -- and this
package builds the deterministic, non-statistical inputs that decomposition
needs before any model is fit: what document a source really is
(:mod:`pipeline.reconcilers.records`), and what kind of claim its text makes
(:mod:`pipeline.reconcilers.claims`).

Nothing here touches a database, numpy, or PyMC. That is deliberate: the
model-building and model-fitting code that consumes these records is later
work, and keeping this package pure is what lets it be tested on the cases
that matter -- a Wikidata row echoing a Wikipedia infobox, a chronicle naming
no ancient source by name -- without a live Postgres.

Typical use::

    from pipeline.reconcilers import Report, assign_lineages, classify_regime

    regime = classify_regime(report_text)
    lineages = assign_lineages(reports)
"""

from __future__ import annotations

from pipeline.reconcilers.claims import assign_lineages, classify_regime, era_flag, roundness
from pipeline.reconcilers.load import (
    UnfillableSide,
    classify_sides,
    load_reports,
    write_unfillable,
)
from pipeline.reconcilers.records import (
    CLAIM_REGIMES,
    ReconcileCounts,
    Report,
    SideEstimate,
    SourceBias,
    SourceKey,
)

__all__ = [
    "CLAIM_REGIMES",
    "ReconcileCounts",
    "Report",
    "SideEstimate",
    "SourceBias",
    "SourceKey",
    "UnfillableSide",
    "assign_lineages",
    "classify_regime",
    "classify_sides",
    "era_flag",
    "load_reports",
    "roundness",
    "write_unfillable",
]
