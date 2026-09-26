"""Tests for the reconcile design matrix.

The censoring test is the one to keep. ``pm.Censored`` is parameterised by
what it clamps, not by what the source said, so "up to X" becomes a *lower*
bound argument. An implementation that maps the flags straight through samples
cleanly, reports rhat 1.00 and emits a plausible ranking, and nothing
downstream would notice. Only a test that names the direction catches it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pipeline.reconcilers.design import (
    CASUALTY_TYPE_ORDER,
    SCOPE_ORDER,
    Design,
    build_design,
    vocabularies_agree,
)
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _report(
    report_id: int,
    *,
    side_id: int = 1,
    battle_id: int = 1,
    value: float = 10_000.0,
    quantity: str = "troops",
    branch: str = "total",
    casualty_type: str = "total",
    scope: str = "engaged",
    source_type: str = "wikipedia_infobox",
    url: str = "https://example/a",
    year: int | None = -330,
    regime: str = "unlabelled",
    lineage_id: int = 0,
    is_upper_bound: bool = False,
    is_lower_bound: bool = False,
    is_estimate: bool = False,
    roundness: float = 0.0,
) -> Report:
    """Build a Report with everything but the field under test defaulted."""
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=report_id,
        source_key=SourceKey(source_type, url),
        source_type=source_type,
        quantity=quantity,
        branch=branch,
        casualty_type=casualty_type,
        reported_value=value,
        scope=scope,
        is_estimate=is_estimate,
        is_upper_bound=is_upper_bound,
        is_lower_bound=is_lower_bound,
        extracted_context="",
        year_astronomical=year,
        claim_regime=regime,
        lineage_id=lineage_id,
        roundness=roundness,
    )


def _design(
    reports: list[Report],
    *,
    quantity: str = "troops",
    ancient_cutoff_year: int = 500,
    counts: ReconcileCounts | None = None,
) -> Design:
    """Build a design with the usual defaults."""
    return build_design(
        reports,
        quantity=quantity,
        ancient_cutoff_year=ancient_cutoff_year,
        counts=counts,
    )


# ─── Censoring: the direction that matters ───────────────────────────────────


def test_an_upper_bound_report_becomes_a_lower_censoring_threshold() -> None:
    # "up to 5,000" means the latent value lies at or below 5,000. pm.Censored
    # clamps, so an observation at `lower` contributes log CDF(lower), the mass
    # below it. Mapping is_upper_bound straight to `upper` asserts the opposite
    # and still samples perfectly.
    design = _design([_report(1, value=5_000.0, is_upper_bound=True)])

    assert design.censor_lower_at.tolist() == [0]
    assert design.censor_upper_at.tolist() == []
    assert design.point_rows.tolist() == []


def test_a_lower_bound_report_becomes_an_upper_censoring_threshold() -> None:
    design = _design([_report(1, value=5_000.0, is_lower_bound=True)])

    assert design.censor_upper_at.tolist() == [0]
    assert design.censor_lower_at.tolist() == []


def test_a_report_flagged_as_both_bounds_is_treated_as_a_point_observation() -> None:
    # An interval whose ends the row does not distinguish is not a censoring
    # this model can express, so it degrades to a point and is counted.
    counts = ReconcileCounts()
    design = _design([_report(1, is_upper_bound=True, is_lower_bound=True)], counts=counts)

    assert design.point_rows.tolist() == [0]
    assert design.censor_lower_at.size == 0
    assert design.censor_upper_at.size == 0
    assert counts.both_bounds == 1


def test_the_censoring_counters_are_named_by_the_flag_not_the_pymc_argument() -> None:
    # Diagnostics are read by historians, not by samplers: an is_upper_bound
    # row should show up as censored_upper however pm.Censored is parameterised.
    counts = ReconcileCounts()
    _design(
        [
            _report(1, is_upper_bound=True),
            _report(2, is_lower_bound=True),
            _report(3, is_lower_bound=True),
        ],
        counts=counts,
    )

    assert counts.censored_upper == 1
    assert counts.censored_lower == 2


# ─── The era covariate ───────────────────────────────────────────────────────


def test_the_era_covariate_is_centred_so_its_column_sums_to_zero() -> None:
    design = _design(
        [
            _report(1, year=-330),
            _report(2, year=-330),
            _report(3, year=1815),
            _report(4, year=1815),
        ]
    )

    assert design.era.sum() == pytest.approx(0.0)
    assert design.era_mean == pytest.approx(0.5)


def test_an_all_ancient_corpus_has_an_identically_zero_era_covariate() -> None:
    # Uncentred this is a column of ones: a second intercept confounded with
    # the corpus mean, which funnels or shifts every estimate by a constant.
    # Centred it is exactly zero, the honest answer when the data carry no era
    # contrast at all.
    design = _design([_report(i, year=-330) for i in range(1, 6)])

    assert np.allclose(design.era, 0.0)
    assert design.era_mean == pytest.approx(1.0)


def test_a_battle_with_a_null_year_sits_at_the_corpus_average_not_at_modern() -> None:
    # A missing date is not evidence that a battle is recent.
    design = _design([_report(1, year=-330), _report(2, year=1815), _report(3, year=None)])

    assert design.era[2] == pytest.approx(0.0)
    assert design.era[0] > 0.0
    assert design.era[1] < 0.0


def test_an_undated_battle_does_not_drag_the_era_mean() -> None:
    # The mean is taken over rows carrying a year, so undated rows must not
    # move the centring for the rows that have one.
    dated = _design([_report(1, year=-330), _report(2, year=1815)])
    mixed = _design([_report(1, year=-330), _report(2, year=1815), _report(3, year=None)])

    assert mixed.era_mean == pytest.approx(dated.era_mean)


# ─── Reference levels ────────────────────────────────────────────────────────


def test_engaged_is_the_scope_reference_and_carries_index_zero() -> None:
    # The model pins level 0's offset to exactly zero. If a corpus with no
    # 'engaged' rows silently promoted 'on_paper' to index 0, that pinned
    # offset would apply to the wrong level and every estimate would shift.
    design = _design([_report(1, scope="on_paper"), _report(2, scope="theatre_strength")])

    assert design.levels[0] == "engaged"
    assert SCOPE_ORDER[0] == "engaged"


def test_total_is_the_casualty_reference_and_carries_index_zero() -> None:
    design = _design(
        [
            _report(1, quantity="casualties", casualty_type="killed", value=900.0),
            _report(2, quantity="casualties", casualty_type="wounded", value=300.0),
        ],
        quantity="casualties",
    )

    assert design.levels[0] == "total"
    assert CASUALTY_TYPE_ORDER[0] == "total"


def test_the_scope_levels_keep_their_containment_order() -> None:
    design = _design(
        [
            _report(1, scope="on_paper"),
            _report(2, scope="engaged"),
            _report(3, scope="available"),
        ]
    )

    assert design.levels == ("engaged", "available", "on_paper")


# ─── Source keys and lineages ────────────────────────────────────────────────


def test_two_source_rows_sharing_a_type_and_url_collapse_to_one_bias_key() -> None:
    # sources has no unique constraint, so one document can hold several
    # source_ids. Indexing bias by source_id splits its evidence across two
    # parameters that each shrink further toward zero.
    design = _design(
        [_report(1, url="https://example/x"), _report(2, url="https://example/x")]
    )

    assert design.n_sources == 1
    assert design.source_index.tolist() == [0, 0]


def test_reports_repeating_one_claim_share_a_lineage_index() -> None:
    design = _design(
        [
            _report(1, url="https://example/x", lineage_id=7),
            _report(2, url="https://example/y", source_type="wikidata", lineage_id=7),
            _report(3, url="https://example/z", source_type="peer_reviewed", lineage_id=9),
        ]
    )

    assert design.lineage_index.tolist() == [0, 0, 1]


# ─── Zero casualties ─────────────────────────────────────────────────────────


def test_a_zero_casualty_row_is_censored_rather_than_dropped_or_log1p_shifted() -> None:
    # Zero casualties is a fact. log1p on every row would move the scale of
    # every other observation to accommodate a handful of zeros.
    design = _design(
        [
            _report(1, quantity="casualties", value=0.0),
            _report(2, quantity="casualties", value=400.0),
        ],
        quantity="casualties",
    )

    assert design.zero_rows.tolist() == [0]
    assert design.log_y[0] == pytest.approx(0.0)
    assert design.point_rows.tolist() == [1]


# ─── Shape and guards ────────────────────────────────────────────────────────


def test_every_index_vector_is_the_same_length_as_the_observation_vector() -> None:
    reports = [_report(i, side_id=i % 3, value=1000.0 * i) for i in range(1, 8)]
    design = _design(reports)

    for name in (
        "side_index",
        "source_index",
        "source_type_index",
        "regime_index",
        "level_index",
        "lineage_index",
        "era",
        "roundness",
        "is_estimate",
    ):
        column: Any = getattr(design, name)
        assert len(column) == design.n_obs, name


def test_the_partitions_are_disjoint_and_cover_every_row() -> None:
    reports = [
        _report(1),
        _report(2, is_upper_bound=True),
        _report(3, is_lower_bound=True),
        _report(4, is_upper_bound=True, is_lower_bound=True),
    ]
    design = _design(reports)

    covered = sorted(
        [
            *design.point_rows.tolist(),
            *design.censor_lower_at.tolist(),
            *design.censor_upper_at.tolist(),
            *design.zero_rows.tolist(),
        ]
    )
    assert covered == list(range(len(reports)))


def test_a_quantity_with_no_reports_raises_rather_than_fitting_the_priors() -> None:
    # An empty design samples its priors and reports clean convergence, which
    # looks identical to success.
    with pytest.raises(ValueError, match="No usable casualties reports"):
        build_design([_report(1)], quantity="casualties", ancient_cutoff_year=500)


def test_the_level_orderings_stay_subsets_of_the_extractor_vocabularies() -> None:
    assert vocabularies_agree()
