"""P1 troop side prior: the battle and era-bin indices the design carries.

The side prior is ``m0 + e[era_bin] + s_b * zb[battle] + s_d * zd[side]``, so the
design must say which battle each side belongs to and which of five era bins its
battle falls in. An undated battle has no bin of its own: it takes a sixth index
that the model maps to a zero era effect.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from pipeline.reconcilers.design import (
    ERA_BIN_LABELS,
    UNDATED_ERA_BIN,
    Design,
    build_design,
    era_bin,
)
from pipeline.reconcilers.records import Report, SourceKey


def _report(report_id: int, side_id: int, battle_id: int, year: int | None) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", f"https://example/{report_id}"),
        source_type="wikipedia_infobox",
        quantity="troops",
        reported_value=5_000.0 + 10.0 * report_id,
        scope="engaged",
        year_astronomical=year,
        claim_regime="unlabelled",
        lineage_id=report_id,
    )


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (-2000, 0),
        (-330, 0),
        (0, 0),
        (499, 0),
        (500, 1),
        (1499, 1),
        (1500, 2),
        (1799, 2),
        (1800, 3),
        (1899, 3),
        (1900, 4),
        (2024, 4),
    ],
)
def test_era_bins_cut_at_500_1500_1800_and_1900(year: int, expected: int) -> None:
    assert era_bin(year) == expected


def test_an_undated_battle_gets_the_extra_index_after_the_five_bins() -> None:
    assert len(ERA_BIN_LABELS) == 5
    assert UNDATED_ERA_BIN == 5
    assert era_bin(None) == UNDATED_ERA_BIN


def _design() -> Design:
    reports = [
        _report(1, side_id=10, battle_id=1, year=-330),
        _report(2, side_id=11, battle_id=1, year=-330),
        _report(3, side_id=20, battle_id=2, year=1815),
        _report(4, side_id=21, battle_id=2, year=1815),
        _report(5, side_id=30, battle_id=3, year=None),
        _report(6, side_id=40, battle_id=4, year=1944),
    ]
    return build_design(reports, quantity="troops", ancient_cutoff_year=500)


def test_both_sides_of_a_battle_share_a_battle_index() -> None:
    design = _design()
    battle_of = dict(zip(design.side_ids, design.battle_of_side, strict=True))

    assert battle_of[10] == battle_of[11]
    assert battle_of[20] == battle_of[21]
    assert len({battle_of[10], battle_of[20], battle_of[30], battle_of[40]}) == 4
    assert design.n_battles == 4
    assert design.battle_of_side.shape == (design.n_sides,)


def test_each_side_carries_the_era_bin_of_its_battle() -> None:
    design = _design()
    bin_of = dict(zip(design.side_ids, design.era_bin_of_side, strict=True))

    assert bin_of[10] == bin_of[11] == 0  # 330 BC
    assert bin_of[20] == bin_of[21] == 3  # 1815
    assert bin_of[30] == UNDATED_ERA_BIN
    assert bin_of[40] == 4  # 1944


def test_the_design_records_each_rows_year_and_nan_for_undated() -> None:
    design = _design()
    years = {
        design.side_ids[int(design.side_index[i])]: design.year_of_row[i]
        for i in range(design.n_obs)
    }

    assert years[10] == -330.0
    assert years[20] == 1815.0
    assert math.isnan(years[30])


def test_coords_name_the_battle_and_era_bin_dimensions() -> None:
    design = _design()
    coords = design.coords()

    assert len(coords["battle"]) == design.n_battles
    assert coords["era_bin"] == list(ERA_BIN_LABELS)


def test_a_hand_built_design_defaults_to_one_battle_per_side_and_no_era() -> None:
    design = dataclasses.replace(
        _design(),
        side_battle_index=None,
        side_era_bin=None,
        battle_ids=(),
        row_year=None,
    )

    np.testing.assert_array_equal(design.battle_of_side, np.arange(design.n_sides))
    assert design.n_battles == design.n_sides
    assert (design.era_bin_of_side == UNDATED_ERA_BIN).all()
    assert np.isnan(design.year_of_row).all()
    assert len(design.coords()["battle"]) == design.n_sides
