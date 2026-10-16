"""Headline split, rank overlap and parameter validation of the report stage."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.reporters.config import ReportParams
from pipeline.reporters.disclosures import p_above_next_note
from pipeline.reporters.load import load_pooled_draws
from pipeline.reporters.records import GeneralEstimate, PooledDraws
from pipeline.reporters.tables import overlap_share, split_generals, top_headline, z_for_level
from tests.unit.reporters_helpers import estimate, view

# ─── headline versus under-determined ────────────────────────────────────────


def _split(estimates, **kwargs):  # type: ignore[no-untyped-def]
    defaults = {"min_battles": 3, "z": 1.0, "overlap_threshold": 0.5}
    return split_generals(estimates, {}, **{**defaults, **kwargs})


def test_generals_below_min_battles_are_under_determined_without_a_rank() -> None:
    ests = [
        estimate(1, rank=1, n_battles=5),
        estimate(2, rank=2, n_battles=2),
        estimate(3, rank=3, n_battles=3),
    ]

    headline, under = _split(ests)

    assert [g.estimate.general_id for g in headline] == [1, 3]
    assert [g.estimate.general_id for g in under] == [2]
    assert under[0].rank is None
    assert not under[0].is_headline


def test_headline_keeps_the_stored_ranks_and_never_renumbers() -> None:
    ests = [
        estimate(10, rank=4, n_battles=9),
        estimate(11, rank=2, n_battles=1),
        estimate(12, rank=9, n_battles=4),
        estimate(13, rank=1, n_battles=3),
    ]

    headline, _ = _split(ests)

    assert [g.rank for g in headline] == [1, 4, 9]
    assert [g.estimate.general_id for g in headline] == [13, 10, 12]


def test_ordering_follows_the_stored_rank_even_when_skill_disagrees() -> None:
    ests = [
        estimate(1, rank=1, skill_mean=-3.0),
        estimate(2, rank=2, skill_mean=4.0),
    ]

    headline, _ = _split(ests)

    assert [g.estimate.general_id for g in headline] == [1, 2]


def test_top_headline_takes_the_leading_rows_only() -> None:
    ests = [estimate(i, rank=i) for i in range(1, 6)]
    headline, _ = _split(ests)

    assert [g.rank for g in top_headline(headline, 2)] == [1, 2]


def test_under_determined_are_listed_by_battles_then_name() -> None:
    ests = [
        estimate(1, "Zed", n_battles=1),
        estimate(2, "Amy", n_battles=1),
        estimate(3, "Bob", n_battles=2),
    ]

    _, under = _split(ests)

    assert [g.estimate.name for g in under] == ["Bob", "Amy", "Zed"]


# ─── rank overlap ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0.0, 2.0), (1.0, 3.0), 0.5),
        ((0.0, 2.0), (3.0, 4.0), 0.0),
        ((0.0, 4.0), (1.0, 2.0), 1.0),
        ((1.0, 1.0), (0.0, 2.0), 1.0),
        ((1.0, 1.0), (2.0, 3.0), 0.0),
    ],
)
def test_overlap_share_is_a_share_of_the_narrower_interval(
    a: tuple[float, float], b: tuple[float, float], expected: float
) -> None:
    assert overlap_share(a, b) == pytest.approx(expected)


def _pair(threshold: float):  # type: ignore[no-untyped-def]
    # With z=1 the intervals are (0, 2) and (1, 3): an overlap of exactly half.
    ests = [
        estimate(1, rank=1, total_war=1.0, war_sd=1.0, stored_war_interval=False),
        estimate(2, rank=2, total_war=2.0, war_sd=1.0, stored_war_interval=False),
    ]
    headline, _ = _split(ests, overlap_threshold=threshold)
    return headline


def test_overlap_exactly_at_the_threshold_is_not_flagged() -> None:
    assert [g.overlaps_next for g in _pair(0.5)] == [False, None]


def test_overlap_just_over_the_threshold_is_flagged() -> None:
    assert [g.overlaps_next for g in _pair(0.49)] == [True, None]


def test_a_general_without_a_war_interval_has_no_overlap_answer() -> None:
    ests = [
        estimate(1, rank=1, total_war=None, war_sd=None),
        estimate(2, rank=2),
    ]

    headline, _ = _split(ests)

    assert headline[0].war_lo is None
    assert headline[0].overlaps_next is None
    assert headline[1].overlaps_next is None


def test_overlap_flags_can_be_switched_off() -> None:
    ests = [estimate(1, rank=1), estimate(2, rank=2)]

    headline, _ = _split(ests, show_overlap=False)

    assert all(g.overlaps_next is None for g in headline)


def test_the_overlap_interval_is_total_war_plus_or_minus_z_sd() -> None:
    headline, _ = _split(
        [estimate(1, rank=1, total_war=3.0, war_sd=0.5, stored_war_interval=False)], z=2.0
    )

    assert (headline[0].war_lo, headline[0].war_hi) == (2.0, 4.0)


def test_the_stored_war_interval_wins_over_the_normal_approximation() -> None:
    est = estimate(1, rank=1, total_war=3.0, war_sd=0.5)
    stored = GeneralEstimate(**{**est.__dict__, "war_ci_lo": 1.0, "war_ci_hi": 7.0})

    headline, _ = _split([stored], z=2.0)

    assert (headline[0].war_lo, headline[0].war_hi) == (1.0, 7.0)
    assert not headline[0].war_interval_approx


def test_a_general_without_a_stored_interval_is_marked_as_approximate() -> None:
    headline, _ = _split([estimate(1, rank=1, stored_war_interval=False)], z=2.0)

    assert headline[0].war_interval_approx


def test_z_for_the_stored_level_is_the_usual_multiplier() -> None:
    assert z_for_level(0.95) == pytest.approx(1.959964, abs=1e-5)


# ─── parameters ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", [0.9, 0.99, 0.5])
def test_a_ci_level_other_than_the_stored_95_is_refused(level: float) -> None:
    with pytest.raises(ValueError, match="ci_level"):
        ReportParams.from_mapping({"ci_level": level})


def test_the_spec_defaults_parse() -> None:
    import yaml

    from tests.unit.evaluators_helpers import ROOT

    raw = yaml.safe_load((ROOT / "agents" / "report.yaml").read_text(encoding="utf-8"))

    parsed = ReportParams.from_mapping(raw["params"])

    assert (parsed.top_n, parsed.min_battles, parsed.ci_level) == (100, 3, 0.95)
    assert parsed.report_path == "data/processed/report/manifest.json"


@pytest.mark.parametrize(
    "bad",
    [{"top_n": 0}, {"min_battles": 0}, {"overlap_threshold": 1.5}, {"run_selection": "oldest"}],
)
def test_invalid_parameters_are_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ReportParams.from_mapping(bad)


# ─── P(above next) over the headline ordering ────────────────────────────────


def _draws(columns: dict[int, list[float]], basis: str = "war") -> PooledDraws:
    return PooledDraws({g: np.array(v) for g, v in columns.items()}, basis)


def test_p_above_next_skips_a_hidden_general_between_two_listed_ones() -> None:
    ests = [
        estimate(1, rank=1, n_battles=5, p_above_next_rank=0.99),
        estimate(2, rank=2, n_battles=1, p_above_next_rank=0.99),
        estimate(3, rank=3, n_battles=5, p_above_next_rank=0.99),
    ]
    # 1 beats the hidden 2 always, but beats the listed 3 in only 1 of 4 draws.
    pooled = _draws({1: [5, 0, 0, 0], 2: [1, 1, 1, 1], 3: [2, 2, 2, 2]})

    headline, _ = _split(ests, pooled=pooled)

    assert headline[0].p_above_next == pytest.approx(0.25)
    assert headline[1].p_above_next is None


def test_p_above_next_falls_back_to_the_stored_column_without_draws() -> None:
    ests = [
        estimate(1, rank=1, p_above_next_rank=0.8),
        estimate(2, rank=2, p_above_next_rank=None),
    ]

    headline, _ = _split(ests)

    assert [g.p_above_next for g in headline] == [0.8, None]


def test_a_general_missing_from_the_draws_gets_no_probability() -> None:
    ests = [estimate(1, rank=1, p_above_next_rank=0.7), estimate(2, rank=2)]

    headline, _ = _split(ests, pooled=_draws({1: [1.0, 2.0]}))

    assert headline[0].p_above_next is None


def _write_npz(directory, **arrays):  # type: ignore[no-untyped-def]
    run = directory / "run7"
    run.mkdir()
    np.savez(run / "pooled_draws.npz", **arrays)


def test_skill_ranked_run_compares_skill_draws(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_npz(
        tmp_path,
        general_ids=np.array([1, 2]),
        war_draws=np.array([[0.0, 1.0], [0.0, 1.0]]),
        skill_draws=np.array([[2.0, 1.0], [2.0, 1.0]]),
    )

    pooled = load_pooled_draws(tmp_path, 7, "skill_mean")

    assert pooled is not None and pooled.basis == "skill" and not pooled.skill_unavailable
    headline, _ = _split([estimate(1, rank=1), estimate(2, rank=2)], pooled=pooled)
    assert headline[0].p_above_next == 1.0


def test_skill_ranked_run_without_skill_draws_keeps_war_and_says_so(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_npz(tmp_path, general_ids=np.array([1, 2]), war_draws=np.array([[0.0, 1.0], [0.0, 1.0]]))

    pooled = load_pooled_draws(tmp_path, 7, "skill_mean")

    assert pooled is not None and pooled.basis == "war" and pooled.skill_unavailable
    assert "no skill draws" in p_above_next_note("skill_unavailable")


def test_missing_draws_file_gives_none_and_the_disclosure_names_the_stored_value(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert load_pooled_draws(tmp_path, 7) is None
    v = view([estimate(1, rank=1), estimate(2, rank=2)])
    assert any("model stage's stored value" in d for d in v.disclosures)
    assert any("unwidened mixture" in d for d in v.disclosures)


def test_view_with_draws_discloses_the_recomputation() -> None:
    v = view(
        [estimate(1, rank=1), estimate(2, rank=2)],
        pooled_draws=_draws({1: [2.0, 2.0], 2: [1.0, 3.0]}),
    )

    assert v.headline[0].p_above_next == pytest.approx(0.5)
    assert any("recomputed from the pooled draws" in d for d in v.disclosures)
