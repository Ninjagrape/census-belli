"""Unit tests for impute configuration, era binning and key derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.imputers.load import polity_key, primary_commander_keys, subset_sides
from pipeline.imputers.priors import ImputeParams, era_of_year, load_priors
from pipeline.imputers.records import SideRow

PRIORS_PATH = Path(__file__).resolve().parents[2] / "config" / "imputation_priors.yaml"


@pytest.mark.parametrize(
    ("year", "era"),
    [
        (-480, "ancient"),  # Salamis, BC years are negative
        (-30, "ancient"),  # Actium, astronomical numbering
        (0, "ancient"),
        (499, "ancient"),
        (500, "medieval"),
        (1499, "medieval"),
        (1500, "early_modern"),
        (1791, "early_modern"),
        (1792, "napoleonic"),
        (1815, "napoleonic"),
        (1816, "industrial"),
        (1913, "industrial"),
        (1914, "modern"),
        (2003, "modern"),
        (None, "undated"),
    ],
)
def test_era_binning(year: int | None, era: str) -> None:
    priors = load_priors(PRIORS_PATH)

    assert era_of_year(year, priors.era_bins) == era


def test_priors_cover_every_era_and_widen_the_original_sds() -> None:
    priors = load_priors(PRIORS_PATH)

    assert priors.era_names == (
        "ancient",
        "medieval",
        "early_modern",
        "napoleonic",
        "industrial",
        "modern",
        "undated",
    )
    # agents/impute.yaml's earlier sketch plus 0.5.
    assert priors.era_priors["ancient"].log_mu == pytest.approx(9.2)
    assert priors.era_priors["ancient"].log_sd == pytest.approx(2.0)
    assert priors.era_priors["napoleonic"].log_sd == pytest.approx(1.5)
    assert priors.era_priors["undated"].log_sd >= max(
        p.log_sd for n, p in priors.era_priors.items() if n != "undated"
    )
    assert set(priors.hyper) >= {"tau_b", "sigma", "tau_p", "tau_c", "zsn_outcome"}


def test_missing_era_prior_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(
        "eras: [{name: a, before: 10}, {name: b, before: null}]\n"
        "era_priors: {a: {log_mu: 1, log_sd: 1}, undated: {log_mu: 1, log_sd: 1}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\['b'\]"):
        load_priors(path)


def test_absent_priors_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_priors(tmp_path / "nope.yaml")


def test_params_defaults_and_overrides() -> None:
    defaults = ImputeParams.from_params({})
    custom = ImputeParams.from_params(
        {"scale": "published", "n_imputations": 5, "mnar_delta": -0.25, "draws_path": "x.npz"}
    )

    assert defaults.scale == "neutral"
    assert defaults.n_imputations == 20
    assert defaults.mnar_delta == 0.0
    assert defaults.write_max_rhat == 1.01
    assert custom.scale == "published"
    assert custom.n_imputations == 5
    assert custom.mnar_delta == -0.25
    assert custom.draws_path == "x.npz"


def test_unknown_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="scale"):
        ImputeParams.from_params({"scale": "gaussian"})


def test_agent_spec_params_parse_and_match_the_design_defaults() -> None:
    import yaml

    spec = yaml.safe_load(
        (PRIORS_PATH.parents[1] / "agents" / "impute.yaml").read_text(encoding="utf-8")
    )
    params = ImputeParams.from_params(spec["params"])

    assert params.n_imputations == 20
    assert params.mnar_delta == 0.0
    assert params.write_max_rhat == 1.01
    assert params.write_min_ess_bulk == 400
    assert params.ratio_limit == 4.0
    assert (params.value_min, params.value_max) == (50, 10_000_000)


@pytest.mark.parametrize(
    ("wikidata_id", "polity", "expected"),
    [
        ("Q142", "France", "Q142"),
        (None, "Kingdom of France", "Kingdom of France"),
        ("  ", " Rome ", "Rome"),
        (None, None, "unknown"),
        ("", "", "unknown"),
    ],
)
def test_polity_key_prefers_wikidata_then_string_then_unknown(
    wikidata_id: str | None, polity: str | None, expected: str
) -> None:
    assert polity_key(wikidata_id, polity) == expected


def test_primary_commander_is_max_weight_with_lowest_bc_id_on_ties() -> None:
    rows = [
        (1, 10, 100, 0.85),  # side 1: Agrippa-like, wins on weight
        (1, 11, 101, 0.15),
        (2, 20, 200, 0.5),  # side 2: tie, lowest bc_id wins
        (2, 19, 199, 0.5),
        (3, 30, 300, None),  # NULL weight counts as 0
        (3, 31, 301, 0.1),
    ]

    keys = primary_commander_keys(rows)

    assert keys == {1: "general_100", 2: "general_199", 3: "general_301"}
    assert 4 not in keys  # a side with no commanders is absent; callers use 'unknown'


def _row(side_id: int, battle_id: int) -> SideRow:
    return SideRow(side_id, battle_id, 1800, None, None, None, "unknown", "unknown")


def test_subset_sides_keeps_whole_battles() -> None:
    sides = [_row(1, 5), _row(2, 5), _row(3, 6), _row(4, 7)]

    assert [s.side_id for s in subset_sides(sides, 2)] == [1, 2, 3]
    assert len(subset_sides(sides, None)) == 4


def test_impute_stage_satisfies_the_stage_contract_and_is_in_the_pipeline_order() -> None:
    from pipeline.orchestrator import STAGE_ORDER
    from pipeline.stages import impute
    from pipeline.stages.base import check_conforms

    check_conforms(impute)
    assert "impute" in STAGE_ORDER
