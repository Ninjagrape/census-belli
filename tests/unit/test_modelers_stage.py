"""Model stage units that need neither a database nor a sampler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.modelers.components import components
from pipeline.modelers.config import load_model_params
from pipeline.modelers.design import build_design
from pipeline.modelers.gates import exclusions_reported, top_generals_cross_component
from pipeline.modelers.load import (
    ImputedRound,
    check_round_provenance,
    read_rounds,
    subset_sides,
)
from pipeline.modelers.outputs import to_jsonable
from pipeline.quality import QualityRunner
from tests.unit.modelers_helpers import battle


def _write_round(root: Path, n: int, run_id: int, scale: str = "neutral") -> None:
    directory = root / scale / f"round_{n}"
    directory.mkdir(parents=True)
    (directory / "sides.csv").write_text(
        "side_id,battle_id,log_troops,is_imputed,scale\n1,1,9.5,False,neutral\n2,1,8.0,True,neutral\n",
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "scale": scale}), encoding="utf-8"
    )


def test_read_rounds_orders_numerically_and_parses_values(tmp_path: Path) -> None:
    for n in (10, 2, 1):
        _write_round(tmp_path, n, 5)

    rounds = read_rounds(tmp_path)

    assert [r.index for r in rounds] == [1, 2, 10]
    assert rounds[0].log_troops == {1: 9.5, 2: 8.0}
    assert len(read_rounds(tmp_path, max_rounds=2)) == 2


def test_read_rounds_without_rounds_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="impute stage first"):
        read_rounds(tmp_path)


def test_read_rounds_rejects_a_wrong_scale_manifest(tmp_path: Path) -> None:
    _write_round(tmp_path, 0, 5)
    manifest = tmp_path / "neutral" / "round_0" / "manifest.json"
    manifest.write_text(json.dumps({"run_id": 5, "scale": "published"}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'neutral'"):
        read_rounds(tmp_path)


def _rounds(*run_ids: int) -> list[ImputedRound]:
    return [ImputedRound(i, r, "neutral", {}) for i, r in enumerate(run_ids)]


def test_provenance_accepts_matching_run() -> None:
    assert check_round_provenance(_rounds(4, 4), 4) == 4


@pytest.mark.parametrize(
    ("run_ids", "expected", "message"),
    [
        ((4, 5), 5, "several impute runs"),
        ((4, 4), 5, "latest neutral impute run"),
        ((4,), None, "latest neutral impute run"),
    ],
)
def test_provenance_refuses_mixed_or_stale(
    run_ids: tuple[int, ...], expected: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_round_provenance(_rounds(*run_ids), expected)


def test_subset_sides_keeps_whole_battles() -> None:
    sides = battle(1, (1,), (2,)) + battle(2, (3,), (4,)) + battle(3, (5,), (6,))

    kept = subset_sides(sides, 2)

    assert {s.battle_id for s in kept} == {1, 2} and len(kept) == 4
    assert len(subset_sides(sides, None)) == 6


def test_config_overrides_merge_section_by_section(tmp_path: Path) -> None:
    base = tmp_path / "m.yaml"
    base.write_text(yaml.safe_dump({"sampler": {"draws": 10, "tune": 20}, "rank_by": "total_war"}))

    params = load_model_params({"model_config": str(base), "sampler": {"draws": 99}})

    assert (params.sampler.draws, params.sampler.tune) == (99, 20)
    with pytest.raises(ValueError, match="rank_by"):
        load_model_params({"model_config": str(base), "rank_by": "vibes"})


def test_default_config_loads() -> None:
    params = load_model_params({})

    assert params.sampler.draws == 1500 and params.max_rhat == 1.02
    assert params.gates.ppc_level == 0.99 and params.replacement_percentile == 40


def test_cross_component_counts_top_generals_outside_the_largest() -> None:
    result = top_generals_cross_component(
        general_ids=np.array([10, 11, 12]),
        component_id=np.array([0, 1, 0]),
        top_idx=np.array([0, 1, 2]),
    )

    assert not result.passed and result.details["general_ids"] == [11]


def test_exclusions_reported_warns_above_the_share() -> None:
    design = build_design(
        battle(1, (1,), (2,)) + battle(2, (3,), (4,), a_outcome=None, b_outcome=None)
    )

    assert exclusions_reported(design, 0.4).passed is False
    assert exclusions_reported(design, 0.6).passed is True


def test_to_jsonable_handles_numpy_and_nonfinite() -> None:
    out = to_jsonable({"a": np.float64(1.5), "b": np.array([1, 2]), "c": float("nan")})

    assert json.dumps(out) == '{"a": 1.5, "b": [1, 2], "c": null}'


def test_model_gate_handler_reads_the_newest_run_directory(tmp_path: Path) -> None:
    for run_id, passed in ((3, False), (12, True)):
        directory = tmp_path / f"run{run_id}"
        directory.mkdir()
        gate = {"name": "g", "passed": passed, "actual": f"run {run_id}", "message": "m"}
        (directory / "gate_report.json").write_text(json.dumps({"gates": [gate]}))
    check = {"name": "g", "method": "model_gate", "params": {"root": str(tmp_path)}}

    result = QualityRunner(None).run_one(check)

    assert result.passed and result.actual_value == "run 12"
    missing = QualityRunner(None).run_one({**check, "params": {"root": str(tmp_path / "x")}})
    assert not missing.passed


def test_joint_fit_failure_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.stages import model as stage

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("sampler exploded")

    monkeypatch.setattr(stage, "_fit_joint", boom)

    result = stage._try_fit_joint(
        None,
        owns_connection=False,
        mp=load_model_params({}),
        design=build_design(battle(1, (1,), (2,))),
        imputation_run_id=1,
    )

    assert result is None


def test_circularity_gate_says_so_when_the_joint_fit_did_not_run() -> None:
    from pipeline.modelers.checks import prior_predictive
    from pipeline.modelers.components import career_spans
    from pipeline.modelers.config import GateParams
    from pipeline.modelers.gates import run_gates
    from pipeline.modelers.model import PosteriorDraws, assemble_side_scores
    from pipeline.modelers.priors import ModelPriors
    from pipeline.modelers.summarise import pool_reductions, reduce_draws

    sides = (
        battle(1, (1,), (2,))
        + battle(2, (2,), (3,))
        + battle(3, (3,), (1,), a_outcome="defeat", b_outcome="victory")
    )
    design = build_design(sides)
    rng = np.random.default_rng(0)
    skills = rng.normal(size=(30, design.n_generals))
    draws = PosteriorDraws(
        skills=skills,
        side_scores=assemble_side_scores(design, skills, np.ones(30), None, None, None),
        beta_force=np.full(30, 0.5),
        beta_def=np.zeros(30),
        c1=np.full(30, 0.5),
        c2=np.full(30, 2.0),
        sigma_skill=np.ones(30),
    )
    x = np.zeros(design.n_battles)
    pooled = pool_reductions(design, [reduce_draws(design, draws, x, 40.0)] * 2, "total_war")

    results = run_gates(
        design=design,
        components=components(design),
        spans=career_spans(design),
        pooled=pooled,
        joint=None,
        prior_pp=prior_predictive(design, x, ModelPriors(), n_draws=50, n_pairs=20),
        ppc_draws=draws,
        ppc_x_diff=x,
        params=GateParams(ppc_sims=20),
    )

    circ = next(r for r in results if r.name == "circularity_check")
    assert not circ.passed and circ.severity == "warning"
    assert "did not run" in circ.message


def test_naval_and_aerial_included_battles_are_counted() -> None:
    from pipeline.stages.model import _type_counts

    design = build_design(battle(1, (1,), (2,)) + battle(2, (3,), (4,)) + battle(3, (5,), (6,)))

    counts = _type_counts(design, {1: "naval", 2: "aerial", 3: "field"})

    assert counts == {"battles_included_naval": 1, "battles_included_aerial": 1}
