"""Shared fixtures and fakes for the evaluate stage unit tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from pipeline.evaluators import sensitivity as sens_module
from pipeline.evaluators.config import EvaluateParams
from pipeline.evaluators.fit import JointFit, PooledResult, RoundFit
from pipeline.evaluators.sensitivity import (
    SensitivityInputs,
)
from pipeline.evaluators.target import PublishedRun
from pipeline.modelers.config import ModelParams
from pipeline.modelers.design import build_design
from pipeline.modelers.load import ImputedRound
from pipeline.modelers.model import PosteriorDraws, assemble_side_scores
from pipeline.modelers.records import ModelDesign
from tests.unit.modelers_helpers import battle

ROOT = Path(__file__).resolve().parents[2]
K = 30
N_ROUNDS = 3
GENERALS = (1, 2, 3, 4)
# The spec names its priors variant; follow it rather than hard-code the name.
PRIORS = next(
    v["name"]
    for v in yaml.safe_load((ROOT / "agents" / "evaluate.yaml").read_text(encoding="utf-8"))[
        "params"
    ]["sensitivity_variants"]
    if v["kind"] == "priors"
)


def _sides() -> list[Any]:
    out: list[Any] = []
    for b in range(1, 21):
        a, c = GENERALS[b % 4], GENERALS[(b + 1 + b // 4) % 4]
        if a == c:
            c = GENERALS[(b + 2) % 4]
        win = b % 3 != 0
        out += battle(
            b,
            (a,),
            (c,),
            a_outcome="decisive_victory" if win else "defeat",
            b_outcome="decisive_defeat" if win else "victory",
            year=-200 + 100 * b,
        )
    return out


def _design() -> ModelDesign:
    return build_design(_sides())


def _rounds(design: ModelDesign) -> list[ImputedRound]:
    rng = np.random.default_rng(1)
    return [
        ImputedRound(i, 5, "neutral", {int(s): float(rng.normal(9, 0.3)) for s in design.side_ids})
        for i in range(N_ROUNDS)
    ]


def _draws(design: ModelDesign) -> PosteriorDraws:
    rng = np.random.default_rng(2)
    skills = rng.normal(0.0, 1.0, (K, design.n_generals))
    sigma = np.ones(K)
    return PosteriorDraws(
        skills=skills,
        side_scores=assemble_side_scores(design, skills, sigma, None, None, None),
        beta_force=np.full(K, 0.5),
        beta_def=np.zeros(K),
        c1=np.full(K, 0.5),
        c2=np.full(K, 2.0),
        sigma_skill=sigma,
    )


def _published(design: ModelDesign) -> PublishedRun:
    n = design.n_generals
    return PublishedRun(
        run_id=7,
        imputation_run_id=5,
        general_ids=design.general_ids.copy(),
        names=[f"General {g}" for g in design.general_ids],
        wikidata_ids=[f"Q{100 + int(g)}" for g in design.general_ids],
        skill_mean=np.linspace(1.0, -1.0, n),
        total_war=np.linspace(2.0, -2.0, n),
        rank=np.arange(1, n + 1, dtype=float),
        n_battles=design.n_included_battles,
        ci_lo=np.linspace(0.5, -1.5, n),
        ci_hi=np.linspace(1.5, -0.5, n),
    )


def _pooled(design: ModelDesign) -> SimpleNamespace:
    ids = design.general_ids.astype(float)
    return SimpleNamespace(
        score=-ids, skill_mean=-ids / 2, realised_mean=-ids * 3, rank=np.arange(ids.size)
    )


class FakeFitter:
    """Stands in for ``fit_pooled``: records calls, never samples."""

    def __init__(self, *, fail_label: str | None = None, unconverged: tuple[str, ...] = ()) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_label = fail_label
        self.unconverged = unconverged

    def __call__(
        self, design: ModelDesign, x_diffs: Any, mp: Any, *, seed: int, label: str, **kw: Any
    ) -> PooledResult:
        self.calls.append(
            {"label": label, "seed": seed, "x_diffs": x_diffs, "design": design, **kw}
        )
        if label == self.fail_label:
            raise RuntimeError("sampler exploded")
        converged = label not in self.unconverged
        draws = _draws(design)
        rounds = [
            RoundFit(i, SimpleNamespace(), draws, None, np.asarray(x))  # type: ignore[arg-type]
            for i, x in enumerate(x_diffs)
        ]
        worst = {"converged": "true" if converged else "false", "max_rhat": 1.0}
        return PooledResult(rounds, converged, worst, _pooled(design) if converged else None)  # type: ignore[arg-type]


class FakeJointFitter:
    """Stands in for ``fit_joint``: records calls, returns draws carrying an x_diff."""

    def __init__(self, *, unconverged: bool = False, fail: bool = False, x_value: float = 0.0):
        self.calls: list[dict[str, Any]] = []
        self.unconverged = unconverged
        self.fail = fail
        self.x_value = x_value

    def __call__(
        self,
        design: ModelDesign,
        obs_mean: np.ndarray,
        obs_sd: np.ndarray,
        mp: Any,
        *,
        seed: int,
        label: str,
        loglik_weight: np.ndarray | None = None,
    ) -> JointFit:
        self.calls.append({"label": label, "seed": seed, "loglik_weight": loglik_weight})
        if self.fail:
            raise RuntimeError("joint exploded")
        if self.unconverged:
            return JointFit(False, {"converged": "false"}, None)
        draws = dataclasses.replace(
            _draws(design), x_diff=np.full((K, design.n_battles), self.x_value)
        )
        return JointFit(True, {"converged": "true"}, draws)


def _mp() -> ModelParams:
    return ModelParams.from_mapping({"pooling": {"draws_kept_per_round": 10}})


def _params(tmp_path: Path, **overrides: Any) -> EvaluateParams:
    raw = yaml.safe_load((ROOT / "agents" / "evaluate.yaml").read_text(encoding="utf-8"))["params"]
    raw.update(
        model_spec=str(ROOT / "agents" / "model.yaml"),
        output_root=str(tmp_path / "evaluation"),
        report_path=str(tmp_path / "evaluation_report.json"),
        held_out_folds=3,
        cv_rounds=2,
        sensitivity_rounds=2,
        top_n=3,
        min_top_overlap=2,
        eval_draws=20,
    )
    raw.update(overrides)
    return EvaluateParams.from_mapping(raw)


def _measured(design: ModelDesign) -> tuple[np.ndarray, np.ndarray]:
    return np.full(design.n_sides, 9.0), np.full(design.n_sides, 0.1)


def _inputs(tmp_path: Path, design: ModelDesign, *, quality: dict[int, float] | None = None) -> Any:
    rounds = _rounds(design)
    return SensitivityInputs(
        design=design,
        sides=_sides(),
        rounds=rounds,
        x_diffs=[np.zeros(design.n_battles)] * N_ROUNDS,
        mp=_mp(),
        params=_params(tmp_path),
        measured=_measured(design),
        load_quality=lambda: quality if quality is not None else {int(design.battle_ids[0]): 0.1},
        published=_published(design),
    )


def _patch_sensitivity(monkeypatch: pytest.MonkeyPatch, fitter: FakeFitter) -> None:
    monkeypatch.setattr(sens_module, "fit_pooled", fitter)
    monkeypatch.setattr(
        sens_module,
        "repool",
        lambda design, rounds, **kw: SimpleNamespace(
            score=-design.general_ids.astype(float) * kw["replacement_percentile"],
            skill_mean=-design.general_ids.astype(float),
            realised_mean=design.general_ids.astype(float),
        ),
    )
