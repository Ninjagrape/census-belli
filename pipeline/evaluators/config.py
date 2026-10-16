"""Parameters of the evaluate stage and the model parameters its refits use."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pipeline.modelers.config import ModelParams, load_model_params

__all__ = ["EvaluateParams", "load_eval_model_params"]

PRIMARY_SCALE = "neutral"


@dataclass(frozen=True)
class EvaluateParams:
    """The ``params`` block of ``agents/evaluate.yaml``.

    Attributes:
        model_spec: Path of the model stage's spec, whose params give the
            priors, sampler and write thresholds.
        expert_rankings: Path of the expert rankings YAML.
        reconcile_root: Directory holding reconcile's ``neutral_draws_run*.npz``,
            read by the ``g_unlabelled`` sensitivity variants.
        output_root: Root of the per-run output directories.
        report_path: Where the latest report is copied.
        seed: Base seed of every evaluation fit.
        held_out_folds: Number of cross-validation folds.
        cv_rounds: Imputed rounds per fold fit.
        eval_draws: Sampler draws per chain for evaluation fits.
        eval_min_ess_bulk: Bulk ESS floor of evaluation fits. Evaluation fits
            use the lower of this and the model's ``min_ess_bulk``.
        calibration_bins: Equal-width bins of the calibration table.
        sensitivity_rounds: Imputed rounds per sensitivity fit.
        top_n: Size of the top set compared across variants.
        min_top_overlap: Required top-set overlap at ``top_n``.
        sensitivity_variants: Raw variant mappings for ``parse_variants``.
        raw: The mapping as given, recorded in the report.
    """

    model_spec: str
    expert_rankings: str
    reconcile_root: str
    output_root: str
    report_path: str
    seed: int
    held_out_folds: int
    cv_rounds: int
    eval_draws: int
    eval_min_ess_bulk: float
    calibration_bins: int
    sensitivity_rounds: int
    top_n: int
    min_top_overlap: int
    sensitivity_variants: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> EvaluateParams:
        """Read and validate the spec's ``params``.

        Args:
            params: ``agents/evaluate.yaml``'s ``params``.

        Returns:
            The parameters, with the spec's defaults for absent keys.

        Raises:
            ValueError: If a count is below its minimum or ``min_top_overlap``
                exceeds ``top_n``.
        """
        out = cls(
            model_spec=str(params.get("model_spec", "agents/model.yaml")),
            expert_rankings=str(params.get("expert_rankings", "config/expert_rankings.yaml")),
            reconcile_root=str(params.get("reconcile_root", "data/processed/reconcile")),
            output_root=str(params.get("output_root", "data/processed/evaluation")),
            report_path=str(params.get("report_path", "data/processed/evaluation_report.json")),
            seed=int(params.get("seed", 20261014)),
            held_out_folds=int(params.get("held_out_folds", 5)),
            cv_rounds=int(params.get("cv_rounds", 3)),
            eval_draws=int(params.get("eval_draws", 1500)),
            eval_min_ess_bulk=float(params.get("eval_min_ess_bulk", 400)),
            calibration_bins=int(params.get("calibration_bins", 5)),
            sensitivity_rounds=int(params.get("sensitivity_rounds", 5)),
            top_n=int(params.get("top_n", 20)),
            min_top_overlap=int(params.get("min_top_overlap", 12)),
            sensitivity_variants=tuple(params.get("sensitivity_variants") or ()),
            raw=dict(params),
        )
        out._validate()
        return out

    def _validate(self) -> None:
        minimums = {
            "held_out_folds": (self.held_out_folds, 2),
            "cv_rounds": (self.cv_rounds, 1),
            "eval_draws": (self.eval_draws, 2),
            "eval_min_ess_bulk": (self.eval_min_ess_bulk, 1),
            "calibration_bins": (self.calibration_bins, 1),
            "sensitivity_rounds": (self.sensitivity_rounds, 1),
            "top_n": (self.top_n, 1),
            "min_top_overlap": (self.min_top_overlap, 0),
        }
        for name, (value, floor) in minimums.items():
            if value < floor:
                raise ValueError(f"{name} must be at least {floor}, got {value}")
        if self.min_top_overlap > self.top_n:
            raise ValueError("min_top_overlap cannot exceed top_n")


def load_eval_model_params(evaluate_params: EvaluateParams) -> ModelParams:
    """Model parameters for evaluation fits.

    Loads the model stage's spec so evaluation fits use the published run's
    priors, thresholds and sampler, then lowers the draws to ``eval_draws`` and
    lowers the bulk ESS floor to ``eval_min_ess_bulk``, never raising it above
    the model's (evaluation fits are compared, not published). ``max_rhat`` and
    ``max_divergences`` stay the model's.

    Args:
        evaluate_params: The evaluate stage's parameters.

    Returns:
        The model parameters.

    Raises:
        ValueError: If the model spec is not a mapping, or its scale is not
            the primary one: evaluate validates the published ranking only.
    """
    spec = yaml.safe_load(Path(evaluate_params.model_spec).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"{evaluate_params.model_spec} must contain a mapping")
    mp = load_model_params(spec.get("params") or {})
    if mp.scale != PRIMARY_SCALE:
        raise ValueError(
            f"evaluate validates the primary ranking only: scale must be "
            f"{PRIMARY_SCALE!r}, got {mp.scale!r}"
        )
    sampler = dataclasses.replace(mp.sampler, draws=evaluate_params.eval_draws)
    return dataclasses.replace(
        mp, sampler=sampler, min_ess_bulk=min(mp.min_ess_bulk, evaluate_params.eval_min_ess_bulk)
    )
