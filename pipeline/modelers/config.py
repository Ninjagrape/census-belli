"""Run parameters for the model stage.

``config/model_default.yaml`` holds the defaults and says why each is what it
is. ``agents/model.yaml`` names that file in ``params.model_config``; any other
key under its ``params`` overrides the file, section by section, so a test or a
sensitivity run changes one number without copying the file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from pipeline.modelers.priors import ModelPriors
from pipeline.reconcilers.model import SamplerSettings

__all__ = ["DEFAULT_CONFIG_PATH", "GateParams", "ModelParams", "load_model_params"]

DEFAULT_CONFIG_PATH: Final[Path] = Path("config/model_default.yaml")
RANK_BY_CHOICES: Final[tuple[str, ...]] = ("total_war", "skill_mean")
SCALE_CHOICES: Final[tuple[str, ...]] = ("neutral", "published")
CONFIG_KEY: Final[str] = "model_config"


@dataclass(frozen=True)
class GateParams:
    """Thresholds for the dataset-level gates."""

    top_n: int = 50
    fmi_threshold: float = 0.3
    min_credited_share: float = 0.5
    max_default_split_share: float = 0.3
    extreme_log_ratio_cutoff: float = -2.0
    extreme_max_share: float = 0.5
    max_career_span_years: float = 60.0
    prior_max_category: float = 0.4
    prior_max_median_dev: float = 0.35
    ppc_level: float = 0.99
    ppc_sims: int = 500
    circ_slope_sd_factor: float = 0.5
    circ_min_spearman: float = 0.9
    circ_min_battles: int = 3
    circ_min_overlap: int = 40
    max_excluded_share: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GateParams:
        """Read the ``gates`` section, defaulting absent keys.

        Args:
            values: The section.

        Returns:
            The thresholds.
        """
        base = cls()
        out: dict[str, Any] = {}
        for name, default in base.__dict__.items():
            raw = values.get(name, default)
            out[name] = int(raw) if isinstance(default, int) else float(raw)
        return cls(**out)


@dataclass(frozen=True)
class ModelParams:
    """Everything the stage reads from configuration.

    Attributes:
        priors: Prior hyperparameters.
        sampler: Sampler settings for the first attempt of each fit.
        retry_target_accept: ``target_accept`` for the one retry.
        max_rhat: Write threshold per round.
        min_ess_bulk: Write threshold per round.
        max_divergences: Write threshold per round.
        replacement_percentile: Replacement-level percentile.
        rank_by: ``total_war`` or ``skill_mean``.
        scale: Imputed scale to read: ``neutral`` (primary) or ``published``
            (sensitivity; its runs get a ``_published`` model_type suffix).
        imputed_root: Root of the impute stage's output.
        diagnostics_root: Where per-run files go.
        draws_kept_per_round: Draws kept per round for pooling.
        max_rounds: Use at most this many rounds, or None.
        joint_latent: Whether to fit the circularity comparator.
        joint_sigma_w: Within-battle sd of log troops for the comparator.
        prior_draws: Prior predictive draws.
        prior_pairs: Prior predictive pairings per draw.
        prior_seed: Prior predictive seed.
        gates: Gate thresholds.
        raw: The merged mapping, recorded in ``model_runs.config``.
    """

    priors: ModelPriors
    sampler: SamplerSettings
    retry_target_accept: float
    max_rhat: float
    min_ess_bulk: float
    max_divergences: int
    replacement_percentile: float
    rank_by: str
    scale: str
    imputed_root: str
    diagnostics_root: str
    draws_kept_per_round: int
    max_rounds: int | None
    joint_latent: bool
    joint_sigma_w: float
    prior_draws: int
    prior_pairs: int
    prior_seed: int
    gates: GateParams
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, merged: Mapping[str, Any]) -> ModelParams:
        """Build the parameters from a merged configuration mapping.

        Args:
            merged: The YAML defaults with overrides applied.

        Returns:
            The parameters.

        Raises:
            ValueError: On an unknown ``rank_by`` or ``scale``, or fewer than
                two kept draws per round (Rubin's within-variance needs two).
        """
        rank_by = str(merged.get("rank_by", "total_war"))
        if rank_by not in RANK_BY_CHOICES:
            raise ValueError(f"rank_by must be one of {RANK_BY_CHOICES}, got {rank_by!r}")
        scale = str(merged.get("scale", "neutral"))
        if scale not in SCALE_CHOICES:
            raise ValueError(f"scale must be one of {SCALE_CHOICES}, got {scale!r}")
        sampler = dict(merged.get("sampler") or {})
        write = dict(merged.get("write") or {})
        pooling = dict(merged.get("pooling") or {})
        prior_pred = dict(merged.get("prior_predictive") or {})
        defaults = SamplerSettings()
        kept = int(pooling.get("draws_kept_per_round", 500))
        if kept < 2:
            raise ValueError("pooling.draws_kept_per_round must be at least 2")
        max_rounds = pooling.get("max_rounds")
        return cls(
            priors=ModelPriors.from_params(dict(merged.get("priors") or {})),
            sampler=SamplerSettings(
                draws=int(sampler.get("draws", defaults.draws)),
                tune=int(sampler.get("tune", defaults.tune)),
                chains=int(sampler.get("chains", defaults.chains)),
                cores=int(sampler.get("cores", defaults.cores)),
                target_accept=float(sampler.get("target_accept", defaults.target_accept)),
                random_seed=int(sampler.get("random_seed", defaults.random_seed)),
                nuts_sampler=str(sampler.get("nuts_sampler", defaults.nuts_sampler)),
            ),
            retry_target_accept=float(sampler.get("retry_target_accept", 0.95)),
            max_rhat=float(write.get("max_rhat", 1.02)),
            min_ess_bulk=float(write.get("min_ess_bulk", 800.0)),
            max_divergences=int(write.get("max_divergences", 0)),
            replacement_percentile=float(merged.get("replacement_percentile", 40.0)),
            rank_by=rank_by,
            scale=scale,
            imputed_root=str(merged.get("imputed_root", "data/imputed")),
            diagnostics_root=str(
                merged.get("diagnostics_root", "data/processed/model_diagnostics")
            ),
            draws_kept_per_round=kept,
            max_rounds=None if max_rounds is None else int(max_rounds),
            joint_latent=bool((merged.get("joint_latent") or {}).get("enabled", True)),
            joint_sigma_w=float((merged.get("joint_latent") or {}).get("sigma_w", 1.0)),
            prior_draws=int(prior_pred.get("draws", 1000)),
            prior_pairs=int(prior_pred.get("pairs", 500)),
            prior_seed=int(prior_pred.get("seed", 0)),
            gates=GateParams.from_mapping(dict(merged.get("gates") or {})),
            raw=dict(merged),
        )


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins, mappings merge key by key."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_model_params(spec_params: Mapping[str, Any]) -> ModelParams:
    """Read ``config/model_default.yaml`` and apply the spec's overrides.

    Args:
        spec_params: ``agents/model.yaml``'s ``params``; ``model_config`` names
            the YAML file and every other key overrides it.

    Returns:
        The parameters.

    Raises:
        FileNotFoundError: If the config file is absent. There is no silent
            default: a model fit on invented priors is an unreviewed choice.
    """
    path = Path(str(spec_params.get(CONFIG_KEY, DEFAULT_CONFIG_PATH)))
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides = {k: v for k, v in spec_params.items() if k != CONFIG_KEY}
    return ModelParams.from_mapping(_merge(dict(loaded), overrides))
