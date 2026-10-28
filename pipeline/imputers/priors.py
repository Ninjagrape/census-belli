"""Configuration for the impute stage: era bins, priors and run parameters.

Era bins are on ``battles.year_astronomical`` (docs/handover-log.md 5.1), where 31 BC
is -30 and BC years are negative, so one comparison against an exclusive upper
bound puts every BC battle in 'ancient' without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import structlog
import yaml

__all__ = [
    "DEFAULT_PRIORS_PATH",
    "UNDATED",
    "EraPrior",
    "ImputeParams",
    "ImputePriors",
    "era_of_year",
    "load_priors",
]

logger = structlog.get_logger()

DEFAULT_PRIORS_PATH: Final[Path] = Path("config/imputation_priors.yaml")
UNDATED: Final[str] = "undated"


@dataclass(frozen=True)
class EraPrior:
    """Prior on one era's mean log troop count."""

    log_mu: float
    log_sd: float


@dataclass(frozen=True)
class ImputePriors:
    """Era bins, era priors and hyperparameter scales.

    Attributes:
        era_bins: ``(name, exclusive_upper_year)`` in ascending order; the
            last upper bound is None (open-ended).
        era_priors: Prior per era name, including ``'undated'``.
        hyper: HalfNormal scales and ZeroSumNormal sds, by name.
    """

    era_bins: tuple[tuple[str, int | None], ...]
    era_priors: dict[str, EraPrior]
    hyper: dict[str, float] = field(default_factory=dict)

    @property
    def era_names(self) -> tuple[str, ...]:
        """Every era name in model order, with 'undated' last."""
        return (*(name for name, _ in self.era_bins), UNDATED)


@dataclass(frozen=True)
class ImputeParams:
    """Run parameters read from ``agents/impute.yaml``."""

    scale: str = "neutral"
    n_imputations: int = 20
    mnar_delta: float = 0.0
    published_default_sd: float = 1.0
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    random_seed: int = 20261006
    nuts_sampler: str = "nutpie"
    write_max_rhat: float = 1.01
    write_min_ess_bulk: float = 400.0
    write_max_divergences: int = 0
    ratio_limit: float = 4.0
    value_min: float = 50.0
    value_max: float = 10_000_000.0
    ks_alpha: float = 0.01
    ks_min_n: int = 5
    min_pair_slope: float = 0.8
    pair_slope_tolerance: float = 0.1
    min_side_slope: float = 0.9
    slope_gate_min_n: int = 10
    model_type: str = "imputation"
    imputed_root: str = "data/imputed"
    priors_path: str = str(DEFAULT_PRIORS_PATH)
    reconcile_root: str = "data/processed/reconcile"
    draws_path: str | None = None
    propagate_shared_error: bool = True

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ImputeParams:
        """Read the run parameters from a spec's ``params`` block.

        Args:
            params: The spec's ``params`` mapping; absent keys take the
                defaults above.

        Returns:
            The parameters.

        Raises:
            ValueError: If ``scale`` is not ``'neutral'`` or ``'published'``.
        """
        d = cls()
        if "min_lik_precision" in params:
            # Removed 2026-10-22: reconcile exports each side's likelihood, so no
            # side with a point row can be weak. Ignored rather than honoured.
            logger.warning(
                "impute_min_lik_precision_ignored",
                hint="removed; a side is missing only when it has no point row",
            )
        scale = str(params.get("scale", d.scale))
        if scale not in ("neutral", "published"):
            raise ValueError(f"params.scale must be 'neutral' or 'published', got {scale!r}")
        draws_path = params.get("draws_path")
        return cls(
            scale=scale,
            n_imputations=int(params.get("n_imputations", d.n_imputations)),
            mnar_delta=float(params.get("mnar_delta", d.mnar_delta)),
            published_default_sd=float(params.get("published_default_sd", d.published_default_sd)),
            draws=int(params.get("mcmc_samples", d.draws)),
            tune=int(params.get("mcmc_tune", d.tune)),
            chains=int(params.get("mcmc_chains", d.chains)),
            cores=int(params.get("mcmc_cores", d.cores)),
            target_accept=float(params.get("mcmc_target_accept", d.target_accept)),
            random_seed=int(params.get("random_seed", d.random_seed)),
            nuts_sampler=str(params.get("nuts_sampler", d.nuts_sampler)),
            write_max_rhat=float(params.get("write_max_rhat", d.write_max_rhat)),
            write_min_ess_bulk=float(params.get("write_min_ess_bulk", d.write_min_ess_bulk)),
            write_max_divergences=int(params.get("write_max_divergences", d.write_max_divergences)),
            ratio_limit=float(params.get("ratio_limit", d.ratio_limit)),
            value_min=float(params.get("value_min", d.value_min)),
            value_max=float(params.get("value_max", d.value_max)),
            ks_alpha=float(params.get("ks_alpha", d.ks_alpha)),
            ks_min_n=int(params.get("ks_min_n", d.ks_min_n)),
            min_pair_slope=float(params.get("min_pair_slope", d.min_pair_slope)),
            pair_slope_tolerance=float(params.get("pair_slope_tolerance", d.pair_slope_tolerance)),
            min_side_slope=float(params.get("min_side_slope", d.min_side_slope)),
            slope_gate_min_n=int(params.get("slope_gate_min_n", d.slope_gate_min_n)),
            model_type=str(params.get("model_type", d.model_type)),
            imputed_root=str(params.get("imputed_root", d.imputed_root)),
            priors_path=str(params.get("imputation_priors", d.priors_path)),
            reconcile_root=str(params.get("reconcile_root", d.reconcile_root)),
            draws_path=None if draws_path is None else str(draws_path),
            propagate_shared_error=bool(
                params.get("propagate_shared_error", d.propagate_shared_error)
            ),
        )


def era_of_year(year: int | None, bins: tuple[tuple[str, int | None], ...]) -> str:
    """Assign a battle year to an era bin.

    Args:
        year: ``battles.year_astronomical``, or None for an undated battle.
        bins: ``(name, exclusive_upper_year)`` ascending; the last upper
            bound is None.

    Returns:
        The era name, or ``'undated'`` when ``year`` is None.
    """
    if year is None:
        return UNDATED
    for name, upper in bins:
        if upper is None or year < upper:
            return name
    return bins[-1][0]


def load_priors(path: Path | str = DEFAULT_PRIORS_PATH) -> ImputePriors:
    """Read ``config/imputation_priors.yaml``.

    Args:
        path: The YAML file.

    Returns:
        The priors.

    Raises:
        FileNotFoundError: If the file is absent. There is no silent
            default: an imputation on invented era priors would be an
            unreviewed modelling choice.
        ValueError: If an era has no prior, or ``undated`` is missing.
    """
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    bins = tuple(
        (str(entry["name"]), None if entry.get("before") is None else int(entry["before"]))
        for entry in loaded.get("eras", [])
    )
    if not bins:
        raise ValueError(f"{path} defines no eras")
    priors = {
        str(name): EraPrior(log_mu=float(spec["log_mu"]), log_sd=float(spec["log_sd"]))
        for name, spec in (loaded.get("era_priors") or {}).items()
    }
    missing = [n for n in (*(name for name, _ in bins), UNDATED) if n not in priors]
    if missing:
        raise ValueError(f"{path} has no era_priors entry for {missing}")
    hyper = {str(k): float(v) for k, v in (loaded.get("hyperpriors") or {}).items()}
    return ImputePriors(era_bins=bins, era_priors=priors, hyper=hyper)
