"""Prior hyperparameters for the Bradley-Terry model (design section 10)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["ModelPriors"]


@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters, with the design's defaults.

    Attributes:
        skill_prior_sd: Scale of the HalfNormal on ``sigma_skill``.
        mu_unknown_sd: Sd of ``mu_unknown ~ Normal(0, sd)``.
        mu_unattributed_sd: Sd of ``mu_unattributed ~ Normal(0, sd)``.
        beta_force_mu: Mean of the force-ratio coefficient prior.
        beta_force_sd: Its sd.
        beta_def_mu: Mean of the defender coefficient prior.
        beta_def_sd: Its sd.
        log_c1_mu: Mean of ``log c1``.
        log_c1_sd: Sd of ``log c1``.
        log_gap_mu: Mean of ``log (c2 - c1)``.
        log_gap_sd: Sd of ``log (c2 - c1)``.
    """

    skill_prior_sd: float = 1.0
    mu_unknown_sd: float = 1.0
    mu_unattributed_sd: float = 1.0
    beta_force_mu: float = 0.5
    beta_force_sd: float = 0.5
    beta_def_mu: float = 0.3
    beta_def_sd: float = 0.3
    log_c1_mu: float = math.log(0.5)
    log_c1_sd: float = 1.0
    log_gap_mu: float = math.log(1.5)
    log_gap_sd: float = 1.0

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ModelPriors:
        """Read priors from a config mapping, defaulting any missing key.

        Args:
            params: Mapping whose keys are field names.

        Returns:
            The priors.
        """
        defaults = cls()
        values = {
            name: float(params.get(name, getattr(defaults, name)))
            for name in cls.__dataclass_fields__
        }
        return cls(**values)
