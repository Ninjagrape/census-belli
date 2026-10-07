"""The joint imputation model, marginalised analytically.

Imputation here has one missing variable (log troops) and every other field is
a covariate, so chained equations would reduce to a single conditional model.
That model is fitted jointly instead, for a log troop count ``y_s`` per side::

    y_s = a_era[e_b] + beta_out[o_s] + beta_type[t_b] + beta_def[d_s]
          + v_polity[p_s] + c_cmd[k_s] + u_b + eps_s
    u_b ~ N(0, tau_b),  eps_s ~ N(0, sigma)

Observed sides arrive as a likelihood from reconcile, ``m_lik ~ N(y, P^-1)``
(:mod:`pipeline.imputers.likelihood`). Neither ``u_b`` nor any per-side latent
is sampled: for the observed sides of one battle the marginal is exactly::

    m_lik ~ MvNormal(X beta,  tau_b^2 J + sigma^2 I + P^-1)

which handles two-sided missingness (a battle with no observed side simply
contributes nothing) and keeps the sampler to hyperparameters and group
effects only. Missing and latent values are drawn afterwards from the
conditional Gaussian, in :mod:`pipeline.imputers.conditional`.

Battles are grouped by their number of observed sides and each group is one
batched ``MvNormal``. Every per-observation array enters as ``pm.Data``
(handover.md 20.2 item 7: numpy constants make nutpie's compile time explode),
and no variable is named like a dimension (20.2 item 1).

pymc is imported inside the functions, not at module scope, for the reason
given in ``pipeline/reconcilers/model.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from pipeline.imputers.design import ImputeDesign, RandomBlock

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import pymc as pm

__all__ = ["build_model", "is_random_active"]

logger = structlog.get_logger()


def is_random_active(block: RandomBlock) -> bool:
    """Whether a random-effect block is in the model.

    With fewer than two sampled levels the block is a constant shift that the
    era intercept already carries, so it is left out (and contributes zero when
    predicting, see :mod:`pipeline.imputers.conditional`).

    Args:
        block: The block.

    Returns:
        True if at least two levels have observed sides.
    """
    return len(block.sampled) >= 2


def _data(pm_module: Any, name: str, values: np.ndarray) -> Any:
    """Register a per-observation array as ``pm.Data``, never a constant."""
    return pm_module.Data(name, values)


def build_model(design: ImputeDesign) -> pm.Model:
    """Build the marginalised joint model.

    Args:
        design: The aligned arrays from
            :func:`pipeline.imputers.design.build_design`.

    Returns:
        An unsampled :class:`pymc.Model`.

    Raises:
        ImportError: If pymc is not installed.
        ValueError: If no side has a likelihood, so the model would sample
            nothing but its priors.
    """
    import pymc as pm  # noqa: PLC0415 - deliberately deferred; see module docstring
    import pytensor.tensor as pt  # noqa: PLC0415

    if design.n_observed == 0:
        raise ValueError("no observed sides: the imputation model would sample only its priors")

    hyper = design.hyper
    coords: dict[str, Any] = {"era": list(design.eras)}
    for block in (design.outcome, design.battle_type, design.defender):
        if block.is_active:
            coords[f"{block.name}_level"] = list(block.levels)
    for random_block in (design.polity, design.commander):
        if is_random_active(random_block):
            coords[f"{random_block.name}_key"] = list(random_block.sampled)

    obs = design.observed_rows

    with pm.Model(coords=coords) as model:
        a_era = pm.Normal("a_era", mu=design.era_mu, sigma=design.era_sd, dims="era")
        eta = a_era[_data(pm, "data_era_index", design.era_index[obs])]

        for name, block in (
            ("beta_out", design.outcome),
            ("beta_type", design.battle_type),
            ("beta_def", design.defender),
        ):
            if not block.is_active:
                continue
            effect = pm.ZeroSumNormal(name, sigma=block.sd, dims=f"{block.name}_level")
            eta = eta + effect[_data(pm, f"data_{name}_index", block.index[obs])]

        for tau_name, z_name, effect_name, random_block in (
            ("tau_p", "z_polity", "v_polity", design.polity),
            ("tau_c", "z_cmd", "c_cmd", design.commander),
        ):
            if not is_random_active(random_block):
                continue
            tau = pm.HalfNormal(tau_name, sigma=hyper.get(tau_name, 0.5))
            z = pm.Normal(z_name, 0.0, 1.0, dims=f"{random_block.name}_key")
            effect = pm.Deterministic(effect_name, tau * z, dims=f"{random_block.name}_key")
            eta = eta + effect[_data(pm, f"data_{effect_name}_index", random_block.index[obs])]

        tau_b = pm.HalfNormal("tau_b", sigma=hyper.get("tau_b", 1.0))
        sigma = pm.HalfNormal("sigma", sigma=hyper.get("sigma", 1.0))

        for group in design.groups:
            k = group.k
            mu = eta[_data(pm, f"data_pos_k{k}", group.positions)]
            y = _data(pm, f"data_y_k{k}", group.y)
            measurement = _data(pm, f"data_cov_k{k}", group.cov)
            if k == 1:
                pm.Normal(
                    f"obs_k{k}",
                    mu=mu[:, 0],
                    sigma=pt.sqrt(tau_b**2 + sigma**2 + measurement[:, 0, 0]),
                    observed=y[:, 0],
                )
                continue
            cov = tau_b**2 * np.ones((k, k)) + sigma**2 * np.eye(k) + measurement
            pm.MvNormal(f"obs_k{k}", mu=mu, cov=cov, observed=y)

    logger.info(
        "impute_model_built",
        n_sides=design.n_sides,
        n_observed=design.n_observed,
        group_sizes={g.k: len(g.positions) for g in design.groups},
        polities=len(design.polity.sampled),
        commanders=len(design.commander.sampled),
    )
    return model
