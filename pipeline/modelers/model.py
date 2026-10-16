"""The Bradley-Terry / ordinal-logit model in PyMC (design sections 3, 4, 10; R1, R2).

Per side a score ``S_side``::

    named        S = sum_i w_i * skill_g(i)            skill = sigma_skill * z
    unknown      S = mu_unknown      + sigma_skill * z_side
    unattributed S = mu_unattributed + sigma_skill * z_side

A merged camp (3+ side battle) is the general case: ``S = sum_i w_i skill_i +
u_k (mu_unknown + sigma_skill z_side) + u_a (mu_unattributed + sigma_skill
z_side)`` with ``sum_i w_i + u_k + u_a = 1``: the troop share of unknown or
unattributed members stays a latent component and is never handed to the
named commanders. Ordinary sides are the ``u = 0`` and ``w`` summing to 1, or
``u = 1`` and no commanders, cases.

and per battle ``eta = S_A - S_B + beta_force_g * x_diff + beta_def * d_diff``
(every term a difference of side A minus side B, so swapping the sides negates
``eta``) with a cumulative logit over five levels and symmetric cutpoints
(:mod:`pipeline.modelers.ordinal`). Outcomes are interval-censored: an exact
outcome has ``k_lo == k_hi``, a same-class grade disagreement spans two levels.

``build_joint_latent_model`` (R2) replaces the fixed ``x_diff`` by a latent
log-troop difference per battle with a measurement prior, in one fit with no
imputation; it is the comparator for the circularity check.

``build_model`` and ``build_joint_latent_model`` also take an optional per-battle
``loglik_weight`` for held-out cross-validation: weight 0 drops a battle's outcome
from the likelihood while its sides and generals stay in the graph, so they sample
from their prior predictive.
Without it the graph is exactly the unweighted one.

pymc is imported inside the functions, as in ``pipeline/imputers/model.py``.
Every per-observation array is a ``pm.Data`` (numpy constants make nutpie's
compile time explode, handover-log 20.2 item 7).

Variable names: ``sigma_skill``, ``z_skill``, ``skill`` (dim ``general``),
``mu_unknown``, ``mu_unattributed``, ``z_side`` (dim ``latent_side``),
``beta_force`` (the land slope), ``beta_def`` (only when the design has the
defender term), ``delta_water_air`` and ``beta_force_group`` (dim
``force_group``, land then water/air; only when the design has a water/air
group: ``beta_force_g = beta_force + delta_water_air * is_water_air``, with the
skill population shared across groups; amphibious battles are land),
``log_c1``, ``log_gap``, ``c1``, ``c2``; the joint model adds ``z_x`` and
``x_diff_latent`` (one per battle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from pipeline.modelers.ordinal import PROB_FLOOR
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import FORCE_GROUP_LABELS, ModelDesign

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az
    import pymc as pm

__all__ = [
    "PosteriorDraws",
    "assemble_side_scores",
    "build_joint_latent_model",
    "build_model",
    "extract_draws",
]

logger = structlog.get_logger()


def _data(pm_module: Any, name: str, values: np.ndarray) -> Any:
    return pm_module.Data(name, values)


def _check_loglik_weight(
    design: ModelDesign, loglik_weight: np.ndarray | None
) -> np.ndarray | None:
    """Validate an optional per-battle likelihood weight; None passes through."""
    if loglik_weight is None:
        return None
    weight = np.asarray(loglik_weight, dtype=float)
    if weight.shape != (design.n_battles,):
        raise ValueError(f"loglik_weight has shape {weight.shape}, expected ({design.n_battles},)")
    if not np.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
        raise ValueError("loglik_weight must be finite and within [0, 1]")
    return weight


def build_model(
    design: ModelDesign,
    x_diff: np.ndarray,
    priors: ModelPriors,
    *,
    loglik_weight: np.ndarray | None = None,
) -> pm.Model:
    """Build the model with the log-troop difference fixed at ``x_diff``.

    Args:
        design: Aligned arrays from :func:`pipeline.modelers.design.build_design`.
        x_diff: ``x_A - x_B`` per battle, shape ``(n_battles,)``. Held in a
            ``pm.Data`` named ``x_diff``.
        priors: Hyperparameters.
        loglik_weight: Optional weight per battle on its log-likelihood term,
            shape ``(n_battles,)``, finite, in ``[0, 1]``. Held in a ``pm.Data``
            named ``data_loglik_weight``. None leaves the graph unweighted.

    Returns:
        An unsampled :class:`pymc.Model`.

    Raises:
        ImportError: If pymc is not installed.
        ValueError: If the design has no battles, ``x_diff`` has the wrong
            length, or ``loglik_weight`` has the wrong shape or values.
    """
    _check_design(design)
    if x_diff.shape != (design.n_battles,):
        raise ValueError(f"x_diff has shape {x_diff.shape}, expected ({design.n_battles},)")
    weight = _check_loglik_weight(design, loglik_weight)
    return _build(design, priors, x_diff=x_diff, measurement=None, loglik_weight=weight)


def joint_x_diff_prior(
    design: ModelDesign, obs_mean: np.ndarray, obs_sd: np.ndarray, sigma_w: float
) -> tuple[np.ndarray, np.ndarray]:
    """Prior mean and sd of ``x_A - x_B`` per battle for the joint model.

    Only the difference enters the likelihood, so the prior is put on it
    directly and absolute troop levels (era means) drop out. Putting an
    absolute era prior on a missing side and a measurement on its opponent
    would invent a force ratio from the gap between the two scales.

    - both sides measured: ``N(m_A - m_B, sqrt(sd_A^2 + sd_B^2))``
    - one side measured: ``N(0, sqrt(sd_obs^2 + 2 sigma_w^2))``
    - neither: ``N(0, sqrt(2) sigma_w)``

    Args:
        design: The design.
        obs_mean: Observed log-troop mean per side index, NaN where missing.
        obs_sd: Observed log-troop sd per side index, NaN where missing.
        sigma_w: Within-battle sd of log troops (a side's deviation from the
            battle's common level).

    Returns:
        ``(mu, sd)``, each of shape ``(n_battles,)``.
    """
    observed = np.isfinite(obs_mean) & np.isfinite(obs_sd) & (obs_sd > 0)
    a_obs, b_obs = observed[design.side_a], observed[design.side_b]
    sd_a = np.where(a_obs, obs_sd[design.side_a], 0.0)
    sd_b = np.where(b_obs, obs_sd[design.side_b], 0.0)
    both = a_obs & b_obs
    mu = np.where(both, obs_mean[design.side_a] - obs_mean[design.side_b], 0.0)
    # Any missing side adds 2 sigma_w^2 once: the deviations of the two sides
    # from the battle's common level cancel only when both are observed.
    variance = sd_a**2 + sd_b**2 + np.where(both, 0.0, 2.0 * sigma_w**2)
    return mu, np.sqrt(variance)


def build_joint_latent_model(
    design: ModelDesign,
    obs_mean: np.ndarray,
    obs_sd: np.ndarray,
    sigma_w: float,
    priors: ModelPriors,
    *,
    loglik_weight: np.ndarray | None = None,
) -> pm.Model:
    """Build the joint model with a latent log-troop difference per battle (R2).

    The latent is ``x_diff_latent = mu_b + sd_b * z_b`` per battle (see
    :func:`joint_x_diff_prior`); no absolute troop level is modelled.

    Args:
        design: The design.
        obs_mean: Observed log-troop mean per side index, NaN where missing.
        obs_sd: Observed log-troop sd per side index, NaN where missing.
        sigma_w: Within-battle sd of log troops, positive.
        priors: Hyperparameters.
        loglik_weight: Optional weight per battle on its log-likelihood term,
            as in :func:`build_model`. A battle with weight 0 keeps its latent
            ``x_diff_latent`` in the graph, where it samples the measurement
            prior. None leaves the graph unweighted.

    Returns:
        An unsampled :class:`pymc.Model`.

    Raises:
        ValueError: On a length mismatch, a non-positive ``sigma_w`` or an
            invalid ``loglik_weight``.
    """
    _check_design(design)
    if any(a.shape != (design.n_sides,) for a in (obs_mean, obs_sd)):
        raise ValueError(f"measurement arrays must have shape ({design.n_sides},)")
    if not sigma_w > 0:
        raise ValueError("sigma_w must be positive")
    weight = _check_loglik_weight(design, loglik_weight)
    return _build(
        design,
        priors,
        x_diff=None,
        measurement=joint_x_diff_prior(design, obs_mean, obs_sd, sigma_w),
        loglik_weight=weight,
    )


def _check_design(design: ModelDesign) -> None:
    if design.n_battles == 0 or design.n_generals == 0:
        raise ValueError("no included battles: the model would sample only its priors")


def _build(
    design: ModelDesign,
    priors: ModelPriors,
    *,
    x_diff: np.ndarray | None,
    measurement: tuple[np.ndarray, np.ndarray] | None,
    loglik_weight: np.ndarray | None = None,
) -> pm.Model:
    import pymc as pm  # noqa: PLC0415 - deliberately deferred; see module docstring
    import pytensor.tensor as pt  # noqa: PLC0415

    cmd_index, cmd_weight = design.commander_padded()
    latent = design.latent_sides
    latent_slot = np.full(design.n_sides, -1, dtype=np.int64)
    latent_slot[latent] = np.arange(latent.size)
    unknown_share = design.side_unknown_share
    unattributed_share = design.side_unattributed_share
    latent_scale = unknown_share + unattributed_share
    n = design.n_battles
    coords: dict[str, Any] = {"general": [int(g) for g in design.general_ids]}
    if design.n_force_groups > 1:
        coords["force_group"] = list(FORCE_GROUP_LABELS)
    if latent.size:
        coords["latent_side"] = [int(design.side_ids[i]) for i in latent]

    with pm.Model(coords=coords) as model:
        sigma_skill = pm.HalfNormal("sigma_skill", sigma=priors.skill_prior_sd)
        z_skill = pm.Normal("z_skill", 0.0, 1.0, dims="general")
        skill = pm.Deterministic("skill", sigma_skill * z_skill, dims="general")
        scores = pt.sum(
            skill[_data(pm, "data_cmd_index", cmd_index)]
            * _data(pm, "data_cmd_weight", cmd_weight),
            axis=1,
        )

        if latent.size:
            z_side = pm.Normal("z_side", 0.0, 1.0, dims="latent_side")
            padded = pt.concatenate([pt.zeros(1), z_side])
            offset = (
                sigma_skill
                * padded[_data(pm, "data_latent_slot", latent_slot + 1)]
                * _data(pm, "data_latent_scale", latent_scale)
            )
            if unknown_share.any():
                mu_unknown = pm.Normal("mu_unknown", 0.0, priors.mu_unknown_sd)
                offset = offset + mu_unknown * _data(pm, "data_unknown_share", unknown_share)
            if unattributed_share.any():
                mu_unattributed = pm.Normal("mu_unattributed", 0.0, priors.mu_unattributed_sd)
                offset = offset + mu_unattributed * _data(
                    pm, "data_unattributed_share", unattributed_share
                )
            scores = scores + offset

        beta_force = pm.Normal("beta_force", priors.beta_force_mu, priors.beta_force_sd)
        force_slope: Any = beta_force
        if design.n_force_groups > 1:
            # A fixed offset from the land slope: beta_force stays the
            # identified land slope that the circularity gate reads, and a
            # variance parameter on two groups would not be identified.
            delta = pm.Normal("delta_water_air", 0.0, priors.delta_water_air_sd)
            group_slope = pm.Deterministic(
                "beta_force_group",
                pt.stack([beta_force, beta_force + delta]),
                dims="force_group",
            )
            force_slope = group_slope[_data(pm, "data_force_group", design.force_group)]
        side_a = _data(pm, "data_side_a", design.side_a)
        side_b = _data(pm, "data_side_b", design.side_b)
        if x_diff is not None:
            x_term = _data(pm, "x_diff", x_diff)
        else:
            assert measurement is not None
            z_x = pm.Normal("z_x", 0.0, 1.0, shape=n)
            x_term = pm.Deterministic(
                "x_diff_latent",
                _data(pm, "data_x_mu", measurement[0])
                + _data(pm, "data_x_sd", measurement[1]) * z_x,
            )
        eta = scores[side_a] - scores[side_b] + force_slope * x_term
        if design.defender_term_used:
            beta_def = pm.Normal("beta_def", priors.beta_def_mu, priors.beta_def_sd)
            eta = eta + beta_def * _data(pm, "data_defender_diff", design.defender_diff)

        log_c1 = pm.Normal("log_c1", priors.log_c1_mu, priors.log_c1_sd)
        log_gap = pm.Normal("log_gap", priors.log_gap_mu, priors.log_gap_sd)
        c1 = pm.Deterministic("c1", pt.exp(log_c1))
        c2 = pm.Deterministic("c2", c1 + pt.exp(log_gap))
        cuts = pt.stack([-c2, -c1, c1, c2])

        cdf = pt.sigmoid(cuts[None, :] - eta[:, None])
        cdf = pt.concatenate([pt.zeros((n, 1)), cdf, pt.ones((n, 1))], axis=1)
        rows = pt.arange(n)
        prob = (
            cdf[rows, _data(pm, "data_k_hi", design.k_hi) + 1]
            - cdf[rows, _data(pm, "data_k_lo", design.k_lo)]
        )
        log_prob = pt.log(pt.maximum(prob, PROB_FLOOR))
        if loglik_weight is not None:
            log_prob = _data(pm, "data_loglik_weight", loglik_weight) * log_prob
        pm.Potential("outcome_loglik", log_prob.sum())

    logger.info(
        "model_built",
        n_battles=n,
        n_generals=design.n_generals,
        n_latent_sides=int(latent.size),
        defender_term=design.defender_term_used,
        force_groups=design.n_force_groups,
        joint_latent=measurement is not None,
    )
    return model


@dataclass(frozen=True)
class PosteriorDraws:
    """Flattened posterior draws (chains and draws merged, ``T`` in total).

    Attributes:
        skills: Raw (uncentred) skill, shape ``(T, n_generals)``.
        side_scores: ``S`` per side, shape ``(T, n_sides)``.
        beta_force: Land force slope, shape ``(T,)``.
        beta_def: Shape ``(T,)``; zeros when the model has no defender term.
        c1: Shape ``(T,)``.
        c2: Shape ``(T,)``.
        sigma_skill: Shape ``(T,)``.
        x_diff: Per-draw ``x_A - x_B`` ``(T, n_battles)`` for the joint
            model, else None (the caller holds the fixed ``x_diff``).
        beta_force_group: Per-group slopes ``(T, n_force_groups)``, land
            then water/air, or None when the design has a single slope.
        delta_water_air: Water/air offset ``(T,)``, or None without that group.
        mu_unknown: ``(T,)``, or None when no side carries an unknown share.
    """

    skills: np.ndarray
    side_scores: np.ndarray
    beta_force: np.ndarray
    beta_def: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    sigma_skill: np.ndarray
    x_diff: np.ndarray | None = None
    beta_force_group: np.ndarray | None = None
    delta_water_air: np.ndarray | None = None
    mu_unknown: np.ndarray | None = None

    @property
    def n_draws(self) -> int:
        """Total number of draws."""
        return int(self.skills.shape[0])

    def force_slopes(self, design: ModelDesign) -> np.ndarray:
        """The force slope each battle uses.

        Args:
            design: The design the draws were fitted on.

        Returns:
            ``(T,)`` when one slope serves every battle, else ``(T, n_battles)``.
        """
        if self.beta_force_group is None or design.n_force_groups == 1:
            return self.beta_force
        return np.asarray(self.beta_force_group[:, design.force_group])


def assemble_side_scores(
    design: ModelDesign,
    skills: np.ndarray,
    sigma_skill: np.ndarray,
    z_side: np.ndarray | None,
    mu_unknown: np.ndarray | None,
    mu_unattributed: np.ndarray | None,
) -> np.ndarray:
    """Rebuild ``S`` per side and draw from posterior pieces, in numpy.

    Args:
        design: The design.
        skills: ``(T, n_generals)``.
        sigma_skill: ``(T,)``.
        z_side: ``(T, n_latent_sides)`` or None when there are no latent sides.
        mu_unknown: ``(T,)`` or None.
        mu_unattributed: ``(T,)`` or None.

    Returns:
        ``(T, n_sides)`` side scores.
    """
    n_draws = skills.shape[0]
    scores = np.zeros((n_draws, design.n_sides))
    for side, general, weight in zip(
        design.cmd_side, design.cmd_general, design.cmd_weight, strict=True
    ):
        scores[:, side] += weight * skills[:, general]
    latent = design.latent_sides
    if latent.size and z_side is not None:
        scale = (design.side_unknown_share + design.side_unattributed_share)[latent]
        scores[:, latent] += sigma_skill[:, None] * z_side * scale[None, :]
    for share, mu in (
        (design.side_unknown_share, mu_unknown),
        (design.side_unattributed_share, mu_unattributed),
    ):
        if mu is not None and share.any():
            scores += mu[:, None] * share[None, :]
    return scores


def extract_draws(idata: az.InferenceData, design: ModelDesign) -> PosteriorDraws:
    """Flatten a fit's posterior into the arrays WAR and pooling consume.

    Args:
        idata: A fit from :func:`pipeline.reconcilers.model.sample`.
        design: The design the model was built from.

    Returns:
        The flattened draws.
    """
    post = idata.posterior

    def flat(name: str) -> np.ndarray:
        values = np.asarray(post[name].values, dtype=float)
        return values.reshape(-1, *values.shape[2:])

    def optional(name: str) -> np.ndarray | None:
        return flat(name) if name in post else None

    skills = flat("skill")
    sigma = flat("sigma_skill")
    scores = assemble_side_scores(
        design,
        skills,
        sigma,
        optional("z_side"),
        optional("mu_unknown"),
        optional("mu_unattributed"),
    )
    beta_def = optional("beta_def")
    x_diff = optional("x_diff_latent")
    return PosteriorDraws(
        skills=skills,
        side_scores=scores,
        beta_force=flat("beta_force"),
        beta_def=np.zeros_like(sigma) if beta_def is None else beta_def,
        c1=flat("c1"),
        c2=flat("c2"),
        sigma_skill=sigma,
        x_diff=x_diff,
        beta_force_group=optional("beta_force_group"),
        delta_water_air=optional("delta_water_air"),
        mu_unknown=optional("mu_unknown"),
    )
