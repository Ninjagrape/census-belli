"""Draws from the fitted model: one complete dataset per imputation round.

Each round uses one posterior draw of the parameters. Within that draw every
side, observed or missing, gets its latent log troop count from the
conditional Gaussian::

    y_battle ~ N(mu, tau_b^2 J + sigma^2 I)                 (prior)
    m_lik    ~ N(y_obs, P^-1)                               (observed sides)

so the battle's posterior precision is ``prior_precision + H' P H`` and its mean
solves ``precision * mean = prior_precision * mu + H' P m_lik``. Sides with no
likelihood (missing, or too weakly measured) are simply not in ``H``.

Observed sides are redrawn rather than copied through: this is
overimputation. Treating an observed side's noisy reconcile estimate as exact
puts its measurement error into the downstream force ratio as if it were
signal, and a regression of outcome on that ratio is attenuated. Drawing the
latent value given the observation spreads that error the way the data do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from pipeline.imputers.design import ImputeDesign, RandomBlock
from pipeline.imputers.model import is_random_active

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

__all__ = [
    "PosteriorSamples",
    "RoundSet",
    "apply_mnar_delta",
    "conditional_draw",
    "draw_rounds",
    "extract_posterior",
    "select_draw_indices",
    "side_means",
]


@dataclass(frozen=True)
class PosteriorSamples:
    """Posterior draws pooled over chains: first axis is ``chain * n_draws + draw``.

    Absent (inactive) blocks are None and contribute zero.
    """

    a_era: np.ndarray
    beta_out: np.ndarray | None
    beta_type: np.ndarray | None
    beta_def: np.ndarray | None
    v_polity: np.ndarray | None
    c_cmd: np.ndarray | None
    tau_p: np.ndarray | None
    tau_c: np.ndarray | None
    tau_b: np.ndarray
    sigma: np.ndarray
    n_chains: int
    n_draws: int


@dataclass(frozen=True)
class RoundSet:
    """All imputation rounds for one scale.

    Attributes:
        values: ``(n_rounds, n_sides)`` log troop counts, MNAR delta applied.
        is_imputed: Per side, True when it had no likelihood.
        draw_index: The pooled posterior-draw index each round used.
    """

    values: np.ndarray
    is_imputed: np.ndarray
    draw_index: tuple[int, ...]


def _pooled(idata: az.InferenceData, name: str) -> np.ndarray:
    """Pool one posterior variable's chain and draw axes into one."""
    values = np.asarray(idata.posterior[name].values, dtype=float)
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def extract_posterior(idata: az.InferenceData, design: ImputeDesign) -> PosteriorSamples:
    """Pull the arrays the conditional draws need out of an InferenceData.

    Args:
        idata: The sampled posterior.
        design: The design the model was built from; decides which blocks are
            present.

    Returns:
        The pooled samples.
    """
    shape = idata.posterior["tau_b"].values.shape

    def optional(name: str, active: bool) -> np.ndarray | None:
        return _pooled(idata, name) if active else None

    polity_active = is_random_active(design.polity)
    commander_active = is_random_active(design.commander)
    return PosteriorSamples(
        a_era=_pooled(idata, "a_era"),
        beta_out=optional("beta_out", design.outcome.is_active),
        beta_type=optional("beta_type", design.battle_type.is_active),
        beta_def=optional("beta_def", design.defender.is_active),
        v_polity=optional("v_polity", polity_active),
        c_cmd=optional("c_cmd", commander_active),
        tau_p=optional("tau_p", polity_active),
        tau_c=optional("tau_c", commander_active),
        tau_b=_pooled(idata, "tau_b"),
        sigma=_pooled(idata, "sigma"),
        n_chains=int(shape[0]),
        n_draws=int(shape[1]),
    )


def select_draw_indices(n_chains: int, n_draws: int, n_rounds: int) -> list[int]:
    """Choose pooled posterior-draw indices spread across chains and time.

    Rounds are dealt to chains in turn; within a chain they sit at the midpoints
    of equal slices of the draw sequence, so neighbouring rounds are far apart
    and every chain contributes. Deterministic, so a re-run reproduces.

    Args:
        n_chains: Chains in the posterior.
        n_draws: Draws per chain.
        n_rounds: Rounds wanted.

    Returns:
        ``n_rounds`` pooled indices ``chain * n_draws + draw``, all distinct
        provided ``n_rounds <= n_chains * n_draws``.
    """
    per_chain = math.ceil(n_rounds / n_chains)
    indices: list[int] = []
    for j in range(n_rounds):
        chain = j % n_chains
        slot = j // n_chains
        draw = min(n_draws - 1, int((slot + 0.5) * n_draws / per_chain))
        indices.append(chain * n_draws + draw)
    return indices


def _random_effect(
    block: RandomBlock,
    draws: np.ndarray | None,
    tau: float,
    flat: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-side effect from a random block, with prior draws for unseen levels.

    Args:
        block: The block.
        draws: Pooled effect samples, or None when the block is inactive.
        tau: This draw's scale.
        flat: Pooled draw index.
        rng: Source for prior draws.

    Returns:
        A vector of length ``n_sides``; zeros when the block is inactive.
    """
    if draws is None:
        return np.zeros(len(block.index))
    # One fresh draw per level, shared by every side with that level, so two
    # missing sides of the same unseen commander get the same effect.
    fresh = tau * rng.standard_normal(len(block.levels))
    sampled_effect = draws[flat]
    effect: np.ndarray = np.where(
        block.index >= 0,
        sampled_effect[np.maximum(block.index, 0)],
        fresh[block.full_index],
    )
    return effect


def side_means(
    post: PosteriorSamples, flat: int, design: ImputeDesign, rng: np.random.Generator
) -> np.ndarray:
    """The covariate part of every side's mean, in one posterior draw.

    Args:
        post: Pooled posterior samples.
        flat: Pooled draw index.
        design: The design.
        rng: Source for prior draws of unseen random-effect levels.

    Returns:
        ``(n_sides,)`` means of ``y_s`` before the battle effect and noise.
    """
    mean: np.ndarray = post.a_era[flat][design.era_index].copy()
    for samples, block in (
        (post.beta_out, design.outcome),
        (post.beta_type, design.battle_type),
        (post.beta_def, design.defender),
    ):
        if samples is not None:
            mean += samples[flat][block.index]
    tau_p = 0.0 if post.tau_p is None else float(post.tau_p[flat])
    tau_c = 0.0 if post.tau_c is None else float(post.tau_c[flat])
    mean += _random_effect(design.polity, post.v_polity, tau_p, flat, rng)
    mean += _random_effect(design.commander, post.c_cmd, tau_c, flat, rng)
    return mean


def conditional_draw(
    design: ImputeDesign,
    means: np.ndarray,
    tau_b: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw every side's latent log troops given the likelihood, per battle.

    Args:
        design: The design, with each battle's likelihood.
        means: Per-side covariate means from :func:`side_means`.
        tau_b: Between-battle sd in this draw.
        sigma: Within-battle sd in this draw.
        rng: Random generator.

    Returns:
        ``(n_sides,)`` latent log troop counts.
    """
    out = np.empty(design.n_sides)
    tau_sq = tau_b**2
    sigma_sq = sigma**2
    for battle in design.battles:
        mu = means[battle.rows]
        n = len(mu)
        # Sherman-Morrison inverse of sigma^2 I + tau^2 J, in closed form.
        shrink = tau_sq / (sigma_sq + n * tau_sq)
        prior_precision = (np.eye(n) - shrink * np.ones((n, n))) / sigma_sq
        precision = prior_precision
        rhs = prior_precision @ mu
        if len(battle.obs_local):
            ix = battle.obs_local
            precision = prior_precision.copy()
            precision[np.ix_(ix, ix)] += battle.precision
            rhs = rhs.copy()
            rhs[ix] += battle.precision @ battle.m_lik
        chol = np.linalg.cholesky(precision)
        mean = cho_solve((chol, True), rhs)
        noise = solve_triangular(chol.T, rng.standard_normal(n), lower=False)
        out[battle.rows] = mean + noise
    return out


def apply_mnar_delta(
    values: np.ndarray, is_imputed: np.ndarray, mnar: np.ndarray, delta: float
) -> np.ndarray:
    """Shift imputed values for sides classified missing-not-at-random.

    Primary analysis is MAR (delta 0). A non-zero delta is the sensitivity
    knob: it moves only the sides whose missingness is plausibly informative.

    Args:
        values: Log troop counts, ``(n_sides,)`` or ``(n_rounds, n_sides)``.
        is_imputed: Per side, True if imputed.
        mnar: Per side, True if its missing_data_log class is ``mnar``.
        delta: Added to the log value of imputed MNAR sides.

    Returns:
        A new array; the input is not modified.
    """
    shifted = np.array(values, dtype=float, copy=True)
    if delta != 0.0:
        shifted[..., is_imputed & mnar] += delta
    return shifted


def draw_rounds(
    design: ImputeDesign,
    post: PosteriorSamples,
    *,
    n_rounds: int,
    seed: int,
    mnar_delta: float,
) -> RoundSet:
    """Produce every imputation round from the posterior.

    Args:
        design: The design.
        post: Pooled posterior samples.
        n_rounds: Number of rounds.
        seed: Base seed; round ``j`` uses ``default_rng([seed, j])``.
        mnar_delta: See :func:`apply_mnar_delta`.

    Returns:
        The rounds.
    """
    indices = select_draw_indices(post.n_chains, post.n_draws, n_rounds)
    values = np.empty((n_rounds, design.n_sides))
    for j, flat in enumerate(indices):
        rng = np.random.default_rng([seed, j])
        means = side_means(post, flat, design, rng)
        values[j] = conditional_draw(
            design, means, float(post.tau_b[flat]), float(post.sigma[flat]), rng
        )
    is_imputed = ~design.is_observed
    values = apply_mnar_delta(values, is_imputed, design.mnar, mnar_delta)
    return RoundSet(values=values, is_imputed=is_imputed, draw_index=tuple(indices))
