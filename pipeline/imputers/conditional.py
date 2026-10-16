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
from pipeline.imputers.likelihood import SharedError
from pipeline.imputers.model import is_random_active

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

# Third seed component of the reconcile-draw stream, kept apart from the main one.
_THETA_STREAM = 1

__all__ = [
    "PosteriorSamples",
    "RoundSet",
    "apply_mnar_delta",
    "conditional_draw",
    "draw_rounds",
    "era_loading_table",
    "extract_posterior",
    "select_draw_indices",
    "select_theta_indices",
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
        theta_index: The reconcile draw each round took its shared parameters
            from, or None when the shared error is not propagated.
    """

    values: np.ndarray
    is_imputed: np.ndarray
    draw_index: tuple[int, ...]
    theta_index: tuple[int, ...] | None = None


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


def select_theta_indices(theta_centered: np.ndarray, n_rounds: int, seed: int) -> list[int]:
    """Latin hypercube choice of one reconcile draw per round.

    Independent uniform picks leave the round-to-round spread of theta noisy at
    the 20 or so rounds Rubin pooling uses, which is spread the pooled variance
    inherits. Sorting by the leading component and taking one draw from each of
    ``n_rounds`` equal-probability strata covers it without gaps. The position
    inside a stratum is uniform, not the stratum midpoint: midpoints understate
    theta's between-round variance by 6 to 12%, which Rubin's between-imputation
    term then inherits. A permutation seeded by ``seed`` decides which round
    gets which stratum, so rounds are not ordered by theta.

    Args:
        theta_centered: ``(n_draws, p)`` centred shared parameters.
        n_rounds: Rounds wanted.
        seed: Base seed; the permutation and the within-stratum jitter use
            ``default_rng([seed, 1])``.

    Returns:
        ``n_rounds`` row indices into ``theta_centered``, round ``j`` using the
        ``j``-th. Deterministic per seed.
    """
    n_draws = len(theta_centered)
    order = np.argsort(theta_centered[:, 0], kind="stable")
    rng = np.random.default_rng([seed, _THETA_STREAM])
    permutation = rng.permutation(n_rounds)
    jitter = rng.uniform(size=n_rounds)
    positions = np.minimum(
        ((np.arange(n_rounds) + jitter) * n_draws / n_rounds).astype(int), n_draws - 1
    )
    strata = order[positions]
    return [int(i) for i in strata[permutation]]


def era_loading_table(design: ImputeDesign) -> np.ndarray | None:
    """Precision-weighted mean likelihood loading of the observed sides, per era.

    An imputed side, or a side too weak to carry a likelihood, has no data to
    shift with theta, but its prior mean is the era level, which was learned
    from observed sides whose estimates do shift. This table is how much, per
    era, those observed sides move per unit of centred theta.

    Two approximations, both deliberate. Weighting by each side's precision
    over-weights precise sides relative to the impute model's own era-level
    weighting, which is a different average of the same sides. An era with no
    observed side gets a zero shift, so its imputed sides do not move with
    theta even though their true level plausibly would.

    Args:
        design: The design, with each battle's likelihood and its loading.

    Returns:
        ``(n_eras, p)``, zero for an era with no observed side; None when no
        battle carries a loading (the shared error is not propagated).
    """
    n_eras = len(design.eras)
    total: np.ndarray | None = None
    weight = np.zeros(n_eras)
    for battle in design.battles:
        if battle.shift_loading is None or not len(battle.obs_local):
            continue
        if total is None:
            total = np.zeros((n_eras, battle.shift_loading.shape[1]))
        eras = design.era_index[battle.rows[battle.obs_local]]
        for j, era in enumerate(eras):
            w = float(battle.precision[j, j])
            total[era] += w * battle.shift_loading[j]
            weight[era] += w
    if total is None:
        return None
    table: np.ndarray = np.divide(
        total, weight[:, None], out=np.zeros_like(total), where=weight[:, None] > 0
    )
    return table


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
    theta_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Draw every side's latent log troops given the likelihood, per battle.

    Args:
        design: The design, with each battle's likelihood.
        means: Per-side covariate means from :func:`side_means`.
        tau_b: Between-battle sd in this draw.
        sigma: Within-battle sd in this draw.
        rng: Random generator.
        theta_shift: This round's centred shared reconcile parameters, ``(p,)``.
            Every battle's likelihood mean moves by its loading times this, so
            battles in one round share the same reconcile error.

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
            m_lik = battle.m_lik
            if theta_shift is not None and battle.shift_loading is not None:
                m_lik = m_lik + battle.shift_loading @ theta_shift
            rhs[ix] += battle.precision @ m_lik
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
    shared: SharedError | None = None,
) -> RoundSet:
    """Produce every imputation round from the posterior.

    Args:
        design: The design.
        post: Pooled posterior samples.
        n_rounds: Number of rounds.
        seed: Base seed; round ``j`` uses ``default_rng([seed, j])``.
        mnar_delta: See :func:`apply_mnar_delta`.
        shared: Reconcile's shared parameter draws. When given, round ``j``
            picks one reconcile draw from its own stream
            a stratified choice (:func:`select_theta_indices`; its
            permutation is separate from the stream the conditional draws use,
            so switching this on does not change them). Every battle's
            likelihood mean is shifted for it, and so is every side's prior
            mean by its era's mean loading (:func:`era_loading_table`), so an
            imputed side co-moves with the observed sides it is imputed from.

    Returns:
        The rounds.
    """
    indices = select_draw_indices(post.n_chains, post.n_draws, n_rounds)
    values = np.empty((n_rounds, design.n_sides))
    theta_index = (
        [] if shared is None else select_theta_indices(shared.theta_centered, n_rounds, seed)
    )
    era_table = era_loading_table(design) if shared is not None else None
    for j, flat in enumerate(indices):
        rng = np.random.default_rng([seed, j])
        theta_shift = None
        if shared is not None:
            theta_shift = shared.theta_centered[theta_index[j]]
        means = side_means(post, flat, design, rng)
        if theta_shift is not None and era_table is not None:
            # Every side, not only the unobserved: an observed side's likelihood
            # shifts by its own loading, and shifting its prior mean as well keeps
            # the prior-coupled difference to an imputed neighbour from moving.
            means = means + era_table[design.era_index] @ theta_shift
        values[j] = conditional_draw(
            design, means, float(post.tau_b[flat]), float(post.sigma[flat]), rng, theta_shift
        )
    is_imputed = ~design.is_observed
    values = apply_mnar_delta(values, is_imputed, design.mnar, mnar_delta)
    return RoundSet(
        values=values,
        is_imputed=is_imputed,
        draw_index=tuple(indices),
        theta_index=tuple(theta_index) if shared is not None else None,
    )
