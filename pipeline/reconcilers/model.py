"""The hierarchical source-disagreement model.

The only module in this package that touches pymc, and it imports it **inside**
the functions rather than at module scope. ``pipeline/orchestrator.py`` imports
every stage module and calls ``check_conforms`` on it before running anything,
so a top-level ``import pymc`` would add about ten seconds to every pipeline
run that merely mentions reconcile, and would make the module unimportable
anywhere pymc is absent. The loader and the design matrix stay importable
either way, which is what lets most of this stage be unit-tested without the
statistical stack.

## The model

For observation ``i`` on the log scale::

    eta_i  = mu_side[s_i] + beta_source[j_i] + g_regime[r_i]
             + delta_level[k_i] + l_lineage[c_i] + a_era * era_i * unlabelled_i
    beta_j = b_type[t_j] + u_j

``mu_side`` is the quantity published. Everything else exists to explain why a
particular source's number differs from it.

## Identifiability, and the rule behind it

**Every reference-coded block in the linear predictor is collinear with the
intercept whenever its reference level has no observations. Pinning a reference
level identifies nothing if that level is not in the data.**

That rule is the whole of this section, and it was learned the hard way: the
first version of this model pinned the anchor regime to zero, and on a corpus
where no report carries the anchor -- which is about 98% of real reports, see
``handover.md`` §19.2 -- the unlabelled offset became the intercept under
another name. The two were confounded to corr -0.993 and ``m0`` sampled at
ess 5.3. Both priors were proper, so the run completed and reported a number;
it was a prior compromise, reproducible by algebra with no likelihood term.

Four devices, each doing a different job:

1. ``b_type`` is a ``ZeroSumNormal``: the source-type means sum to exactly
   zero, so no constant can slide between them and the corpus level. This is a
   hard constraint, a deterministic transform onto the sum-zero subspace, not
   a penalty.
2. ``u_source`` is zero-sum **within each source type**, not across all keys.
   Zero-sum across keys leaves the type-level mean of the deviations free, and
   that mean is the same parameter as ``b_type``: with one key in a type the
   two are exactly confounded. A type holding a single key therefore gets no
   free deviation at all, which is the correct answer and says so.
3. **The rotation.** The observation-weighted mean of the reference-coded
   blocks is what the data identify, so it is what gets sampled: ``level``
   rather than ``m0``, with ``m0`` derived. The ridge then lies along a
   coordinate axis where a diagonal mass matrix can scale it. This is a change
   of coordinates, not of model: posterior means are unchanged and ESS rises
   by more than an order of magnitude.
4. **The anchor survives the rotation.** ``g_regime`` for the anchor regime is
   still exactly zero, so an estimate still means "what a modern scholarly
   reconstruction would say" rather than "the average of this corpus".

**What the rotation does not fix, and cannot.** In an all-unlabelled corpus
nothing informs the split of the corpus level into ``m0`` and the unlabelled
offset. The rotation lets the sampler report that honestly instead of crawling
a ridge; it does not make ``m0`` recoverable. A recovery test must therefore
assert on ``m0 + g_bar``, which the data identify, not on ``m0``, which they do
not. The inflation adjustment on such a corpus is **entirely prior**, and
``g_bar`` is written to the diagnostics so the adjustment is recoverable and
reversible downstream.

## The inflation prior

The non-anchor regimes' ``g_r`` take priors fitted to
``tests/fixtures/gold/reconcile/``, whose distribution is a **two-component
mixture**: a single Gaussian fits neither mode, because Cannae is accepted at
face value while Nicaea runs 10-20x, and those are different generative
stories rather than two tails of one.

The mixture enters as its implied mean and standard deviation rather than as a
``pm.Mixture``. A mixture prior on a scalar that also sits inside a hierarchy
invites label switching and multimodal geometry for no gain: where the data
exist they dominate it, and where they do not the first two moments are the
whole of what the prior contributes.

## Non-centred throughout

Most group-level offsets in this corpus are weakly identified: a source seen
once, a regime with five cases. Centred parameterisations funnel badly under
weak identification, and non-centred costs nothing when the data are
informative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog
import yaml

from pipeline.reconcilers.design import Design

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az
    import pymc as pm

__all__ = [
    "DEFAULT_PRIORS_PATH",
    "InflationPrior",
    "ModelPriors",
    "SamplerSettings",
    "build_model",
    "load_inflation_prior",
    "sample",
]

logger = structlog.get_logger()

DEFAULT_PRIORS_PATH: Final[Path] = Path("config/inflation_priors.yaml")

# Used when no fitted file exists. Deliberately wide, and deliberately not the
# spec's original 0.3: a prior centred on 1.35x inflation cannot represent any
# documented case, and for an ancient battle with no independent anchor the
# prior is the estimate. Running without the fitted file is a fallback and the
# loader says so at warning level.
_FALLBACK_MU: Final[float] = 1.15
_FALLBACK_SD: Final[float] = 0.90

# The rhetorical weight for a regime the gold set did not cover, which in
# practice means `unlabelled`. It is a judgement call, not a measurement: the
# gold set labels every case by construction, so it contains no unlabelled
# rows to fit against. At 0.5 it implies a median inflation of about 2.5x
# applied to every unmarked figure in the corpus. It is the single most
# consequential constant in this stage; see handover.md §19.9.
_UNLABELLED_WEIGHT: Final[float] = 0.5

# The prior on `level` after the rotation. Wider than the spec's
# side_prior_mu_sd, deliberately: before the rotation that prior supplied
# roughly half the identification of the unlabelled offset, and
# "armies are about 10,000 strong" is a spec constant rather than a measured
# quantity. It should not be quietly doing half the debiasing of the corpus.
_LEVEL_PRIOR_SD: Final[float] = 2.0

_TROOPS: Final[str] = "troops"


@dataclass(frozen=True)
class InflationPrior:
    """The fitted claim-regime inflation prior.

    Attributes:
        anchor_regime: The regime pinned to exactly zero bias.
        faithful_mu: Mean of the low-inflation component.
        faithful_sd: Standard deviation of the low-inflation component.
        rhetorical_mu: Mean of the high-inflation component.
        rhetorical_sd: Standard deviation of the high-inflation component.
        regime_weights: Per-regime probability of the rhetorical component.
        era_mu: Prior mean on the ancient-minus-later contrast. Measured from
            the gold set, not derived from the mixture gap: those are different
            quantities and the mixture gap has the opposite sign.
        era_sd: Prior standard deviation on that contrast.
        provenance: The fitted file's provenance block, carried into the run's
            diagnostics so an estimate is traceable to the gold set behind it.
    """

    anchor_regime: str = "modern_scholarly"
    faithful_mu: float = 0.0
    faithful_sd: float = 0.25
    rhetorical_mu: float = _FALLBACK_MU
    rhetorical_sd: float = _FALLBACK_SD
    regime_weights: dict[str, float] = field(default_factory=dict)
    era_mu: float = 0.0
    era_sd: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)

    def moments_for(self, regime: str) -> tuple[float, float]:
        """Mean and standard deviation of the prior on one regime's bias.

        The mixture's first two moments at that regime's fitted weight. The
        anchor is handled by the caller, not here.

        Args:
            regime: A member of
                :data:`pipeline.reconcilers.records.CLAIM_REGIMES`.

        Returns:
            ``(mu, sd)`` for the normal standing in for the mixture.
        """
        weight = self.regime_weights.get(regime, _UNLABELLED_WEIGHT)

        mu = (1.0 - weight) * self.faithful_mu + weight * self.rhetorical_mu
        # Law of total variance: within-component plus between-component.
        within = (1.0 - weight) * self.faithful_sd**2 + weight * self.rhetorical_sd**2
        between = (1.0 - weight) * (self.faithful_mu - mu) ** 2 + weight * (
            self.rhetorical_mu - mu
        ) ** 2
        return mu, float(np.sqrt(within + between))


@dataclass(frozen=True)
class ModelPriors:
    """Scale parameters for the hierarchy, read from the agent spec."""

    side_prior_mu: float = 9.2
    side_prior_mu_sd: float = 1.0
    side_prior_sd_scale: float = 1.5
    bias_prior_sd: float = 1.0
    source_deviation_prior_sd: float = 0.5
    sigma_deviation_prior_sd: float = 0.3
    lineage_prior_sd: float = 0.3
    scope_offset_prior_sd: float = 0.4
    casualty_type_offset_prior_sd: float = 0.7
    scope_unknown_dispersion_sd: float = 0.4
    estimate_dispersion_sd: float = 0.3
    roundness_dispersion_sd: float = 0.3
    precision_prior_alpha: float = 2.0
    precision_prior_beta: float = 1.0
    likelihood_family: str = "student_t"

    @classmethod
    def from_params(cls, params: dict[str, Any], *, quantity: str) -> ModelPriors:
        """Read the scales from an agent spec's params block.

        Args:
            params: The spec's ``params`` mapping.
            quantity: ``"troops"`` or ``"casualties"``; selects the
                side-strength prior, since casualty counts sit an order of
                magnitude below troop totals.

        Returns:
            The priors.
        """
        prefix = "" if quantity == _TROOPS else "casualty_"
        default_mu = 9.2 if quantity == _TROOPS else 7.0
        return cls(
            side_prior_mu=float(params.get(f"{prefix}side_prior_mu", default_mu)),
            side_prior_mu_sd=float(params.get(f"{prefix}side_prior_mu_sd", 1.0)),
            side_prior_sd_scale=float(params.get(f"{prefix}side_prior_sd_scale", 1.5)),
            bias_prior_sd=float(params.get("bias_prior_sd", 1.0)),
            source_deviation_prior_sd=float(params.get("source_deviation_prior_sd", 0.5)),
            sigma_deviation_prior_sd=float(params.get("sigma_deviation_prior_sd", 0.3)),
            lineage_prior_sd=float(params.get("lineage_prior_sd", 0.3)),
            scope_offset_prior_sd=float(params.get("scope_offset_prior_sd", 0.4)),
            casualty_type_offset_prior_sd=float(
                params.get("casualty_type_offset_prior_sd", 0.7)
            ),
            scope_unknown_dispersion_sd=float(params.get("scope_unknown_dispersion_sd", 0.4)),
            estimate_dispersion_sd=float(params.get("estimate_dispersion_sd", 0.3)),
            roundness_dispersion_sd=float(params.get("roundness_dispersion_sd", 0.3)),
            precision_prior_alpha=float(params.get("precision_prior_alpha", 2.0)),
            precision_prior_beta=float(params.get("precision_prior_beta", 1.0)),
            likelihood_family=str(params.get("likelihood_family", "student_t")),
        )


@dataclass(frozen=True)
class SamplerSettings:
    """How to run NUTS."""

    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    random_seed: int = 20260923
    nuts_sampler: str = "nutpie"

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SamplerSettings:
        """Read sampler settings from an agent spec's params block.

        Args:
            params: The spec's ``params`` mapping.

        Returns:
            The settings.
        """
        return cls(
            draws=int(params.get("mcmc_samples", 2000)),
            tune=int(params.get("mcmc_tune", 1000)),
            chains=int(params.get("mcmc_chains", 4)),
            # More than one core respawns the interpreter under pytest on
            # Windows, which turns a test run into a fork bomb.
            cores=int(params.get("mcmc_cores", 1)),
            target_accept=float(params.get("mcmc_target_accept", 0.9)),
            random_seed=int(params.get("random_seed", 20260923)),
            nuts_sampler=str(params.get("nuts_sampler", "nutpie")),
        )


def load_inflation_prior(path: Path = DEFAULT_PRIORS_PATH) -> InflationPrior:
    """Read the fitted inflation prior, falling back to a wide default.

    Args:
        path: The fitted YAML, normally written by
            ``scripts/fit_inflation_prior.py``.

    Returns:
        The prior. When the file is absent a wide fallback is returned and the
        absence is logged at warning level, because an unfitted prior is a
        materially weaker claim and must not pass unremarked.
    """
    if not path.exists():
        logger.warning(
            "reconcile_inflation_prior_missing",
            path=str(path),
            hint="run python -m scripts.fit_inflation_prior",
            fallback_mu=_FALLBACK_MU,
        )
        return InflationPrior()

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mixture = loaded.get("mixture") or {}
    regimes = loaded.get("regimes") or {}
    era = loaded.get("era") or {}
    prior = InflationPrior(
        anchor_regime=str(loaded.get("anchor_regime", "modern_scholarly")),
        faithful_mu=float(mixture.get("faithful_mu", 0.0)),
        faithful_sd=float(mixture.get("faithful_sd", 0.25)),
        rhetorical_mu=float(mixture.get("rhetorical_mu", _FALLBACK_MU)),
        rhetorical_sd=float(mixture.get("rhetorical_sd", _FALLBACK_SD)),
        regime_weights={
            str(name): float((info or {}).get("weight_rhetorical", _UNLABELLED_WEIGHT))
            for name, info in regimes.items()
        },
        era_mu=float(era.get("mu", 0.0)),
        era_sd=float(era.get("sd", 0.5)),
        provenance=dict(loaded.get("provenance") or {}),
    )
    logger.info(
        "reconcile_inflation_prior_loaded",
        path=str(path),
        anchor=prior.anchor_regime,
        regimes=sorted(prior.regime_weights),
        era_mu=prior.era_mu,
        digest=prior.provenance.get("gold_set_digest"),
    )
    return prior


def build_model(
    design: Design,
    *,
    priors: ModelPriors,
    inflation: InflationPrior,
) -> pm.Model:
    """Build the hierarchical source-disagreement model for one quantity.

    Args:
        design: The aligned arrays from
            :func:`pipeline.reconcilers.design.build_design`.
        priors: Hierarchy scales from the agent spec.
        inflation: The fitted claim-regime inflation prior.

    Returns:
        An unsampled :class:`pymc.Model`.

    Raises:
        ImportError: If pymc is not installed. Raised here rather than at
            import time, so the rest of the package stays usable without it.
        ValueError: If the likelihood family is not recognised, or the anchor
            regime is not one the design knows about.
    """
    import pymc as pm  # noqa: PLC0415 - deliberately deferred; see module docstring
    import pytensor.tensor as pt  # noqa: PLC0415

    if inflation.anchor_regime not in design.regimes:
        # design.py forces the reference regime into the vocabulary, so this can
        # only fire when the YAML names a different anchor. Left unchecked it
        # would create a phantom parameter with no observations sampling its
        # prior, on a differently scaled corpus, silently.
        raise ValueError(
            f"Anchor regime {inflation.anchor_regime!r} is not in the design's regimes "
            f"{design.regimes}. config/inflation_priors.yaml and "
            "pipeline.reconcilers.records.CLAIM_REGIMES have diverged."
        )

    coords = design.coords()
    free_regimes = [r for r in design.regimes if r != inflation.anchor_regime]
    coords["regime_free"] = free_regimes
    source_type_of_key = _source_type_of_key(design)

    # The observation-weighted share of each reference-coded level. These are
    # the directions the data identify; see device 3 in the module docstring.
    w_regime = np.bincount(design.regime_index, minlength=len(design.regimes)) / design.n_obs
    w_level = np.bincount(design.level_index, minlength=len(design.levels)) / design.n_obs

    with pm.Model(coords=coords) as model:
        # ── claim regime: the inflation term, anchor pinned to zero ──────────
        g_regime = _regime_bias(pm, pt, design, inflation, free_regimes)
        g_regime = pm.Deterministic("g_regime", g_regime, dims="regime")

        # ── scope or casualty-type offsets ───────────────────────────────────
        delta_level = pm.Deterministic(
            "delta_level", _level_offsets(pm, pt, design, priors), dims="level"
        )

        # ── the rotation ─────────────────────────────────────────────────────
        # Sample the corpus-average log level, which the data identify, and
        # derive m0 from it. Without this, a reference level with no rows makes
        # its block's offset the intercept under another name.
        offset_bar = pm.Deterministic(
            "offset_bar", pt.dot(w_regime, g_regime) + pt.dot(w_level, delta_level)
        )
        level = pm.Normal("level", mu=priors.side_prior_mu, sigma=_LEVEL_PRIOR_SD)
        m0 = pm.Deterministic("m0", level - offset_bar)

        s0 = pm.HalfNormal("s0", sigma=priors.side_prior_sd_scale)
        z_side = pm.Normal("z_side", 0.0, 1.0, dims="side")
        mu_side = pm.Deterministic("mu_side", m0 + s0 * z_side, dims="side")

        # ── source bias: type mean plus within-type deviation ────────────────
        b_type = pm.ZeroSumNormal("b_type", sigma=priors.bias_prior_sd, dims="source_type")
        tau_u = pm.HalfNormal("tau_u", sigma=priors.source_deviation_prior_sd)
        z_source = _within_type_deviations(pm, pt, design, source_type_of_key)
        u_source = pm.Deterministic("u_source", tau_u * z_source, dims="source_key")
        beta_source = pm.Deterministic(
            "beta_source", b_type[source_type_of_key] + u_source, dims="source_key"
        )

        # ── claim lineage: a repeated claim counts once ──────────────────────
        tau_lineage = pm.HalfNormal("tau_lineage", sigma=priors.lineage_prior_sd)
        z_lineage = pm.Normal("z_lineage", 0.0, 1.0, dims="lineage")
        l_lineage = pm.Deterministic("l_lineage", tau_lineage * z_lineage, dims="lineage")

        # ── era fallback, on unlabelled rows only ────────────────────────────
        # Gated, because an ancient chronicle row already carries g_chronicle,
        # which was fitted on cases that were themselves mostly pre-modern:
        # applying the era term to it as well deflates the same rows twice.
        # The prior is measured from the gold set's own years. The mixture-gap
        # value used before was a different quantity with the opposite sign.
        a_era = pm.Normal("a_era", mu=inflation.era_mu, sigma=inflation.era_sd)
        era_column = design.era * _unlabelled_mask(design)

        # ── observation scale ────────────────────────────────────────────────
        sigma_sq_type = pm.InverseGamma(
            "sigma_sq_type",
            alpha=priors.precision_prior_alpha,
            beta=priors.precision_prior_beta,
            dims="source_type",
        )
        tau_sigma = pm.HalfNormal("tau_sigma", sigma=priors.sigma_deviation_prior_sd)
        z_sigma = pm.Normal("z_sigma", 0.0, 1.0, dims="source_key")
        sigma_source = pm.Deterministic(
            "sigma_source",
            pm.math.sqrt(sigma_sq_type[source_type_of_key]) * pm.math.exp(tau_sigma * z_sigma),
            dims="source_key",
        )

        sigma_estimate = pm.HalfNormal("sigma_estimate", sigma=priors.estimate_dispersion_sd)
        sigma_round = pm.HalfNormal("sigma_round", sigma=priors.roundness_dispersion_sd)

        eta = (
            mu_side[design.side_index]
            + beta_source[design.source_index]
            + g_regime[design.regime_index]
            + delta_level[design.level_index]
            + l_lineage[design.lineage_index]
            + a_era * era_column
        )

        # Monotone, interpretable inflation of the observation scale. These
        # terms stand for extra dispersion from a latent mixture: an unreadable
        # scope, a hedged figure, a number rounded to a convention. Added in
        # variance rather than in scale because that is the algebra for
        # independent error components.
        var = (
            sigma_source[design.source_index] ** 2
            + sigma_estimate**2 * design.is_estimate.astype(float)
            + sigma_round**2 * design.roundness
        )
        unknown_mask = _unknown_level_mask(design)
        if unknown_mask.any():
            sigma_unknown = pm.HalfNormal(
                "sigma_unknown", sigma=priors.scope_unknown_dispersion_sd
            )
            var = var + sigma_unknown**2 * unknown_mask
        sd = pm.math.sqrt(var)

        _attach_likelihood(pm, design, eta, sd, priors.likelihood_family)

    logger.info(
        "reconcile_model_built",
        quantity=design.quantity,
        n_obs=design.n_obs,
        n_sides=design.n_sides,
        n_sources=design.n_sources,
        likelihood=priors.likelihood_family,
        anchor_regime=inflation.anchor_regime,
        free_regimes=free_regimes,
        era_mu=inflation.era_mu,
    )
    return model


def _source_type_of_key(design: Design) -> np.ndarray:
    """Map each source key onto its source-type index.

    Args:
        design: The design, whose ``source_keys`` carry their own type.

    Returns:
        An integer array of length ``n_sources``.
    """
    lookup = {name: i for i, name in enumerate(design.source_types)}
    return np.array([lookup[key.source_type] for key in design.source_keys], dtype=int)


def _within_type_deviations(
    pm_module: Any, pt_module: Any, design: Design, source_type_of_key: np.ndarray
) -> Any:
    """Per-source deviations, constrained to sum to zero within each type.

    Zero-sum across all keys leaves the type-level mean of the deviations free,
    and that mean is the same parameter as ``b_type``. With a single key in a
    type the two are exactly confounded; with several the ridge is softer but
    still costs most of the effective sample size on ``tau_u``. A type holding
    one key therefore gets no free deviation, which is correct: ``b_type``
    carries it entirely.

    Args:
        pm_module: The imported pymc module.
        pt_module: The imported pytensor.tensor module.
        design: The design.
        source_type_of_key: Each key's source-type index.

    Returns:
        A pytensor vector over the ``source_key`` dimension.
    """
    z = pt_module.zeros(len(design.source_keys))
    for type_index, type_name in enumerate(design.source_types):
        keys = np.flatnonzero(source_type_of_key == type_index)
        if keys.size < 2:
            continue
        z = pt_module.set_subtensor(
            z[keys],
            pm_module.ZeroSumNormal(f"z_source_{type_name}", sigma=1.0, shape=keys.size),
        )
    return z


def _regime_bias(
    pm_module: Any,
    pt_module: Any,
    design: Design,
    inflation: InflationPrior,
    free_regimes: list[str],
) -> Any:
    """Build the claim-regime bias vector, with the anchor pinned to zero.

    Args:
        pm_module: The imported pymc module.
        pt_module: The imported pytensor.tensor module.
        design: The design, for the regime labels.
        inflation: The fitted prior.
        free_regimes: The regimes that are not the anchor.

    Returns:
        A pytensor vector over the ``regime`` dimension.
    """
    full = pt_module.zeros(len(design.regimes))
    if not free_regimes:
        return full

    mus = np.array([inflation.moments_for(r)[0] for r in free_regimes])
    sds = np.array([inflation.moments_for(r)[1] for r in free_regimes])
    free = pm_module.Normal("g_free", mu=mus, sigma=sds, dims="regime_free")

    positions = np.array([design.regimes.index(r) for r in free_regimes], dtype=int)
    return pt_module.set_subtensor(full[positions], free)


def _level_offsets(pm_module: Any, pt_module: Any, design: Design, priors: ModelPriors) -> Any:
    """Build the scope or casualty-type offsets against a pinned reference.

    For troops the offsets are **cumulative** non-negative increments, which is
    what actually encodes the containment hierarchy (engaged within available
    within a theatre roster). Independent half-normals would only bound each
    level above the reference and would happily put ``theatre_strength`` below
    ``available``. ``unknown`` is the exception: it is a latent mixture over
    the real scopes, so it takes a free offset weakly centred on the reference,
    and its extra variance is added separately.

    For casualties the components are *subsets* of a total, so their offsets
    are non-positive and independent rather than nested: killed is not nested
    inside wounded.

    Args:
        pm_module: The imported pymc module.
        pt_module: The imported pytensor.tensor module.
        design: The design, for the level labels.
        priors: Hierarchy scales.

    Returns:
        A pytensor vector over the ``level`` dimension, with index 0 zero.
    """
    is_troops = design.quantity == _TROOPS
    parts: list[Any] = [pt_module.constant(0.0)]
    running: Any = pt_module.constant(0.0)

    for level in design.levels[1:]:
        if is_troops and level == "unknown":
            parts.append(pm_module.Normal("delta_unknown", mu=0.0, sigma=0.3))
        elif is_troops:
            # Cumulative: each ordered scope sits at or above the one before.
            running = running + pm_module.HalfNormal(
                f"d_{level}", sigma=priors.scope_offset_prior_sd
            )
            parts.append(running)
        else:
            parts.append(
                -pm_module.HalfNormal(
                    f"d_{level}", sigma=priors.casualty_type_offset_prior_sd
                )
            )

    return pt_module.stack(parts)


def _unknown_level_mask(design: Design) -> np.ndarray:
    """Flag the rows whose level is the residual 'unknown' scope.

    Args:
        design: The design.

    Returns:
        A float array, 1.0 where the row's level is ``"unknown"``.
    """
    if "unknown" not in design.levels:
        return np.zeros(design.n_obs, dtype=float)
    index = design.levels.index("unknown")
    mask: np.ndarray = (design.level_index == index).astype(float)
    return mask


def _unlabelled_mask(design: Design) -> np.ndarray:
    """Flag the rows whose claim regime could not be read.

    Args:
        design: The design.

    Returns:
        A float array, 1.0 where the row's regime is ``"unlabelled"``.
    """
    if "unlabelled" not in design.regimes:
        return np.zeros(design.n_obs, dtype=float)
    index = design.regimes.index("unlabelled")
    mask: np.ndarray = (design.regime_index == index).astype(float)
    return mask


def _attach_likelihood(pm_module: Any, design: Design, eta: Any, sd: Any, family: str) -> None:
    """Attach the point and censored likelihood terms.

    The censoring direction is inverted relative to the flag names, and that
    inversion happens once, in :mod:`pipeline.reconcilers.design`. Here the
    partitions are taken at face value: ``censor_lower_at`` means the latent
    value lies below the observation, which ``pm.Censored`` expresses as
    ``lower``.

    Args:
        pm_module: The imported pymc module.
        design: The design.
        eta: The linear predictor.
        sd: The per-observation scale.
        family: ``"student_t"`` or ``"normal"``.

    Raises:
        ValueError: If the family is not recognised.
    """
    if family not in ("student_t", "normal"):
        raise ValueError(f"Unknown likelihood_family {family!r}; expected student_t or normal")

    nu = None
    if family == "student_t":
        # Concentrated on small values on purpose. PyMC's Gamma takes a rate,
        # so Gamma(2, 0.1) would have mean 20, where StudentT is visually
        # Normal -- the robustness the family is here for would be asserted and
        # not delivered. Gamma(2, 0.5) has mean 4, so nu sits near 6 and the
        # tails stay genuinely heavy. Registered as a Deterministic so it
        # reaches the trace and the convergence diagnostics.
        nu = pm_module.Deterministic(
            "nu", pm_module.Gamma("nu_raw", alpha=2.0, beta=0.5) + 2.0
        )

    def dist(rows: np.ndarray) -> Any:
        if family == "normal":
            return pm_module.Normal.dist(mu=eta[rows], sigma=sd[rows])
        return pm_module.StudentT.dist(nu=nu, mu=eta[rows], sigma=sd[rows])

    if design.point_rows.size:
        rows = design.point_rows
        if family == "normal":
            pm_module.Normal(
                "obs_point", mu=eta[rows], sigma=sd[rows], observed=design.log_y[rows]
            )
        else:
            pm_module.StudentT(
                "obs_point", nu=nu, mu=eta[rows], sigma=sd[rows], observed=design.log_y[rows]
            )

    if design.censor_lower_at.size:
        rows = design.censor_lower_at
        pm_module.Censored(
            "obs_censor_lower",
            dist(rows),
            lower=design.log_y[rows],
            upper=np.inf,
            observed=design.log_y[rows],
        )

    if design.censor_upper_at.size:
        rows = design.censor_upper_at
        pm_module.Censored(
            "obs_censor_upper",
            dist(rows),
            lower=-np.inf,
            upper=design.log_y[rows],
            observed=design.log_y[rows],
        )

    if design.zero_rows.size:
        # "One casualty or fewer". Kept rather than dropped, because zero
        # casualties is a fact, and censored rather than log1p-shifted, because
        # log1p would move the scale of every other observation.
        rows = design.zero_rows
        pm_module.Censored(
            "obs_zero",
            dist(rows),
            lower=design.log_y[rows],
            upper=np.inf,
            observed=design.log_y[rows],
        )


def sample(model: pm.Model, settings: SamplerSettings) -> az.InferenceData:
    """Sample a built model.

    Args:
        model: The model from :func:`build_model`.
        settings: Sampler settings.

    Returns:
        The posterior, as ArviZ InferenceData.

    Raises:
        ImportError: If pymc is not installed.
    """
    import pymc as pm  # noqa: PLC0415 - deliberately deferred; see module docstring

    kwargs: dict[str, Any] = {
        "draws": settings.draws,
        "tune": settings.tune,
        "chains": settings.chains,
        "cores": settings.cores,
        "target_accept": settings.target_accept,
        "random_seed": settings.random_seed,
        "progressbar": False,
    }
    if settings.nuts_sampler and settings.nuts_sampler != "pymc":
        kwargs["nuts_sampler"] = settings.nuts_sampler
    else:
        # nutpie ignores idata_kwargs and warns; only pass it where it applies.
        # The pointwise log-likelihood is n_obs by n_draws and nothing reads it.
        kwargs["idata_kwargs"] = {"log_likelihood": False}

    with model:
        try:
            idata = pm.sample(**kwargs)
        except (ImportError, ValueError, RuntimeError) as exc:
            # nutpie is the default because PyTensor without a C compiler falls
            # back to Python ops, which is far too slow for this model. If it
            # is unavailable, say so loudly and carry on rather than failing
            # the whole stage over a sampler choice.
            if "nuts_sampler" not in kwargs:
                raise
            logger.warning(
                "reconcile_nuts_sampler_unavailable",
                requested=settings.nuts_sampler,
                error=str(exc),
                error_type=type(exc).__name__,
                hint="falling back to the default PyMC sampler; expect this to be slow",
            )
            kwargs.pop("nuts_sampler")
            kwargs["idata_kwargs"] = {"log_likelihood": False}
            idata = pm.sample(**kwargs)

    return idata
