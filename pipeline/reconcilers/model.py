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
             + delta_level[k_i] + a_era * era_i * unlabelled_i
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
``docs/handover-log.md`` §19.2 -- the unlabelled offset became the intercept under
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

   The exception (pilot review S4) is a type holding **both** own-effect and
   pooled keys (a "mixed" type). There the own-effect keys' deviations
   (``u_source`` and ``z_sigma`` alike) are plain ``Normal(0, 1)`` draws scaled
   by their tau, so
   ``b_type`` against the own keys' mean deviation is settled by the deviation
   prior (partial pooling), not by the pooled keys: after S3 the pooled keys
   are those that rarely co-report, so they pin ``b_type`` only weakly. The sum
   of the effective type biases is then not exactly zero in a mixed type, and
   the absolute level depends softly on ``tau_u``'s prior. Exposure on the
   current corpus is zero (no type has own-effect keys at all). A type whose
   keys all have their own effect keeps the ``ZeroSumNormal``; it is the only
   thing identifying ``b_type`` there.
3. **The rotation.** The observation-weighted mean of the reference-coded
   blocks is what the data identify, so it is what gets sampled: ``corpus_level``
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

## Source effects are nested in battles on the real corpus

``u_source`` and ``sigma_source`` are one pair per source key, which identifies
them only when keys are crossed with sides. On the real corpus every key is one
battle's Wikipedia infobox (0 of 148 span two battles), so a per-key bias is the
battle level under another name. It absorbed the between-side spread (tau_u 0.87,
``s0`` collapsing to 0.10) and, because the published estimate leaves the source
term out, every side came back at about 35,000 on a clean convergence. A key
therefore carries its own ``u_source`` and its own noise deviation only if
``Design.source_has_own_effect`` says it **co-reports**: another source key
reports the same side in at least ``min_battles_per_source_effect`` battles
(pilot review S3). Spanning that many battles is not enough, because a key
alone on its sides has its bias as the side level under another name however
many battles it covers. Every other key is pooled: its bias is its
source type's ``b_type`` and its noise is its type's ``sqrt(sigma_sq_type)``.
When no source type has a key that can carry a deviation there is no
``u_source``, ``tau_u``, ``z_sigma`` or ``tau_sigma`` at all. The recovery tests
that pass with cross-classified sources therefore keep passing, and the nested
case has its own.

**tau_u and tau_sigma need data to be hyperparameters** (pilot review N1). A
scale learned from two or three deviations is the prior in a different guise,
and it samples as a funnel. They are ``HalfNormal`` only when at least
``min_keys_for_hierarchical_deviation`` (spec param, default 5) keys carry a
deviation; below that they are constants at ``source_deviation_prior_sd`` and
``sigma_deviation_prior_sd``, so the deviations are plain draws from a fixed
prior. The count is of the keys that actually get a deviation, which excludes
the lone key of a type that has no pooled keys, since that key is ``b_type``.

## Observation scale, casualty ratios and one-sided bounds

``sigma_sq_type`` is ``InverseGamma(3, 4/3)``. The first version was IG(2, 1),
whose infinite variance sits on a parameter that sets every interval width.
Alpha 3 fixes that, but IG(3, 2) keeping the *mean* at 1 was a much stronger
prior than that suggests (mode 0.5, a thin right tail) and visibly shrank
two-report sides. IG(3, 4/3) instead keeps the old *mode*
(``beta / (alpha + 1) = 1/3``), so the bulk of the prior is where it was, and
has finite variance.

``delta_level`` for casualties is a corpus constant, so a killed report implies
the same killed-to-total ratio at Cannae as at the Somme. Each (side, non-total
level) pair that occurs therefore gets a non-centred offset
``r = sigma_ratio * z_ratio`` with ``z_ratio ~ N(0, 1)`` and
``sigma_ratio ~ HalfNormal(casualty_ratio_dispersion_sd)``, default 0.5, added to
``eta``. It moves the component's level for that side, which a per-row variance
(the first version) could only blur. A side with a single component report has
that offset confounded with its own noise; the prior is what keeps it honest.
Troops are unchanged.

``bound_likelihood`` chooses how a lone bound is modelled. ``exgaussian`` (the
default) reads a bound as a noisy report plus a one-sided excess: "up to X" is
``log y ~ ExGaussian(eta, sd, nu)`` and "at least X" is
``-log y ~ ExGaussian(-eta, sd, nu)``, with ``nu`` the mean excess in log units and
``HalfNormal(bound_excess_prior_sd)``, one per direction and shared across sides
and sources (``lambda_upper`` and ``lambda_lower`` in the trace; their
posterior and contraction are in the run diagnostics). Its noise is Gaussian,
not Student-t, so a mis-extracted bound is absorbed less robustly than a
mis-extracted point; its ``sd`` is scaled to the point rows' variance so a bound
is not treated as a more precise report than a point from the same source.
``censored`` treats "up to X" as ``latent <= X`` through ``pm.Censored``, which
says nothing about how far below X the truth lies and, on a bound-heavy corpus,
lets the level drift (the recovery ablation measures it). Ranges are merged
points before this point and are unaffected, and exactly-zero casualty rows
stay censored in both settings.

## What a period mean would not separate (N3)

``mu_side`` and ``a_era`` are not separately identified on unlabelled rows: a
period mean on the side level and the era slope move the same rows, because
``era_side`` equals ``era_row`` there (a side's rows share its battle's date).
Nothing in the likelihood tells "armies of this period were smaller" from "this
period's unmarked figures are deflated". What separates them is ``a_era``'s
fitted prior (``inflation.era_mu`` and ``era_sd``). Adding a period mean on
``mu_side`` would hand that job to a prior that was never fitted for it, so
none is added.

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

## The neutral fit

``build_model(..., neutral=True)`` is the same model with two priors changed,
and is fit for troops only. The unlabelled offset is almost all prior, so in a
force ratio it cancels only when both sides share a regime mix; Gaugamela, with
a labelled Persian claim against an unlabelled Macedonian figure, is tilted by
the whole of ``_UNLABELLED_WEIGHT``. In the neutral fit the unlabelled regime
takes the faithful-component prior (an unmarked figure is read as an ordinary
modern count) and ``a_era`` takes ``N(0, _NEUTRAL_ERA_SD)``. It is a prior
rather than a pin at zero, so data that do identify an unlabelled-versus-anchor
gap can still move it. Labelled regimes keep their fitted priors, and every
identifiability device above is untouched. Neutral estimates are for ratios
only, never for display as troop counts.

## Non-centred throughout

Most group-level offsets in this corpus are weakly identified: a source seen
once, a regime with five cases. Centred parameterisations funnel badly under
weak identification, and non-centred costs nothing when the data are
informative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

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
# consequential constant in this stage; see docs/handover-log.md §19.9.
_UNLABELLED_WEIGHT: Final[float] = 0.5

# The prior on `level` after the rotation. Wider than the spec's
# side_prior_mu_sd, deliberately: before the rotation that prior supplied
# roughly half the identification of the unlabelled offset, and
# "armies are about 10,000 strong" is a spec constant rather than a measured
# quantity. It should not be quietly doing half the debiasing of the corpus.
_LEVEL_PRIOR_SD: Final[float] = 2.0

# a_era's prior in the neutral fit. Tight on purpose: the era term is the
# other route by which an unmarked ancient figure gets deflated, and the
# neutral fit exists to switch that route off for ratios.
_NEUTRAL_ERA_SD: Final[float] = 0.25

_TROOPS: Final[str] = "troops"
_UNLABELLED: Final[str] = "unlabelled"


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
    side_prior_nu: float = 4.0
    bias_prior_sd: float = 1.0
    source_deviation_prior_sd: float = 0.5
    sigma_deviation_prior_sd: float = 0.3
    scope_offset_prior_sd: float = 0.4
    casualty_type_offset_prior_sd: float = 0.7
    scope_unknown_dispersion_sd: float = 0.4
    estimate_dispersion_sd: float = 0.3
    roundness_dispersion_sd: float = 0.3
    precision_prior_alpha: float = 3.0
    precision_prior_beta: float = 4.0 / 3.0
    min_keys_for_hierarchical_deviation: int = 5
    casualty_ratio_dispersion_sd: float = 0.5
    bound_likelihood: str = "exgaussian"
    bound_excess_prior_sd: float = 0.5
    likelihood_family: str = "student_t"
    contamination_pi_alpha: float = 1.0
    contamination_pi_beta: float = 9.0
    contamination_sd: float = 2.0

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
            side_prior_nu=float(params.get("side_prior_nu", 4.0)),
            bias_prior_sd=float(params.get("bias_prior_sd", 1.0)),
            source_deviation_prior_sd=float(params.get("source_deviation_prior_sd", 0.5)),
            sigma_deviation_prior_sd=float(params.get("sigma_deviation_prior_sd", 0.3)),
            scope_offset_prior_sd=float(params.get("scope_offset_prior_sd", 0.4)),
            casualty_type_offset_prior_sd=float(params.get("casualty_type_offset_prior_sd", 0.7)),
            scope_unknown_dispersion_sd=float(params.get("scope_unknown_dispersion_sd", 0.4)),
            estimate_dispersion_sd=float(params.get("estimate_dispersion_sd", 0.3)),
            roundness_dispersion_sd=float(params.get("roundness_dispersion_sd", 0.3)),
            precision_prior_alpha=float(params.get("precision_prior_alpha", 3.0)),
            precision_prior_beta=float(params.get("precision_prior_beta", 4.0 / 3.0)),
            min_keys_for_hierarchical_deviation=int(
                params.get("min_keys_for_hierarchical_deviation", 5)
            ),
            casualty_ratio_dispersion_sd=float(params.get("casualty_ratio_dispersion_sd", 0.5)),
            bound_likelihood=str(params.get("bound_likelihood", "exgaussian")),
            bound_excess_prior_sd=float(params.get("bound_excess_prior_sd", 0.5)),
            likelihood_family=str(params.get("likelihood_family", "student_t")),
            contamination_pi_alpha=float(params.get("contamination_pi_alpha", 1.0)),
            contamination_pi_beta=float(params.get("contamination_pi_beta", 9.0)),
            contamination_sd=float(params.get("contamination_sd", 2.0)),
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
    neutral: bool = False,
) -> pm.Model:
    """Build the hierarchical source-disagreement model for one quantity.

    Args:
        design: The aligned arrays from
            :func:`pipeline.reconcilers.design.build_design`.
        priors: Hierarchy scales from the agent spec.
        inflation: The fitted claim-regime inflation prior.
        neutral: Build the regime-neutral variant used for force ratios; see
            "The neutral fit" in the module docstring. Troops only.

    Returns:
        An unsampled :class:`pymc.Model`.

    Raises:
        ImportError: If pymc is not installed. Raised here rather than at
            import time, so the rest of the package stays usable without it.
        ValueError: If the likelihood family is not recognised, the anchor
            regime is not one the design knows about, or ``neutral`` is
            requested for a quantity other than troops.
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

    if neutral and design.quantity != _TROOPS:
        raise ValueError("The neutral fit exists for troops only; casualties get none.")

    coords = design.coords()
    free_regimes = [r for r in design.regimes if r != inflation.anchor_regime]
    coords["regime_free"] = free_regimes
    if _ratio_level_mask(design).any():
        coords["ratio_pair"] = _ratio_pairs(design)[1]
    source_type_of_key = _source_type_of_key(design)

    # The observation-weighted share of each reference-coded level. These are
    # the directions the data identify; see device 3 in the module docstring.
    w_regime = np.bincount(design.regime_index, minlength=len(design.regimes)) / design.n_obs
    w_level = np.bincount(design.level_index, minlength=len(design.levels)) / design.n_obs

    with pm.Model(coords=coords) as model:
        # ── claim regime: the inflation term, anchor pinned to zero ──────────
        g_regime = _regime_bias(pm, pt, design, inflation, free_regimes, neutral=neutral)
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
        level = pm.Normal("corpus_level", mu=priors.side_prior_mu, sigma=_LEVEL_PRIOR_SD)
        m0 = pm.Deterministic("m0", level - offset_bar)

        s0 = pm.HalfNormal("s0", sigma=priors.side_prior_sd_scale)
        # Heavy-tailed, with nu fixed from the spec. A Normal side prior against
        # a Student-t likelihood makes the model explain a genuinely large side
        # as outlying reports around a typical one (Kursk's 2.5M read as noise).
        # Non-centred: mu_side = m0 + s0 * z_side is unchanged.
        z_side = pm.StudentT("z_side", nu=priors.side_prior_nu, mu=0.0, sigma=1.0, dims="side")
        mu_side = pm.Deterministic("mu_side", m0 + s0 * z_side, dims="side")

        # ── source bias: type mean plus within-type deviation ────────────────
        # Only keys spanning enough battles get a deviation (see the nesting
        # section of the module docstring). With none, there is no u_source and
        # no tau_u: a zero-size or all-zero deviation would leave tau_u
        # sampling its prior and report clean convergence on nothing.
        b_type = pm.ZeroSumNormal("b_type", sigma=priors.bias_prior_sd, dims="source_type")
        beta = b_type[source_type_of_key]
        deviation_groups = _deviation_groups(design, source_type_of_key)
        n_deviating_keys = sum(group.keys.size for group in deviation_groups)
        # Below the floor the scale would be learned from a handful of
        # deviations and is the prior under another name; fix it instead.
        hierarchical_deviation = n_deviating_keys >= priors.min_keys_for_hierarchical_deviation
        if deviation_groups:
            tau_u: Any = (
                pm.HalfNormal("tau_u", sigma=priors.source_deviation_prior_sd)
                if hierarchical_deviation
                else priors.source_deviation_prior_sd
            )
            z_source = _within_type_deviations(pm, pt, design, deviation_groups)
            u_source = pm.Deterministic("u_source", tau_u * z_source, dims="source_key")
            beta = beta + u_source
        beta_source = pm.Deterministic("beta_source", beta, dims="source_key")

        # A repeated claim counts once because build_design collapses each
        # lineage to one row; there is no lineage term. See collapse_lineages.

        # ── era fallback, on unlabelled rows only ────────────────────────────
        # Gated, because an ancient chronicle row already carries g_chronicle,
        # which was fitted on cases that were themselves mostly pre-modern:
        # applying the era term to it as well deflates the same rows twice.
        # The prior is measured from the gold set's own years. The mixture-gap
        # value used before was a different quantity with the opposite sign.
        a_era = pm.Normal(
            "a_era",
            mu=0.0 if neutral else inflation.era_mu,
            sigma=_NEUTRAL_ERA_SD if neutral else inflation.era_sd,
        )
        era_column = _data(pm, "data_era", design.era * _unlabelled_mask(design))

        # ── observation scale ────────────────────────────────────────────────
        sigma_sq_type = pm.InverseGamma(
            "sigma_sq_type",
            alpha=priors.precision_prior_alpha,
            beta=priors.precision_prior_beta,
            dims="source_type",
        )
        sigma_type_scale = pm.math.sqrt(sigma_sq_type[source_type_of_key])
        if deviation_groups:
            tau_sigma: Any = (
                pm.HalfNormal("tau_sigma", sigma=priors.sigma_deviation_prior_sd)
                if hierarchical_deviation
                else priors.sigma_deviation_prior_sd
            )
            # Zero-sum within a type of own-effect keys only, for the same reason
            # as u_source: a free mean of z_sigma within such a type is the same
            # parameter as that type's sigma_sq_type. Measured on the recovery
            # corpus as tau_sigma at ess 201 and rhat 1.02, the last thing
            # failing the convergence gate.
            z_sigma = pm.Deterministic(
                "z_sigma",
                _within_type_deviations(pm, pt, design, deviation_groups, prefix="z_sigma"),
                dims="source_key",
            )
            sigma_type_scale = sigma_type_scale * pm.math.exp(tau_sigma * z_sigma)
        # A pooled key's noise is its type's scale: one sigma per source type,
        # drawn from the same inverse-gamma the per-key scales are centred on.
        sigma_source = pm.Deterministic("sigma_source", sigma_type_scale, dims="source_key")

        sigma_estimate = pm.HalfNormal("sigma_estimate", sigma=priors.estimate_dispersion_sd)
        sigma_round = pm.HalfNormal("sigma_round", sigma=priors.roundness_dispersion_sd)

        # Every per-observation array enters as pm.Data, never as a numpy
        # constant. PyTensor's numba backend compiles constants into the
        # generated code as literals, and nutpie's compile then grows faster
        # than linearly in the corpus: 43s at 160 observations, 334s at 320,
        # and 11 GB without finishing at 670. As inputs, the compiled function
        # is the same size whatever the corpus. The model is unchanged.
        source_ix = _data(pm, "data_source_index", design.source_index)
        eta = (
            mu_side[_data(pm, "data_side_index", design.side_index)]
            + beta_source[source_ix]
            + g_regime[_data(pm, "data_regime_index", design.regime_index)]
            + delta_level[_data(pm, "data_level_index", design.level_index)]
            + a_era * era_column
        )
        ratio_mask = _ratio_level_mask(design)
        if ratio_mask.any():
            # delta_level is one corpus-wide constant per component, so the
            # killed-to-total ratio would be identical at Cannae and the Somme.
            # Each (side, non-total level) pair that occurs gets its own
            # offset r ~ N(0, sigma_ratio), non-centred, so the ratio varies
            # by battle and moves a level instead of only widening an interval.
            pair_of_row, _ = _ratio_pairs(design)
            sigma_ratio = pm.HalfNormal("sigma_ratio", sigma=priors.casualty_ratio_dispersion_sd)
            z_ratio = pm.Normal("z_ratio", 0.0, 1.0, dims="ratio_pair")
            eta = eta + (sigma_ratio * z_ratio)[_data(pm, "data_ratio_pair", pair_of_row)] * _data(
                pm, "data_ratio_mask", ratio_mask
            )

        # Monotone, interpretable inflation of the observation scale. These
        # terms stand for extra dispersion from a latent mixture: an unreadable
        # scope, a hedged figure, a number rounded to a convention. Added in
        # variance rather than in scale because that is the algebra for
        # independent error components.
        var = (
            sigma_source[source_ix] ** 2
            + sigma_estimate**2 * _data(pm, "data_is_estimate", design.is_estimate.astype(float))
            + sigma_round**2 * _data(pm, "data_roundness", design.roundness)
            # A stated range is one point row whose width is the extra spread.
            + _data(pm, "data_range_var", design.range_var)
        )
        unknown_mask = _unknown_level_mask(design)
        if unknown_mask.any():
            sigma_unknown = pm.HalfNormal("sigma_unknown", sigma=priors.scope_unknown_dispersion_sd)
            var = var + sigma_unknown**2 * _data(pm, "data_unknown_mask", unknown_mask)
        sd = pm.math.sqrt(var)

        log_y = _data(pm, "data_log_y", design.log_y)
        _attach_likelihood(pm, pt, design, eta, sd, log_y, priors)
        # Not registered as variables (n_obs-long deterministics would be
        # sampled and diagnosed for nothing); kept so a test can compile the
        # model's own per-row location and scale against summarise's rebuild.
        model.row_eta = eta
        model.row_sd = sd

    logger.info(
        "reconcile_model_built",
        quantity=design.quantity,
        n_obs=design.n_obs,
        n_sides=design.n_sides,
        n_sources=design.n_sources,
        n_source_keys_own_effect=design.n_source_keys_own_effect,
        n_source_keys_pooled=design.n_source_keys_pooled,
        likelihood=priors.likelihood_family,
        bound_likelihood=priors.bound_likelihood,
        hierarchical_deviation=hierarchical_deviation,
        anchor_regime=inflation.anchor_regime,
        free_regimes=free_regimes,
        era_mu=inflation.era_mu,
        neutral=neutral,
    )
    return model


def _data(pm_module: Any, name: str, values: np.ndarray) -> Any:
    """Register a per-observation array as model data rather than a constant.

    Args:
        pm_module: The imported pymc module.
        name: The data container's name; prefixed ``data_`` so it can never
            collide with a dimension or a likelihood term.
        values: The array.

    Returns:
        The ``pm.Data`` container, usable wherever the array was.
    """
    return pm_module.Data(name, values)


def _source_type_of_key(design: Design) -> np.ndarray:
    """Map each source key onto its source-type index.

    Args:
        design: The design, whose ``source_keys`` carry their own type.

    Returns:
        An integer array of length ``n_sources``.
    """
    lookup = {name: i for i, name in enumerate(design.source_types)}
    return np.array([lookup[key.source_type] for key in design.source_keys], dtype=int)


class DeviationGroup(NamedTuple):
    """One source type's keys that carry a within-type deviation.

    Attributes:
        type_name: The source type.
        keys: Indices of the type's own-effect keys.
        zero_sum: Whether their deviations are constrained to sum to zero. Only
            when the type has no pooled key: a pooled key identifies ``b_type``,
            and a zero-sum on the others would pin it to their mean.
    """

    type_name: str
    keys: np.ndarray
    zero_sum: bool


def _deviation_groups(design: Design, source_type_of_key: np.ndarray) -> list[DeviationGroup]:
    """The source-type groups that can carry a within-type deviation.

    A type whose keys all have their own effect needs two of them, with a
    zero-sum: one key's deviation is its type's mean, which ``b_type`` already
    carries. A type that also holds pooled keys needs only one, with no
    zero-sum. There ``b_type`` against the own keys' mean deviation is settled
    by the deviation prior (partial pooling), not by the pooled keys, which
    after S3 rarely co-report and so pin it only weakly; the type biases then
    no longer sum exactly to zero and the absolute level depends softly on
    ``tau_u``'s prior. Exposure on the current corpus is zero.

    Args:
        design: The design, whose ``source_has_own_effect`` selects the keys.
        source_type_of_key: Each key's source-type index.

    Returns:
        One :class:`DeviationGroup` per qualifying type. Empty when none
        qualifies, in which case the model builds no deviation terms at all.
    """
    groups: list[DeviationGroup] = []
    for type_index, type_name in enumerate(design.source_types):
        in_type = source_type_of_key == type_index
        own = np.flatnonzero(in_type & design.source_has_own_effect)
        if own.size == 0:
            continue
        has_pooled = bool((in_type & ~design.source_has_own_effect).any())
        if has_pooled:
            groups.append(DeviationGroup(type_name, own, zero_sum=False))
        elif own.size >= 2:
            groups.append(DeviationGroup(type_name, own, zero_sum=True))
    return groups


def _within_type_deviations(
    pm_module: Any,
    pt_module: Any,
    design: Design,
    groups: list[DeviationGroup],
    *,
    prefix: str = "z_source",
) -> Any:
    """Per-source deviations, zero-sum within a type only where that is safe.

    Zero-sum across all keys leaves the type-level mean of the deviations free,
    and that mean is the same parameter as ``b_type``. With a single key in a
    type the two are exactly confounded; with several the ridge is softer but
    still costs most of the effective sample size on ``tau_u``. A type whose
    keys all carry a deviation is therefore zero-sum within the type, and one
    holding a single such key gets none. In a type that also holds pooled keys
    those identify ``b_type``, so the deviations are plain standard normals
    there; see device 2 in the module docstring.

    Args:
        pm_module: The imported pymc module.
        pt_module: The imported pytensor.tensor module.
        design: The design.
        groups: From :func:`_deviation_groups`.
        prefix: Name stem for the per-type variables, so the bias deviations and
            the noise-scale deviations stay distinct.

    Returns:
        A pytensor vector over the ``source_key`` dimension, exactly zero for
        every key outside ``groups``.
    """
    z = pt_module.zeros(len(design.source_keys))
    for group in groups:
        name = f"{prefix}_{group.type_name}"
        draw = (
            pm_module.ZeroSumNormal(name, sigma=1.0, shape=group.keys.size)
            if group.zero_sum
            else pm_module.Normal(name, mu=0.0, sigma=1.0, shape=group.keys.size)
        )
        z = pt_module.set_subtensor(z[group.keys], draw)
    return z


def _regime_bias(
    pm_module: Any,
    pt_module: Any,
    design: Design,
    inflation: InflationPrior,
    free_regimes: list[str],
    *,
    neutral: bool = False,
) -> Any:
    """Build the claim-regime bias vector, with the anchor pinned to zero.

    Args:
        pm_module: The imported pymc module.
        pt_module: The imported pytensor.tensor module.
        design: The design, for the regime labels.
        inflation: The fitted prior.
        free_regimes: The regimes that are not the anchor.
        neutral: Give the unlabelled regime the faithful-component prior
            instead of its mixture-weighted one.

    Returns:
        A pytensor vector over the ``regime`` dimension.
    """
    full = pt_module.zeros(len(design.regimes))
    if not free_regimes:
        return full

    def moments(regime: str) -> tuple[float, float]:
        if neutral and regime == _UNLABELLED:
            return inflation.faithful_mu, inflation.faithful_sd
        return inflation.moments_for(regime)

    mus = np.array([moments(r)[0] for r in free_regimes])
    sds = np.array([moments(r)[1] for r in free_regimes])
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
                -pm_module.HalfNormal(f"d_{level}", sigma=priors.casualty_type_offset_prior_sd)
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


def _ratio_level_mask(design: Design) -> np.ndarray:
    """Flag the casualty rows whose type is a component rather than the total.

    Args:
        design: The design.

    Returns:
        A float array, 1.0 where a casualty row's level is not the reference
        (``total``) level; all zeros for troops.
    """
    if design.quantity == _TROOPS:
        return np.zeros(design.n_obs, dtype=float)
    mask: np.ndarray = (design.level_index != 0).astype(float)
    return mask


def _ratio_pairs(design: Design) -> tuple[np.ndarray, list[str]]:
    """The (side, component level) pairs that carry a casualty ratio offset.

    Args:
        design: The design.

    Returns:
        ``(pair_of_row, labels)``: for each row the index of its pair (0 for a
        row that is not a ratio row, which the ratio mask zeroes out), and the
        ``"{side_id}:{level}"`` label of each pair. Only pairs that occur get
        an offset, so a side with no component-level report carries none.
    """
    mask = _ratio_level_mask(design) > 0
    pair_of_row = np.zeros(design.n_obs, dtype=int)
    index: dict[tuple[int, int], int] = {}
    labels: list[str] = []
    for row in np.flatnonzero(mask):
        key = (int(design.side_index[row]), int(design.level_index[row]))
        if key not in index:
            index[key] = len(labels)
            labels.append(f"{design.side_ids[key[0]]}:{design.levels[key[1]]}")
        pair_of_row[row] = index[key]
    return pair_of_row, labels


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


def _attach_likelihood(
    pm_module: Any,
    pt_module: Any,
    design: Design,
    eta: Any,
    sd: Any,
    log_y: Any,
    priors: ModelPriors,
) -> None:
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
        log_y: The observed log values, as model data.
        pt_module: The imported pytensor.tensor module.
        priors: The hierarchy scales; ``likelihood_family`` is ``"contaminated"``
            (a two-component Normal mixture on point rows, so a scope or
            extraction error goes to a wide component instead of inflating
            sigma), ``"student_t"`` or ``"normal"``. Censored and bound rows are
            Normal in the contaminated family. ``bound_likelihood`` is
            ``"censored"`` or ``"exgaussian"``; the latter replaces the two
            ``pm.Censored`` terms with a one-sided excess, Gaussian whatever the
            family.

    Raises:
        ValueError: If the family or the bound likelihood is not recognised.
    """
    family = priors.likelihood_family
    if family not in ("contaminated", "student_t", "normal"):
        raise ValueError(
            f"Unknown likelihood_family {family!r}; expected contaminated, student_t or normal"
        )
    if priors.bound_likelihood not in ("censored", "exgaussian"):
        raise ValueError(
            f"Unknown bound_likelihood {priors.bound_likelihood!r}; expected censored or exgaussian"
        )

    nu: Any = None
    if family == "student_t":
        # Concentrated on small values on purpose. PyMC's Gamma takes a rate,
        # so Gamma(2, 0.1) would have mean 20, where StudentT is visually
        # Normal -- the robustness the family is here for would be asserted and
        # not delivered. Gamma(2, 0.5) has mean 4, so nu sits near 6 and the
        # tails stay genuinely heavy. Registered as a Deterministic so it
        # reaches the trace and the convergence diagnostics.
        nu = pm_module.Deterministic("nu", pm_module.Gamma("nu_raw", alpha=2.0, beta=0.5) + 2.0)

    def dist(rows: Any) -> Any:
        if family != "student_t":
            return pm_module.Normal.dist(mu=eta[rows], sigma=sd[rows])
        return pm_module.StudentT.dist(nu=nu, mu=eta[rows], sigma=sd[rows])

    def rows_data(name: str, rows: np.ndarray) -> Any:
        # The row partitions are per-observation arrays too; see build_model.
        return pm_module.Data(f"data_rows_{name}", rows)

    pi: Any = None
    if family == "contaminated" and (
        design.point_rows.size or priors.bound_likelihood == "exgaussian"
    ):
        pi = pm_module.Beta(
            "pi_contam", alpha=priors.contamination_pi_alpha, beta=priors.contamination_pi_beta
        )

    if design.point_rows.size:
        rows = rows_data("point", design.point_rows)
        if family == "contaminated":
            wide_sd = pt_module.sqrt(sd[rows] ** 2 + priors.contamination_sd**2)
            pm_module.Mixture(
                "obs_point",
                w=pt_module.stack([1.0 - pi, pi]),
                comp_dists=[
                    pm_module.Normal.dist(mu=eta[rows], sigma=sd[rows]),
                    pm_module.Normal.dist(mu=eta[rows], sigma=wide_sd),
                ],
                observed=log_y[rows],
            )
        elif family == "normal":
            pm_module.Normal("obs_point", mu=eta[rows], sigma=sd[rows], observed=log_y[rows])
        else:
            pm_module.StudentT(
                "obs_point", nu=nu, mu=eta[rows], sigma=sd[rows], observed=log_y[rows]
            )

    if priors.bound_likelihood == "exgaussian":
        # Point rows get heavy tails (student_t) or a wide component
        # (contaminated); the ExGaussian's noise is Gaussian. Match the
        # variance so a bound row is not given a tighter noise scale than a
        # point row from the same source. The tails still differ (see
        # _attach_exgaussian_bounds).
        if family == "student_t":
            bound_sd = sd * pt_module.sqrt(nu / (nu - 2.0))
        elif family == "contaminated":
            bound_sd = pt_module.sqrt(sd**2 + pi * priors.contamination_sd**2)
        else:
            bound_sd = sd
        _attach_exgaussian_bounds(pm_module, design, eta, bound_sd, log_y, priors, rows_data)
    else:
        if design.censor_lower_at.size:
            rows = rows_data("censor_lower", design.censor_lower_at)
            pm_module.Censored(
                "obs_censor_lower",
                dist(rows),
                lower=log_y[rows],
                upper=np.inf,
                observed=log_y[rows],
            )

        if design.censor_upper_at.size:
            rows = rows_data("censor_upper", design.censor_upper_at)
            pm_module.Censored(
                "obs_censor_upper",
                dist(rows),
                lower=-np.inf,
                upper=log_y[rows],
                observed=log_y[rows],
            )

    if design.zero_rows.size:
        # "One casualty or fewer". Kept rather than dropped, because zero
        # casualties is a fact, and censored rather than log1p-shifted, because
        # log1p would move the scale of every other observation.
        rows = rows_data("zero", design.zero_rows)
        pm_module.Censored(
            "obs_zero",
            dist(rows),
            lower=log_y[rows],
            upper=np.inf,
            observed=log_y[rows],
        )


def _attach_exgaussian_bounds(
    pm_module: Any,
    design: Design,
    eta: Any,
    sd: Any,
    log_y: Any,
    priors: ModelPriors,
    rows_data: Any,
) -> None:
    """Model a lone bound as a noisy report plus a one-sided excess.

    Pure censoring says the truth lies on one side of the figure and nothing
    about how far, so with many bound reports it lets the latent level drift
    away from them. Here "up to X" is ``log y = eta + noise + excess`` and "at
    least X" is ``log y = eta - excess + noise`` with ``excess`` exponential, which
    puts the truth a typical distance from the stated figure. The second case is
    fitted as ``-log y ~ ExGaussian(-eta, sd, nu)``.

    Two limits worth knowing. The ExGaussian's noise has Gaussian tails, not the
    Student-t tails of point rows, so a mis-extracted bound (a figure that is
    not a bound at all) is less robustly absorbed than a mis-extracted point;
    the caller inflates ``sd`` to the point rows' variance so the scale at
    least matches. And there is one excess rate per direction, shared by every
    side and source: "up to X" is assumed to overshoot by the same typical
    amount in a Roman and a modern report.

    Args:
        pm_module: The imported pymc module.
        design: The design; ``censor_lower_at`` holds the "up to X" rows and
            ``censor_upper_at`` the "at least X" rows (see the module
            docstring of :mod:`pipeline.reconcilers.design`).
        eta: The linear predictor.
        sd: The per-observation scale, already matched to the point rows'
            noise variance.
        log_y: The observed log values, as model data.
        priors: Supplies ``bound_excess_prior_sd``.
        rows_data: Registers a row partition as model data.
    """
    if design.censor_lower_at.size:
        rows = rows_data("censor_lower", design.censor_lower_at)
        lambda_upper = pm_module.HalfNormal("lambda_upper", sigma=priors.bound_excess_prior_sd)
        pm_module.ExGaussian(
            "obs_bound_upper",
            mu=eta[rows],
            sigma=sd[rows],
            nu=lambda_upper,
            observed=log_y[rows],
        )

    if design.censor_upper_at.size:
        rows = rows_data("censor_upper", design.censor_upper_at)
        lambda_lower = pm_module.HalfNormal("lambda_lower", sigma=priors.bound_excess_prior_sd)
        pm_module.ExGaussian(
            "obs_bound_lower",
            mu=-eta[rows],
            sigma=sd[rows],
            nu=lambda_lower,
            observed=-log_y[rows],
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
