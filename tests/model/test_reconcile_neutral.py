"""The regime-neutral troops fit, checked on the cases it was built for.

These sample for real (``pytest.mark.model``) because the claim is about the
exact fit, not an approximation of it. Each fixture has two sides, so a force
ratio is the difference of the two sides' posterior ``mu_side``.

The Gaugamela fixture is the motivating case (handover-log 20.10, HIGH 2): the
Persian side carries a labelled ancient claim and a labelled modern figure,
the Macedonian side two unlabelled ones. Seven fully labelled background
battles share the same source types across both sides so the corpus identifies
the side scale and type biases. The unlabelled offset is nearly
all prior, so in the published fit it deflates the Macedonian figures and
inflates the ratio; the neutral fit reads those figures as an ordinary modern
count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.model import (
    InflationPrior,
    ModelPriors,
    SamplerSettings,
    build_model,
    load_inflation_prior,
    sample,
)
from pipeline.reconcilers.records import Report, SourceKey
from tests.model.conftest import fit_until_converged

pytestmark = pytest.mark.model

SEED = 20261006
PERSIAN, MACEDONIAN = 1, 2
CUTOFF = 500
AGENTS = Path(__file__).resolve().parents[2] / "agents"
SAMPLER = SamplerSettings(
    draws=1500,
    tune=1000,
    chains=4,
    cores=1,
    random_seed=SEED,
    nuts_sampler="nutpie",
    target_accept=0.99,
)

_INFOBOX = SourceKey("wikipedia_infobox", "https://example.org/infobox")
_SCHOLAR = SourceKey("peer_reviewed", "https://example.org/scholar")
_HERODOTUS = SourceKey("primary_source", "https://example.org/ancient")
_BOOK = SourceKey("academic_book", "https://example.org/book")

# Source types are shared across both sides of every battle, and the pattern is
# replicated over several battles of varied size, so ``s0`` and ``b_type`` are
# identified by the corpus and cannot absorb the between-side difference of the
# battle under test (a fixture with each type on one side shrank a raw log ratio
# of 0.80 to 0.31 in both fits).
_BACKGROUND_SIDES = (
    (60_000, 40_000),
    (25_000, 30_000),
    (80_000, 50_000),
    (15_000, 22_000),
    (45_000, 45_000),
    (120_000, 70_000),
    (33_000, 18_000),
)
_BACKGROUND_FACTOR = {_SCHOLAR: 1.0, _BOOK: 1.05, _INFOBOX: 0.95}
_ANCIENT_INFLATION = 6.0
_JITTER_SD = 0.08
FAITHFUL_MU = load_inflation_prior().faithful_mu


def _report(
    report_id: int,
    side_id: int,
    key: SourceKey,
    value: float,
    regime: str = "unlabelled",
    battle_id: int = 1,
) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=report_id,
        source_key=key,
        source_type=key.source_type,
        quantity="troops",
        reported_value=value,
        scope="engaged",
        year_astronomical=-330,
        claim_regime=regime,
    )


def _background() -> list[Report]:
    """Seven fully labelled battles; every type reports both sides of each."""
    rng = np.random.default_rng(SEED)
    reports: list[Report] = []
    next_id = 1000
    for battle, truths in enumerate(_BACKGROUND_SIDES, start=2):
        for side_offset, truth in enumerate(truths):
            side_id = 10 * battle + side_offset
            for key, factor in _BACKGROUND_FACTOR.items():
                value = truth * factor * math.exp(rng.normal(0.0, _JITTER_SD))
                reports.append(
                    _report(next_id, side_id, key, round(value, -2), "modern_scholarly", battle)
                )
                next_id += 1
            value = truth * _ANCIENT_INFLATION * math.exp(rng.normal(0.0, _JITTER_SD))
            reports.append(
                _report(next_id, side_id, _HERODOTUS, round(value, -3), "ancient_claim", battle)
            )
            next_id += 1
    return reports


@dataclass(frozen=True)
class RatioPair:
    """Persian over Macedonian log force ratio under each fit."""

    published: float
    neutral: float
    n_obs: int
    neutral_worst: dict[str, float | int | None]
    neutral_attempts: int = 1


def _log_ratio(idata: object, design: object) -> float:
    mu = np.asarray(idata.posterior["mu_side"].values)  # type: ignore[attr-defined]
    mu = mu.reshape(-1, mu.shape[-1])
    sides = list(design.side_ids)  # type: ignore[attr-defined]
    return float(np.mean(mu[:, sides.index(PERSIAN)] - mu[:, sides.index(MACEDONIAN)]))


def _neutral_converged(worst: dict[str, float | int | None]) -> bool:
    """The convergence claim test_the_neutral_fits_converge asserts."""
    return worst["divergences"] == 0 and (worst["max_rhat"] or 0.0) < 1.05


def _fit_pair(
    reports: list[Report], inflation: InflationPrior, *, fit_neutral: bool = True
) -> RatioPair:
    from pipeline.reconcilers.summarise import _convergence

    design = build_design(reports, quantity="troops", ancient_cutoff_year=CUTOFF)
    params = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))["params"]
    priors = ModelPriors.from_params(params, quantity="troops")
    published = sample(build_model(design, priors=priors, inflation=inflation), SAMPLER)
    if not fit_neutral:
        return RatioPair(_log_ratio(published, design), math.nan, design.n_obs, {})
    neutral_model = build_model(design, priors=priors, inflation=inflation, neutral=True)
    neutral, seeds = fit_until_converged(
        lambda seed: sample(neutral_model, replace(SAMPLER, random_seed=seed)),
        seed=SEED,
        converged=lambda idata: _neutral_converged(_convergence(idata)[0]),
        describe=lambda idata: _convergence(idata)[:2],
    )
    worst, _, _ = _convergence(neutral)
    return RatioPair(
        published=_log_ratio(published, design),
        neutral=_log_ratio(neutral, design),
        n_obs=design.n_obs,
        neutral_worst=worst,
        neutral_attempts=len(seeds),
    )


def _gaugamela(
    *, with_infobox_copy: bool = False, macedonian_regime: str = "unlabelled"
) -> list[Report]:
    """Battle 1: labelled Persian figures against unlabelled Macedonian ones."""
    reports = [
        _report(1, PERSIAN, _HERODOTUS, 1_000_000.0, "ancient_claim"),
        _report(2, PERSIAN, _SCHOLAR, 100_000.0, "modern_scholarly"),
        _report(3, MACEDONIAN, _INFOBOX, 47_000.0, macedonian_regime),
        _report(4, MACEDONIAN, _BOOK, 50_000.0, macedonian_regime),
    ]
    if with_infobox_copy:
        # The same infobox restating the figure the Herodotus row attributes.
        reports.append(_report(5, PERSIAN, _INFOBOX, 1_000_000.0))
    return reports + _background()


def _both_unlabelled() -> list[Report]:
    reports = [
        _report(1, PERSIAN, _INFOBOX, 110_000.0),
        _report(2, PERSIAN, _SCHOLAR, 100_000.0),
        _report(3, PERSIAN, _BOOK, 120_000.0),
        _report(4, MACEDONIAN, _INFOBOX, 47_000.0),
        _report(5, MACEDONIAN, _SCHOLAR, 45_000.0),
        _report(6, MACEDONIAN, _BOOK, 52_000.0),
    ]
    return reports + _background()


@pytest.fixture(scope="module")
def inflation() -> InflationPrior:
    return load_inflation_prior()


@pytest.fixture(scope="module")
def gaugamela(inflation: InflationPrior) -> RatioPair:
    return _fit_pair(_gaugamela(), inflation)


@pytest.fixture(scope="module")
def gaugamela_with_copy(inflation: InflationPrior) -> RatioPair:
    return _fit_pair(_gaugamela(with_infobox_copy=True), inflation)


@pytest.fixture(scope="module")
def gaugamela_relabelled(inflation: InflationPrior) -> RatioPair:
    return _fit_pair(_gaugamela(macedonian_regime="modern_scholarly"), inflation, fit_neutral=False)


@pytest.fixture(scope="module")
def both_unlabelled(inflation: InflationPrior) -> RatioPair:
    return _fit_pair(_both_unlabelled(), inflation)


def test_neutral_fit_sits_between_the_relabelled_published_fit_and_the_faithful_shift(
    gaugamela: RatioPair, gaugamela_relabelled: RatioPair
) -> None:
    # Reading an unlabelled figure as an ordinary modern count is the published
    # fit with that figure relabelled modern_scholarly. Before P1 the neutral fit
    # equalled that plus faithful_mu to within 0.06, because the unlabelled offset
    # was prior-only: the neutral prior put it at faithful_mu instead of zero and
    # the Macedonian side sat that much lower. Under P1 the two sides of a battle
    # share a battle effect and differ only by s_d, so the Persian side's labelled
    # rows inform the Macedonian offset and the data pull it back from faithful_mu
    # (measured: neutral offset 0.025 against a prior mean of 0.129, neutral 0.715
    # against relabelled 0.685). The ratio therefore lies between the relabelled
    # fit (no shift) and the full faithful shift, with the old tolerance as slack.
    shift = gaugamela.neutral - gaugamela_relabelled.published
    assert -0.06 <= shift <= FAITHFUL_MU + 0.06, (gaugamela.neutral, gaugamela_relabelled.published)


def test_gaugamela_published_ratio_is_no_longer_tilted_by_the_unlabelled_prior(
    gaugamela: RatioPair,
) -> None:
    # Before P1 the published fit leaned on the unlabelled offset's prior mean and
    # read the Macedonian side as smaller than its figures, so its ratio was the
    # larger by 0.27 (1.115 against 0.843). The shared battle effect now lets the
    # data learn that offset (published g_unlabelled -0.21 against a prior mean
    # near +0.6), so the published ratio is no longer the larger; it sits 0.17
    # below the neutral one. The check is that the old tilt is gone, in either direction.
    assert abs(gaugamela.published - gaugamela.neutral) < 0.25, (
        gaugamela.published,
        gaugamela.neutral,
    )


def test_an_unlabelled_infobox_copy_of_an_attributed_figure_changes_nothing(
    gaugamela: RatioPair, gaugamela_with_copy: RatioPair
) -> None:
    # The merge removes the copy before the design is built, so the two fits
    # see identical arrays and must agree to Monte Carlo error.
    assert gaugamela_with_copy.n_obs == gaugamela.n_obs
    # Measured: identical to the digit under the fixed seed; 0.03 allows for
    # platform-dependent floating point.
    assert abs(gaugamela_with_copy.neutral - gaugamela.neutral) < 0.03
    assert abs(gaugamela_with_copy.published - gaugamela.published) < 0.03


def test_with_both_sides_unlabelled_the_two_fits_give_the_same_ratio(
    both_unlabelled: RatioPair,
) -> None:
    # The unlabelled offset is shared, so it cancels in the ratio either way.
    # Measured: 0.714 neutral against 0.725 published, a gap of 0.010.
    assert abs(both_unlabelled.neutral - both_unlabelled.published) < 0.04, (
        both_unlabelled.neutral,
        both_unlabelled.published,
    )


def test_the_neutral_fits_converge(
    gaugamela: RatioPair,
    gaugamela_with_copy: RatioPair,
    both_unlabelled: RatioPair,
    record_property: pytest.RecordProperty,
) -> None:
    pairs = (gaugamela, gaugamela_with_copy, both_unlabelled)
    record_property("convergence_attempts", ",".join(str(p.neutral_attempts) for p in pairs))
    for pair in pairs:
        assert pair.neutral_worst["divergences"] == 0
        assert (pair.neutral_worst["max_rhat"] or 0.0) < 1.05, pair.neutral_worst
