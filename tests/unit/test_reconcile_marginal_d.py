"""Troop reparameterisation A (centred battle effect) and B (side deviation marginalised).

A: ``zb`` is centred within each era bin of dated battles, so the bin mean of ``b`` is zero and
``e`` carries the whole period mean. B: on a troop side with exactly one point row under the
Normal likelihood, ``zd`` has no likelihood term and the row's variance gains ``s_d**2``;
``mu_side`` is then drawn from the exact conditional ``d | theta, y``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from pipeline.reconcilers import model as model_module
from pipeline.reconcilers.design import UNDATED_ERA_BIN, Design, build_design
from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")

_NORMAL = ModelPriors(likelihood_family="normal")


def _report(
    report_id: int,
    *,
    side_id: int,
    battle_id: int,
    year: int | None,
    source: int | None = None,
    quantity: str = "troops",
    upper: bool = False,
) -> Report:
    src = report_id if source is None else source
    kind = ("wikipedia_infobox", "wikidata")[report_id % 2]
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=src,
        source_key=SourceKey(kind, f"https://example/{kind}/{src}"),
        source_type=kind,
        quantity=quantity,  # type: ignore[arg-type]
        reported_value=5_000.0 + 731.0 * report_id,
        scope="engaged" if quantity == "troops" else "unknown",
        year_astronomical=year,
        claim_regime="unlabelled",
        lineage_id=report_id,
        is_upper_bound=upper,
    )


def _design(reports: list[Report], quantity: str = "troops") -> Design:
    return build_design(reports, quantity=quantity, ancient_cutoff_year=500)


def _three_side_design() -> Design:
    reports = [
        _report(1, side_id=1, battle_id=1, year=1200),
        _report(2, side_id=2, battle_id=1, year=1200),
        _report(3, side_id=3, battle_id=2, year=1700),
    ]
    return _design(reports)


def _explicit(monkeypatch: pytest.MonkeyPatch, design: Design) -> Any:
    with monkeypatch.context() as patch:
        patch.setattr(
            model_module, "_marginal_side_mask", lambda d, p: np.zeros(d.n_sides, dtype=bool)
        )
        return build_model(design, priors=_NORMAL, inflation=InflationPrior())


def _compiled(model: Any, outputs: list[Any]) -> Any:
    import pytensor

    graphs = model.replace_rvs_by_values(outputs)
    fn = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")

    def run(point: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(x) for x in fn(*[point[v.name] for v in model.value_vars])]

    return run


def _random_point(model: Any, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: v + rng.normal(0.0, 0.6, size=np.shape(v)) for k, v in model.initial_point().items()}


# --- A: the projector and the centred battle effect -------------------------


def _mixed_battles() -> Design:
    years = [-400, -400, -380, 1200, 1700, 1700, None, None]
    battles = [1, 1, 2, 3, 4, 4, 5, 6]
    reports = [
        _report(i + 1, side_id=i + 1, battle_id=battles[i], year=years[i]) for i in range(8)
    ]
    return _design(reports)


def _battle_bins(design: Design) -> np.ndarray:
    bins = np.full(design.n_battles, UNDATED_ERA_BIN)
    bins[design.battle_of_side] = design.era_bin_of_side
    return bins


def test_the_bin_projector_averages_dated_battles_within_their_bin_and_skips_undated() -> None:
    design = _mixed_battles()
    onehot, mean_w = model_module._battle_bin_projector(design)

    n_bins = len(model_module.ERA_BIN_LABELS)
    assert onehot.shape == (design.n_battles, n_bins)
    assert mean_w.shape == onehot.shape
    undated = _battle_bins(design) == UNDATED_ERA_BIN
    assert undated.any()
    assert onehot[undated].sum() == 0.0
    assert (onehot[~undated].sum(axis=1) == 1.0).all()
    zb = np.random.default_rng(0).normal(size=design.n_battles)
    centred = zb - onehot @ (zb @ mean_w)
    assert centred[undated] == pytest.approx(zb[undated])
    for k in range(n_bins):
        members = onehot[:, k] == 1
        if members.any():
            assert mean_w[:, k].sum() == pytest.approx(1.0)
            assert centred[members].mean() == pytest.approx(0.0, abs=1e-12)


def test_the_battle_effect_has_zero_mean_in_each_era_bin_and_undated_battles_are_untouched() -> (
    None
):
    design = _mixed_battles()
    model = build_model(design, priors=_NORMAL, inflation=InflationPrior())
    point = _random_point(model)
    b, zb, s_b = _compiled(model, [model["b"], model["zb"], model["s_b"]])(point)

    bins = _battle_bins(design)
    for k in sorted(set(bins.tolist()) - {UNDATED_ERA_BIN}):
        assert b[bins == k].mean() == pytest.approx(0.0, abs=1e-10)
    assert (bins == UNDATED_ERA_BIN).any()
    assert b[bins == UNDATED_ERA_BIN] == pytest.approx(float(s_b) * zb[bins == UNDATED_ERA_BIN])
    # The registered deterministic itself, not just its bin means: b = s_b * (zb - binmean(zb))
    # for dated battles (bin means taken directly from the bins, independent of the projector)
    # and b = s_b * zb for undated ones.
    expected = float(s_b) * zb
    for k in sorted(set(bins.tolist()) - {UNDATED_ERA_BIN}):
        in_bin = bins == k
        expected[in_bin] = float(s_b) * (zb[in_bin] - zb[in_bin].mean())
    assert b == pytest.approx(expected)


def test_no_era_effect_means_no_centring() -> None:
    design = _mixed_battles()
    priors = dataclasses.replace(_NORMAL, era_effect="none")
    model = build_model(design, priors=priors, inflation=InflationPrior())
    point = _random_point(model)
    b, zb, s_b = _compiled(model, [model["b"], model["zb"], model["s_b"]])(point)
    assert b == pytest.approx(float(s_b) * zb)


# --- B: the mask ------------------------------------------------------------


def _mask_design() -> Design:
    reports = [
        _report(1, side_id=1, battle_id=1, year=1200),  # one point row: marginal
        _report(2, side_id=2, battle_id=1, year=1200, source=20),  # two rows: explicit
        _report(3, side_id=2, battle_id=1, year=1200, source=30),
        _report(4, side_id=3, battle_id=2, year=1700, upper=True),  # lone bound: explicit
        _report(5, side_id=4, battle_id=2, year=1700),  # one point row: marginal
    ]
    return _design(reports)


def test_the_mask_marks_only_single_point_row_sides() -> None:
    design = _mask_design()
    mask = model_module._marginal_side_mask(design, _NORMAL)
    by_id = dict(zip(design.side_ids, mask.tolist(), strict=True))
    assert by_id == {1: True, 2: False, 3: False, 4: True}


@pytest.mark.parametrize("family", ["student_t", "contaminated"])
def test_the_mask_is_empty_for_other_likelihood_families(family: str) -> None:
    priors = dataclasses.replace(_NORMAL, likelihood_family=family)
    assert not model_module._marginal_side_mask(_mask_design(), priors).any()


def test_the_mask_is_empty_for_casualties() -> None:
    reports = [
        _report(i + 1, side_id=i + 1, battle_id=i + 1, year=1200 + i, quantity="casualties")
        for i in range(3)
    ]
    design = _design(reports, quantity="casualties")
    assert not model_module._marginal_side_mask(design, _NORMAL).any()


# --- B: conditional mean and variance of d ----------------------------------


def test_the_marginalised_deviation_has_the_exact_conditional_mean_and_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _three_side_design()
    marg = build_model(design, priors=_NORMAL, inflation=InflationPrior())
    expl = _explicit(monkeypatch, design)
    point = _random_point(expl)
    zero = {**point, "zd": np.zeros_like(point["zd"])}

    mu_expl0, eta, row_sd, s_d = _compiled(
        expl, [expl["mu_side"], expl.row_eta, expl.row_sd, expl["s_d"]]
    )(zero)
    run_marg = _compiled(marg, [marg["mu_side"]])
    mu0 = run_marg(zero)[0]

    r = design.log_y - eta
    v = row_sd**2
    k = float(s_d) ** 2 / (float(s_d) ** 2 + v)
    rows = np.empty(design.n_sides, dtype=int)
    rows[design.side_index] = np.arange(design.n_obs)
    mean = (k * r)[rows]
    var = (k * v)[rows]

    assert mu0 - mu_expl0 == pytest.approx(mean, abs=1e-9)
    slopes = np.array(
        [
            run_marg({**zero, "zd": np.eye(design.n_sides)[j]})[0][j] - mu0[j]
            for j in range(design.n_sides)
        ]
    )
    assert slopes**2 == pytest.approx(var, abs=1e-9)

    # Mean and variance of d over zd ~ N(0, 1), compiled graph, many draws.
    draws = np.random.default_rng(5).standard_normal((4_000, design.n_sides))
    d_draws = np.array(
        [run_marg({**zero, "zd": z})[0] - mu_expl0 for z in draws[:400]]
    )
    assert d_draws.mean(axis=0) == pytest.approx(mean, abs=0.1 * float(np.sqrt(var).max()) + 0.02)
    assert d_draws.var(axis=0) == pytest.approx(var, rel=0.25)


# --- B: marginal log p against explicit quadrature ---------------------------


def test_the_marginal_model_logp_equals_the_explicit_logp_integrated_over_zd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _three_side_design()
    marg = build_model(design, priors=_NORMAL, inflation=InflationPrior())
    expl = _explicit(monkeypatch, design)
    point = _random_point(expl, seed=8)
    zero = {**point, "zd": np.zeros_like(point["zd"])}

    logp_expl = expl.compile_logp()
    logp_marg = marg.compile_logp()
    grid = np.linspace(-9.0, 9.0, 3601)
    step = grid[1] - grid[0]

    # Given theta the sides are independent, so the 3-d integral is a product of 1-d ones.
    base = float(logp_expl(zero))
    log_integral = base
    for j in range(design.n_sides):
        vals = []
        for g in grid:
            z = np.zeros(design.n_sides)
            z[j] = g
            vals.append(float(logp_expl({**zero, "zd": z})) - base)
        log_integral += float(np.log(np.sum(np.exp(vals)) * step))

    # The marginal model keeps zd's N(0, 1) prior as a free variable with no likelihood term;
    # at zd = 0 that prior contributes -0.5 log(2 pi) per side, which integrates out to 1.
    log_prior_zd_at_zero = -0.5 * np.log(2.0 * np.pi) * design.n_sides
    marginal = float(logp_marg(zero)) - log_prior_zd_at_zero
    assert marginal == pytest.approx(log_integral, abs=1e-5)


# --- B: the prior predictive check must not read the data --------------------


def test_the_prior_check_strength_does_not_depend_on_the_observed_values() -> None:
    from pipeline.reconcilers.prior_check import prior_predictive_check

    def strength(shift: float) -> dict[str, float | None]:
        design = _three_side_design()
        design = dataclasses.replace(design, log_y=design.log_y + shift)
        model = build_model(design, priors=_NORMAL, inflation=InflationPrior())
        assert "data_side_marginal" in model.named_vars
        block = prior_predictive_check(model, design, draws=300, seed=4)
        return block["strength_quantiles"]  # type: ignore[no-any-return]

    assert strength(0.0) == strength(6.0)
