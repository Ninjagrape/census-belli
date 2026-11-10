"""The lone-branch offset: flag reaches the design, the variable is guarded, row location shifts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")


def _report(report_id: int, side_id: int, *, lone: bool, quantity: str = "troops") -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", f"https://example/{report_id}"),
        source_type="wikipedia_infobox",
        quantity=quantity,  # type: ignore[arg-type]
        reported_value=4_000.0 + 311.0 * report_id,
        scope="engaged" if quantity == "troops" else "unknown",
        year_astronomical=1850,
        claim_regime="modern_scholarly",
        lineage_id=report_id,
        lone_branch=lone,
    )


def _design(flags: list[bool], quantity: str = "troops", sides: list[int] | None = None) -> Design:
    sides = sides or list(range(1, len(flags) + 1))
    reports = [_report(i + 1, sides[i], lone=f, quantity=quantity) for i, f in enumerate(flags)]
    return build_design(reports, quantity=quantity, ancient_cutoff_year=500)


def _eval(model: Any, outputs: list[Any], point: dict[str, np.ndarray]) -> list[np.ndarray]:
    import pytensor

    graphs = model.replace_rvs_by_values(outputs)
    compiled = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")
    return [np.asarray(x) for x in compiled(*[point[v.name] for v in model.value_vars])]


def _model(design: Design, priors: ModelPriors | None = None) -> Any:
    return build_model(design, priors=priors or ModelPriors(), inflation=InflationPrior())


def test_the_flag_reaches_the_design_rows() -> None:
    design = _design([True, False, True, False])

    assert design.lone_branch_rows.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert design.has_lone_branch


def test_a_design_without_the_array_reads_as_unflagged() -> None:
    design = replace(_design([True, False, False]), lone_branch=None)

    assert not design.has_lone_branch


def test_casualty_designs_ignore_the_flag() -> None:
    assert not _design([True, False, False], quantity="casualties").has_lone_branch


def test_the_variable_exists_only_when_a_row_is_flagged() -> None:
    flagged = _model(_design([True, False, False, False]))
    plain = _model(_design([False, False, False, False]))

    assert "d_lone_branch" in {rv.name for rv in flagged.free_RVs}
    assert flagged.initial_point()["d_lone_branch_interval__"].shape == (1,)
    assert "d_lone_branch" not in plain.named_vars
    assert "data_lone_branch" not in plain.named_vars


def test_the_priors_default_to_log_one_and_a_half_and_read_from_params() -> None:
    default = ModelPriors.from_params({}, quantity="troops")
    custom = ModelPriors.from_params(
        {"lone_branch_offset_prior_mu": 0.2, "lone_branch_offset_prior_sd": 0.3}, quantity="troops"
    )

    assert default.lone_branch_offset_prior_mu == pytest.approx(np.log(1.5), abs=1e-6)
    assert default.lone_branch_offset_prior_sd == 0.6
    assert (custom.lone_branch_offset_prior_mu, custom.lone_branch_offset_prior_sd) == (0.2, 0.3)


def test_the_offset_is_non_negative_in_the_prior() -> None:
    import pymc as pm

    model = _model(_design([True, False, False, False]))
    with model:
        prior = pm.sample_prior_predictive(draws=200, var_names=["d_lone_branch"], random_seed=1)

    values = prior.prior["d_lone_branch"].to_numpy()
    assert (values >= 0.0).all()


def test_the_variable_has_one_entry_per_flagged_side_and_its_rows_share_it() -> None:
    # Sides 1 and 3 are flagged (side 1 on two rows), side 2 is not.
    design = _design([True, True, False, True, False], sides=[1, 1, 2, 3, 4])
    model = _model(design)

    assert model.initial_point()["d_lone_branch_interval__"].shape == (2,)
    assert design.lone_branch_offset_index[:2].tolist() == [0, 0]
    assert design.lone_branch_offset_index[3] == 1

    base = model.initial_point()
    moved = {k: v.copy() for k, v in base.items()}
    key = next(k for k in moved if k.startswith("d_lone_branch"))
    moved[key] = moved[key] + np.array([0.7, 0.0])
    eta_a, d_a = _eval(model, [model.row_eta, model["d_lone_branch"]], base)
    eta_b, d_b = _eval(model, [model.row_eta, model["d_lone_branch"]], moved)
    shift = eta_b - eta_a
    delta_d = d_b - d_a

    # Rows 0 and 1 (side 1) move together by -delta_d[0]; row 3 (side 3) and the
    # unflagged rows share the common rotation shift.
    assert shift[0] == pytest.approx(shift[1], abs=1e-12)
    assert shift[0] - shift[2] == pytest.approx(-delta_d[0], abs=1e-10)
    assert shift[3] == pytest.approx(shift[2], abs=1e-12)
    assert shift[2] == pytest.approx(shift[4], abs=1e-12)


def test_a_flagged_row_is_shifted_down_by_the_offset_relative_to_an_unflagged_row() -> None:
    design = _design([True, False, False, False])
    model = _model(design)
    base = model.initial_point()
    moved = {k: v.copy() for k, v in base.items()}
    key = next(k for k in moved if k.startswith("d_lone_branch"))
    moved[key] = moved[key] + 0.7

    eta_a, d_a = _eval(model, [model.row_eta, model["d_lone_branch"]], base)
    eta_b, d_b = _eval(model, [model.row_eta, model["d_lone_branch"]], moved)
    shift = eta_b - eta_a
    delta_d = float((d_b - d_a)[0])

    assert abs(delta_d) > 1e-3
    # The corpus-level rotation moves every row by the same amount; the difference
    # between a flagged and an unflagged row is exactly the change in d_lone_branch.
    assert shift[0] - shift[1] == pytest.approx(-delta_d, abs=1e-10)
    assert shift[1] == pytest.approx(shift[2], abs=1e-12)


def test_summarises_row_location_includes_the_offset() -> None:
    import pytensor

    from pipeline.reconcilers.summarise import _row_eta_and_var

    design = _design([True, False, True, False])
    model = _model(design)
    names = [
        n
        for n in (
            "mu_side",
            "beta_source",
            "g_regime",
            "delta_level",
            "a_era",
            "d_lone_branch",
            "sigma_source",
            "sigma_source_ancient",
        )
        if n in model.named_vars
    ]
    graphs = model.replace_rvs_by_values([model[n] for n in names] + [model.row_eta, model.row_sd])
    compiled = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")
    point = model.initial_point()
    outputs = compiled(*[point[v.name] for v in model.value_vars])
    values = {n: np.asarray(v)[None] for n, v in zip(names, outputs[: len(names)], strict=True)}

    eta, _ = _row_eta_and_var(lambda n: values[n], design, np.arange(design.n_obs))

    np.testing.assert_allclose(eta[0], outputs[-2], rtol=0, atol=1e-10)
