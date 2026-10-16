"""Source-effect nesting: which source keys get their own bias and noise terms.

On the real corpus every source key is one battle's Wikipedia infobox, so a
per-key bias is the battle level under another name. It absorbs the side
spread, and the published estimate (which leaves the source term out) goes
flat. A key therefore gets its own effect only when it co-reports: another key
reports the same sides in enough battles for the effect to be distinguishable
from the battle it sits in.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")

from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model  # noqa: E402

_REPORT_IDS = itertools.count(1)


def _report(
    *,
    side_id: int,
    battle_id: int,
    url: str,
    source_type: str = "wikipedia_infobox",
) -> Report:
    report_id = next(_REPORT_IDS)
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=report_id,
        source_key=SourceKey(source_type, url),
        source_type=source_type,
        quantity="troops",
        reported_value=5_000.0 + 100.0 * report_id,
        year_astronomical=1800,
        claim_regime="unlabelled",
        lineage_id=report_id,
    )


def _keys_on_battles(
    urls: list[str], battles: range, *, source_type: str = "wikipedia_infobox"
) -> list[Report]:
    """Every key in ``urls`` reporting both sides of every battle in ``battles``."""
    return [
        _report(side_id=100 * battle + side, battle_id=battle, url=url, source_type=source_type)
        for battle in battles
        for side in (1, 2)
        for url in urls
    ]


def _one_key(
    n_battles: int, *, url: str = "https://example/shared", first: int = 0
) -> list[Report]:
    """One source key alone on both sides of ``n_battles`` battles."""
    return _keys_on_battles([url], range(first, first + n_battles))


def _co_reporting_keys(n_keys: int, n_battles: int = 3, *, first: int = 0) -> list[Report]:
    """``n_keys`` keys, each pair of neighbours sharing ``n_battles`` battles.

    Keys come in pairs reporting the same sides, so every key co-reports in
    ``n_battles`` battles.
    """
    reports: list[Report] = []
    for pair in range(n_keys // 2):
        reports += _keys_on_battles(
            [f"https://example/{2 * pair}", f"https://example/{2 * pair + 1}"],
            range(first + pair * n_battles, first + (pair + 1) * n_battles),
        )
    return reports


def _prior(model: object) -> object:
    import pymc as pm

    return pm.sample_prior_predictive(draws=1, model=model, random_seed=1).prior


def _design(reports: list[Report], **kwargs: int) -> Design:
    return build_design(reports, quantity="troops", ancient_cutoff_year=500, **kwargs)


def _flags(design: Design) -> dict[str, bool]:
    return dict(
        zip(
            (k.url for k in design.source_keys),
            design.source_has_own_effect.tolist(),
            strict=True,
        )
    )


# ─── The flag in the design ──────────────────────────────────────────────────


def test_a_source_key_confined_to_one_battle_is_pooled() -> None:
    design = _design(_one_key(1))

    assert design.source_has_own_effect.tolist() == [False]
    assert design.n_source_keys_own_effect == 0
    assert design.n_source_keys_pooled == 1


def test_a_key_alone_on_its_sides_is_pooled_however_many_battles_it_spans() -> None:
    # Its bias would be the side levels under another name: nothing else
    # reports those sides to tell the two apart.
    design = _design(_one_key(6))

    assert design.source_has_own_effect.tolist() == [False]


def test_keys_co_reporting_in_enough_battles_get_their_own_effect() -> None:
    design = _design(_co_reporting_keys(2, n_battles=3))

    assert design.source_has_own_effect.tolist() == [True, True]
    assert design.n_source_keys_own_effect == 2
    assert design.n_source_keys_pooled == 0


def test_the_battle_threshold_is_a_parameter() -> None:
    reports = _co_reporting_keys(2, n_battles=2)

    assert _design(reports).source_has_own_effect.tolist() == [False, False]
    assert _design(reports, min_battles_per_source_effect=2).source_has_own_effect.tolist() == [
        True,
        True,
    ]


def test_both_sides_of_one_battle_count_as_one_battle() -> None:
    # 132 of the corpus's 148 keys cover both sides of a single battle; counting
    # sides instead of battles would qualify them all.
    design = _design(_co_reporting_keys(2, n_battles=2))

    assert design.n_source_keys_own_effect == 0


def test_only_the_battles_where_another_key_reports_the_same_side_count() -> None:
    # Key "a" spans four battles but shares sides with "b" in two of them.
    reports = [
        *_keys_on_battles(["https://example/a", "https://example/b"], range(0, 2)),
        *_keys_on_battles(["https://example/a"], range(2, 4)),
    ]

    assert _flags(_design(reports)) == {"https://example/a": False, "https://example/b": False}
    assert _flags(_design(reports, min_battles_per_source_effect=2)) == {
        "https://example/a": True,
        "https://example/b": True,
    }


def test_another_key_on_a_different_side_of_the_battle_is_not_co_reporting() -> None:
    reports = [_report(side_id=1, battle_id=b, url="https://example/a") for b in range(3)] + [
        _report(side_id=2, battle_id=b, url="https://example/b") for b in range(3)
    ]

    assert not _design(reports).source_has_own_effect.any()


def test_the_flag_is_aligned_with_source_keys() -> None:
    wide = _co_reporting_keys(2, n_battles=4)
    narrow = _one_key(1, url="https://example/narrow", first=50)
    design = _design([*wide, *narrow])

    assert _flags(design) == {
        "https://example/0": True,
        "https://example/1": True,
        "https://example/narrow": False,
    }
    assert design.n_source_keys_own_effect + design.n_source_keys_pooled == design.n_sources


# ─── What the model builds from it ───────────────────────────────────────────


def test_no_source_effect_is_created_when_no_key_qualifies() -> None:
    # A zero-size or unidentified u_source would sample its prior and report
    # clean convergence on nothing.
    reports = [r for k in range(4) for r in _one_key(1, url=f"https://example/{k}", first=k)]
    design = _design(reports)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    assert design.n_source_keys_own_effect == 0
    for name in ("u_source", "tau_u", "tau_sigma", "z_sigma"):
        assert name not in model.named_vars
    assert "beta_source" in model.named_vars
    assert "sigma_source" in model.named_vars


def test_the_deviation_scales_are_fixed_below_the_hierarchical_floor() -> None:
    design = _design(_co_reporting_keys(4))
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    assert design.n_source_keys_own_effect == 4
    assert "u_source" in model.named_vars
    assert "z_sigma" in model.named_vars
    assert "tau_u" not in model.named_vars
    assert "tau_sigma" not in model.named_vars


def test_the_deviation_scales_become_hyperparameters_at_the_floor() -> None:
    design = _design(_co_reporting_keys(6))
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    assert design.n_source_keys_own_effect == 6
    for name in ("u_source", "tau_u", "tau_sigma"):
        assert name in model.named_vars


def test_the_floor_is_a_parameter() -> None:
    design = _design(_co_reporting_keys(4))
    priors = ModelPriors.from_params({"min_keys_for_hierarchical_deviation": 4}, quantity="troops")
    model = build_model(design, priors=priors, inflation=InflationPrior())

    assert "tau_u" in model.named_vars
    assert "tau_sigma" in model.named_vars


def test_a_pooled_key_carries_the_type_effect_and_type_noise_only() -> None:
    reports = [
        *_co_reporting_keys(4, n_battles=3),
        *_one_key(1, url="https://example/lonely", first=90),
    ]
    design = _design(reports)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    lonely = [k.url for k in design.source_keys].index("https://example/lonely")
    assert not bool(design.source_has_own_effect[lonely])
    own = [i for i in range(design.n_sources) if i != lonely]

    prior = _prior(model)
    u = prior["u_source"].values[0, 0]
    z = prior["z_sigma"].values[0, 0]
    assert float(u[lonely]) == 0.0
    assert float(z[lonely]) == 0.0
    assert any(float(u[i]) != 0.0 for i in own)


def test_a_pooled_key_shares_the_type_level_noise_scale() -> None:
    reports = [
        *_co_reporting_keys(4, n_battles=3),
        *_one_key(1, url="https://example/lonely", first=90),
        *_one_key(1, url="https://example/lonely-too", first=95),
    ]
    design = _design(reports)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    urls = [k.url for k in design.source_keys]
    a, b = urls.index("https://example/lonely"), urls.index("https://example/lonely-too")
    sigma = _prior(model)["sigma_source"].values[0, 0]

    assert float(sigma[a]) == pytest.approx(float(sigma[b]))


def test_a_type_with_only_own_effect_keys_keeps_the_zero_sum() -> None:
    design = _design(_co_reporting_keys(4))
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    u = np.asarray(_prior(model)["u_source"].values[0, 0])
    z = np.asarray(_prior(model)["z_sigma"].values[0, 0])

    assert float(u.sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(z.sum()) == pytest.approx(0.0, abs=1e-9)


def test_a_type_mixing_own_effect_and_pooled_keys_has_no_zero_sum() -> None:
    # The pooled key identifies b_type, so a zero-sum on the others would pin
    # b_type to their mean. Over many draws the sum is not identically zero.
    import pymc as pm

    reports = [
        *_co_reporting_keys(4, n_battles=3),
        *_one_key(1, url="https://example/lonely", first=90),
    ]
    design = _design(reports)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    prior = pm.sample_prior_predictive(draws=20, model=model, random_seed=3).prior
    sums = np.asarray(prior["u_source"].values[0]).sum(axis=-1)
    z_sums = np.asarray(prior["z_sigma"].values[0]).sum(axis=-1)

    assert float(np.abs(sums).max()) > 1e-3
    assert float(np.abs(z_sums).max()) > 1e-3


def test_a_lone_own_effect_key_beside_pooled_keys_gets_a_free_deviation() -> None:
    # "a" co-reports in six battles, its two partners in three each, so at a
    # threshold of four only "a" qualifies; it is the sole own-effect key of
    # its type and still carries a deviation, since the pooled keys hold b_type.
    reports = [
        *_keys_on_battles(["https://example/a", "https://example/b"], range(0, 3)),
        *_keys_on_battles(["https://example/a", "https://example/c"], range(3, 6)),
    ]
    design = _design(reports, min_battles_per_source_effect=4)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    urls = [k.url for k in design.source_keys]
    assert design.n_source_keys_own_effect == 1
    u = np.asarray(_prior(model)["u_source"].values[0, 0])
    assert float(u[urls.index("https://example/a")]) != 0.0
    assert float(u[urls.index("https://example/b")]) == 0.0
    assert float(u[urls.index("https://example/c")]) == 0.0
