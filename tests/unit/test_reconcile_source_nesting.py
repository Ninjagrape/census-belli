"""Source-effect nesting: which source keys get their own bias and noise terms.

On the real corpus every source key is one battle's Wikipedia infobox, so a
per-key bias is the battle level under another name. It absorbs the side
spread, and the published estimate (which leaves the source term out) goes
flat. A key therefore gets its own effect only when it spans enough battles
for the effect to be distinguishable from the battle it sits in.
"""

from __future__ import annotations

import pytest

from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")

from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model  # noqa: E402


def _report(
    report_id: int,
    *,
    side_id: int,
    battle_id: int,
    url: str,
    source_type: str = "wikipedia_infobox",
) -> Report:
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


def _one_key(
    n_battles: int, *, url: str = "https://example/shared", first: int = 0
) -> list[Report]:
    """One source key reporting both sides of ``n_battles`` battles."""
    reports = []
    for battle in range(first, first + n_battles):
        for side in (1, 2):
            reports.append(
                _report(
                    100 * battle + side,
                    side_id=100 * battle + side,
                    battle_id=battle,
                    url=url,
                )
            )
    return reports


def _many_keys(n_keys: int, n_battles_each: int) -> list[Report]:
    reports: list[Report] = []
    for key in range(n_keys):
        reports += _one_key(
            n_battles_each, url=f"https://example/{key}", first=key * n_battles_each
        )
    return reports


def _prior(model: object) -> object:
    import pymc as pm

    return pm.sample_prior_predictive(draws=1, model=model, random_seed=1).prior


def _design(reports: list[Report], **kwargs: int) -> Design:
    return build_design(reports, quantity="troops", ancient_cutoff_year=500, **kwargs)


# ─── The flag in the design ──────────────────────────────────────────────────


def test_a_source_key_confined_to_one_battle_is_pooled() -> None:
    design = _design(_one_key(1))

    assert design.source_has_own_effect.tolist() == [False]
    assert design.n_source_keys_own_effect == 0
    assert design.n_source_keys_pooled == 1


def test_a_source_key_spanning_enough_battles_gets_its_own_effect() -> None:
    design = _design(_one_key(3))

    assert design.source_has_own_effect.tolist() == [True]
    assert design.n_source_keys_own_effect == 1
    assert design.n_source_keys_pooled == 0


def test_the_battle_threshold_is_a_parameter() -> None:
    reports = _one_key(2)

    assert _design(reports).source_has_own_effect.tolist() == [False]
    assert _design(reports, min_battles_per_source_effect=2).source_has_own_effect.tolist() == [
        True
    ]


def test_both_sides_of_one_battle_count_as_one_battle() -> None:
    # 132 of the corpus's 148 keys cover both sides of a single battle; counting
    # sides instead of battles would qualify them all.
    design = _design(_one_key(2))

    assert design.n_source_keys_own_effect == 0


def test_the_flag_is_aligned_with_source_keys() -> None:
    wide = _one_key(4, url="https://example/wide")
    narrow = _one_key(1, url="https://example/narrow", first=50)
    design = _design([*wide, *narrow])

    flags = dict(
        zip(
            (k.url for k in design.source_keys),
            design.source_has_own_effect.tolist(),
            strict=True,
        )
    )
    assert flags == {"https://example/wide": True, "https://example/narrow": False}
    assert design.n_source_keys_own_effect + design.n_source_keys_pooled == design.n_sources


# ─── What the model builds from it ───────────────────────────────────────────


def test_no_source_effect_is_created_when_no_key_qualifies() -> None:
    # A zero-size or unidentified u_source would sample its prior and report
    # clean convergence on nothing.
    design = _design(_many_keys(4, 1))
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    assert design.n_source_keys_own_effect == 0
    for name in ("u_source", "tau_u", "tau_sigma", "z_sigma"):
        assert name not in model.named_vars
    assert "beta_source" in model.named_vars
    assert "sigma_source" in model.named_vars


def test_source_effects_are_created_for_keys_spanning_enough_battles() -> None:
    design = _design(_many_keys(4, 3))
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    assert design.n_source_keys_own_effect == 4
    for name in ("u_source", "tau_u", "tau_sigma"):
        assert name in model.named_vars


def test_a_pooled_key_carries_the_type_effect_and_type_noise_only() -> None:
    reports = [
        *_many_keys(3, 3),
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
        *_many_keys(3, 3),
        *_one_key(1, url="https://example/lonely", first=90),
        *_one_key(1, url="https://example/lonely-too", first=95),
    ]
    design = _design(reports)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    urls = [k.url for k in design.source_keys]
    a, b = urls.index("https://example/lonely"), urls.index("https://example/lonely-too")
    sigma = _prior(model)["sigma_source"].values[0, 0]

    assert float(sigma[a]) == pytest.approx(float(sigma[b]))
