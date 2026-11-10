"""SamplerSettings parsing and the kwargs handed to pm.sample (no sampling)."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.reconcilers.model import SamplerSettings, build_sample_kwargs


def test_low_rank_defaults_off() -> None:
    assert SamplerSettings().low_rank_mass_matrix is False
    assert SamplerSettings.from_params({}).low_rank_mass_matrix is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("1", True),
        (False, False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("0", False),
    ],
)
def test_low_rank_parses_real_bool(raw: Any, expected: bool) -> None:
    got = SamplerSettings.from_params({"mcmc_low_rank_mass_matrix": raw})
    assert got.low_rank_mass_matrix is expected


def test_low_rank_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="mcmc_low_rank_mass_matrix"):
        SamplerSettings.from_params({"mcmc_low_rank_mass_matrix": "maybe"})


def test_cli_override_form_parses_true() -> None:
    from pipeline.config import parse_set_overrides  # noqa: PLC0415

    params = parse_set_overrides(["mcmc_low_rank_mass_matrix=true"])
    assert SamplerSettings.from_params(params).low_rank_mass_matrix is True


def test_kwargs_unchanged_when_off() -> None:
    kwargs = build_sample_kwargs(SamplerSettings())
    assert "nuts_sampler_kwargs" not in kwargs
    assert kwargs["nuts_sampler"] == "nutpie"
    assert kwargs["target_accept"] == 0.9


def test_kwargs_carry_low_rank_with_nutpie() -> None:
    kwargs = build_sample_kwargs(SamplerSettings(low_rank_mass_matrix=True))
    assert kwargs["nuts_sampler_kwargs"] == {"adaptation": "low_rank"}
    assert kwargs["target_accept"] == 0.9


def test_low_rank_ignored_for_other_samplers() -> None:
    kwargs = build_sample_kwargs(SamplerSettings(low_rank_mass_matrix=True, nuts_sampler="pymc"))
    assert "nuts_sampler_kwargs" not in kwargs
    assert "nuts_sampler" not in kwargs
