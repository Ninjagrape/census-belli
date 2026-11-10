"""The data root is one setting, read lazily, and every default follows it."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import paths
from pipeline.paths import DATA_ROOT_ENV, data_path, data_root


@pytest.fixture
def alt_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the data root at a second corpus that does not exist on disk."""
    root = tmp_path / "corpus3"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    return root


@pytest.fixture
def default_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure the variable is unset."""
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)


def test_default_root_is_data(default_root: None) -> None:
    assert data_root() == Path("data")
    assert data_path("raw", "x") == Path("data/raw/x")


def test_empty_variable_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "  ")
    assert data_root() == Path("data")


def test_env_override(alt_root: Path) -> None:
    assert data_root() == alt_root
    assert data_path("processed") == alt_root / "processed"


def test_absolute_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert data_path("processed") == tmp_path / "processed"


def test_evaluation_report_resolves_under_env_root(alt_root: Path) -> None:
    from pipeline.quality import default_evaluation_report

    report = default_evaluation_report({"report_under_root": "processed/report/manifest.json"})
    assert report == alt_root / "processed" / "report" / "manifest.json"


def test_evaluation_report_rejects_parent_segment(alt_root: Path) -> None:
    from pipeline.quality import default_evaluation_report

    with pytest.raises(ValueError, match="'..'"):
        default_evaluation_report({"report_under_root": "../x"})


def test_evaluation_report_rejects_absolute_path(alt_root: Path, tmp_path: Path) -> None:
    from pipeline.quality import default_evaluation_report

    with pytest.raises(ValueError, match="relative"):
        default_evaluation_report({"report_under_root": str(tmp_path / "manifest.json")})


def test_root_is_read_at_call_time(default_root: None, monkeypatch: pytest.MonkeyPatch) -> None:
    assert paths.data_root() == Path("data")
    monkeypatch.setenv(DATA_ROOT_ENV, "elsewhere")
    assert paths.data_root() == Path("elsewhere")


def test_offline_dirs_follow_root(alt_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.llm import offline

    # conftest pins REQUEST_DIR for isolation; clear it to see the default.
    monkeypatch.setattr(offline, "REQUEST_DIR", None)
    assert offline.default_request_dir() == alt_root / "llm_requests"
    assert offline.default_response_dir() == alt_root / "llm_responses"


def test_offline_override_still_wins(alt_root: Path) -> None:
    from pipeline.llm import offline

    # conftest's autouse fixture pins REQUEST_DIR to a temp directory.
    assert offline.default_request_dir() != alt_root / "llm_requests"


def test_crawl_state_path_follows_root(alt_root: Path) -> None:
    from pipeline.crawlers.state import default_state_path

    assert default_state_path() == alt_root / "raw" / "crawl_state.json"


def test_crawl_paths_default_follows_root(alt_root: Path) -> None:
    from pipeline.stages.crawl import CrawlPaths

    assert CrawlPaths.under(None).state_file == alt_root / "raw" / "crawl_state.json"
    assert CrawlPaths.under(None).battles_html == alt_root / "raw" / "battles_html"


def test_impute_params_follow_root(alt_root: Path) -> None:
    from pipeline.imputers.priors import ImputeParams

    params = ImputeParams.from_params({})
    assert Path(params.imputed_root) == alt_root / "imputed"
    assert Path(params.reconcile_root) == alt_root / "processed" / "reconcile"


def test_impute_explicit_param_wins(alt_root: Path) -> None:
    from pipeline.imputers.priors import ImputeParams

    params = ImputeParams.from_params({"imputed_root": "somewhere/else"})
    assert params.imputed_root == "somewhere/else"


def test_evaluate_params_follow_root(alt_root: Path) -> None:
    from pipeline.evaluators.config import EvaluateParams

    params = EvaluateParams.from_mapping({})
    assert Path(params.report_path) == alt_root / "processed" / "evaluation_report.json"
    assert Path(params.output_root) == alt_root / "processed" / "evaluation"
    assert Path(params.reconcile_root) == alt_root / "processed" / "reconcile"


def test_evaluate_explicit_param_wins(alt_root: Path) -> None:
    from pipeline.evaluators.config import EvaluateParams

    params = EvaluateParams.from_mapping({"report_path": "x/report.json"})
    assert params.report_path == "x/report.json"


def test_report_params_follow_root(alt_root: Path) -> None:
    from pipeline.reporters.config import ReportParams

    params = ReportParams.from_mapping({})
    assert Path(params.output_root) == alt_root / "processed" / "report"
    assert Path(params.report_path) == alt_root / "processed" / "report" / "manifest.json"
    assert Path(params.diagnostics_root) == alt_root / "processed" / "model_diagnostics"


def test_model_params_follow_root(alt_root: Path) -> None:
    from pipeline.modelers.config import ModelParams

    params = ModelParams.from_mapping({})
    assert Path(params.imputed_root) == alt_root / "imputed"
    assert Path(params.diagnostics_root) == alt_root / "processed" / "model_diagnostics"


def test_overrides_gate_default_root_follows_env(alt_root: Path) -> None:
    from pipeline.quality import _check_overrides_gate  # noqa: PLC2701

    result = _check_overrides_gate("g", {"params": {"stage": "extract"}}, "error", None)
    assert str(alt_root / "processed" / "overrides_extract.json") in result.message


def test_overrides_gate_explicit_root_wins(alt_root: Path, tmp_path: Path) -> None:
    from pipeline.quality import _check_overrides_gate  # noqa: PLC2701

    check = {"params": {"stage": "extract", "root": str(tmp_path)}}
    result = _check_overrides_gate("g", check, "error", None)
    assert str(tmp_path / "overrides_extract.json") in result.message


def test_imputation_gate_default_root_follows_env(alt_root: Path) -> None:
    from pipeline.quality import _check_imputation_gate  # noqa: PLC2701

    result = _check_imputation_gate("g", {}, "error", None)
    assert str(alt_root / "imputed" / "neutral" / "gate_report.json") in result.message


def test_stage_defaults_follow_root(alt_root: Path) -> None:
    from pipeline.stages import classify, extract, reconcile, resolve

    for module in (classify, extract, reconcile, resolve):
        assert module.default_processed_root() == alt_root / "processed"
    for module in (classify, extract, resolve):
        assert module.default_raw_root() == alt_root / "raw"
