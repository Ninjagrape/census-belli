"""Parameters of the report stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["STORED_CI_LEVEL", "ReportParams"]

# general_skill_estimates stores the 95% interval only, so no other level can
# be reported without recomputing it, which this stage must not do.
STORED_CI_LEVEL: Final[float] = 0.95
MAX_FLOAT_PRECISION: Final[int] = 12


@dataclass(frozen=True)
class ReportParams:
    """The ``params`` block of ``agents/report.yaml``.

    Attributes:
        output_root: Directory the stage writes into.
        diagnostics_root: Root of the model stage's ``run{N}`` directories.
        evaluation_root: Root of the evaluate stage's ``run{N}`` directories.
        evaluate_spec_path: The evaluate spec, read for each gate's severity.
        report_path: Where the manifest is written (read by the spec's gates).
        require_passing_diagnostics: Whether a failed blocking model gate
            marks the outputs provisional.
        top_n: Rows of the headline table in the static report.
        ci_level: Credible level shown; must equal the stored 95%.
        min_battles: Fewest credited battles for a headline rank.
        show_rank_overlap: Whether adjacent-rank overlap is flagged.
        overlap_threshold: Overlap share of the narrower interval above which
            adjacent ranks are flagged.
        explorer_max_generals: Cap on generals in the JSON export and explorer.
        export_float_precision: Decimal places of floats in the CSV.
        raw: The mapping as given.
    """

    output_root: str
    diagnostics_root: str
    evaluation_root: str
    evaluate_spec_path: str
    report_path: str
    require_passing_diagnostics: bool
    top_n: int
    ci_level: float
    min_battles: int
    show_rank_overlap: bool
    overlap_threshold: float
    explorer_max_generals: int
    export_float_precision: int
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> ReportParams:
        """Read and validate the spec's ``params``.

        Args:
            params: ``agents/report.yaml``'s ``params``.

        Returns:
            The parameters, with the spec's defaults for absent keys.

        Raises:
            ValueError: If ``ci_level`` is not the stored 0.95, a count is below
                its minimum, the threshold is outside [0, 1], the run selection
                is not ``latest`` or embedding is switched off.
        """
        output_root = str(params.get("output_root", "data/processed/report"))
        out = cls(
            output_root=output_root,
            diagnostics_root=str(
                params.get("diagnostics_root", "data/processed/model_diagnostics")
            ),
            evaluation_root=str(params.get("evaluation_root", "data/processed/evaluation")),
            evaluate_spec_path=str(params.get("evaluate_spec_path", "agents/evaluate.yaml")),
            report_path=str(params.get("report_path", f"{output_root}/manifest.json")),
            require_passing_diagnostics=bool(params.get("require_passing_diagnostics", True)),
            top_n=int(params.get("top_n", 100)),
            ci_level=float(params.get("ci_level", STORED_CI_LEVEL)),
            min_battles=int(params.get("min_battles", 3)),
            show_rank_overlap=bool(params.get("show_rank_overlap", True)),
            overlap_threshold=float(params.get("overlap_threshold", 0.5)),
            explorer_max_generals=int(params.get("explorer_max_generals", 1000)),
            export_float_precision=int(params.get("export_float_precision", 4)),
            raw=dict(params),
        )
        out._validate(params)
        return out

    def _validate(self, params: Mapping[str, Any]) -> None:
        if str(params.get("run_selection", "latest")) != "latest":
            raise ValueError("run_selection must be 'latest'")
        if not params.get("explorer_embed_data", True):
            raise ValueError("explorer_embed_data must be true: the explorer is one offline file")
        if self.ci_level != STORED_CI_LEVEL:
            raise ValueError(
                f"ci_level must be {STORED_CI_LEVEL}, got {self.ci_level}: the stored "
                "skill interval is 95% and this stage does not recompute it"
            )
        for name, value in (
            ("top_n", self.top_n),
            ("min_battles", self.min_battles),
            ("explorer_max_generals", self.explorer_max_generals),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError(f"overlap_threshold must be in [0, 1], got {self.overlap_threshold}")
        if not 0 <= self.export_float_precision <= MAX_FLOAT_PRECISION:
            raise ValueError(
                f"export_float_precision must be in [0, {MAX_FLOAT_PRECISION}], "
                f"got {self.export_float_precision}"
            )
