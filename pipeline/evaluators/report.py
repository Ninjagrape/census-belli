"""Gates, the report dictionary and the files the evaluate stage writes."""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from pipeline.evaluators.compare import min_required_overlap
from pipeline.evaluators.config import EvaluateParams
from pipeline.evaluators.cv import CVResult, FoldRecord
from pipeline.evaluators.sensitivity import STATUS_NO_OP, STATUS_OK, SensitivityResult
from pipeline.evaluators.target import PublishedRun
from pipeline.modelers.ordinal import N_LEVELS
from pipeline.modelers.outputs import write_columns_csv, write_json

__all__ = [
    "ACCURACY_FLOOR",
    "BRIER_CEILING",
    "REPORT_NAME",
    "build_gates",
    "build_report",
    "write_outputs",
]

SCHEMA_VERSION: Final[int] = 1
REPORT_NAME: Final[str] = "evaluation_report.json"
ACCURACY_FLOOR: Final[float] = 0.55
BRIER_CEILING: Final[float] = 0.25

DISCLOSURES: Final[tuple[str, ...]] = (
    "Held-out battles' outcomes are masked out of each fold's likelihood and prediction "
    "uses the measurement-only x_diff prior, so a held-out battle's imputed force ratio "
    "never reaches its own prediction.",
    "Residual leakage remains: the impute stage's hyperparameters were fitted on every "
    "battle's outcome, and the training rounds' x_diff values are outcome-conditioned for "
    "the training battles, so the fitted force coefficient is not free of outcome "
    "information.",
    "Sensitivity variants and the baseline refit use the same rounds and seeds; "
    "post-hoc variants re-reduce the baseline draws without refitting.",
    "The +/-1 sd g_unlabelled sensitivity is not run here (handover-log 21.7).",
    "Cross-validation has two arms. The imputed arm trains on imputed x_diff (outcome-"
    "conditioned, every side redrawn) and predicts from the measurement-only prior, so it "
    "mixes model quality with a circularity penalty; it is the arm the published ranking "
    "resembles and the one the gates use. The joint arm trains and predicts inside the joint "
    "latent-troops model and is reported alongside it, not gated.",
    "equal_attribution also converts unknown and default_split sides into named ones, so it "
    "changes the latent-side set as well as the weights.",
    "weak_hyperpriors keeps skills hierarchically shrunk: it widens the prior on the skill sd "
    "and the force slope, not the pooling structure.",
    "Cross-validation fold fits compute WAR over prior-only generals (those seen only in "
    "held-out battles), so fold WAR is never reused for rankings.",
)


def _orientation_disclosure(binary: Mapping[str, Any]) -> str:
    """Disclosure that listing order predicts the winner and the model ignores it."""
    rate = _finite(binary.get("side_a_win_rate"))
    shown = (
        f"side A won {rate:.0%} of decided battles in the imputed arm"
        if rate is not None
        else ("side A's win rate is unavailable (no decided held-out battles)")
    )
    return (
        "Listing order predicts the winner in this corpus (side A, the lower side_id and "
        f"usually the first-listed combatant: {shown}). The model ignores orientation by "
        "design, so accuracy is judged against 0.5 and calibration in the favourite's "
        "frame (bins over max(p, 1 - p)), not P(A wins)."
    )


def _gate(name: str, passed: bool, actual: Any, threshold: str, message: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "message": message,
    }


def _finite(value: Any) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _metric_gate(
    name: str, value: float | None, n: int, *, above: bool, bound: float, label: str
) -> dict[str, Any]:
    threshold = f"{'>' if above else '<'} {bound}"
    if value is None or n == 0:
        return _gate(name, False, None, threshold, f"no held-out binary predictions for {label}")
    passed = value > bound if above else value < bound
    verdict = "meets" if passed else "misses"
    return _gate(
        name, passed, value, threshold, f"{label} {value:.3f} {verdict} {threshold} (n={n})"
    )


def _cv_converged_gate(folds: Sequence[FoldRecord]) -> dict[str, Any]:
    bad = [f.fold for f in folds if not f.converged]
    actual = {"converged": len(folds) - len(bad), "folds": len(folds)}
    message = (
        f"all {len(folds)} folds met the write thresholds"
        if not bad
        else f"fold(s) {bad} did not converge; their battles have no predictions"
    )
    return _gate("cv_converged", not bad, actual, "every fold converged", message)


def _stability_gate(sens: SensitivityResult, params: EvaluateParams) -> dict[str, Any]:
    threshold = (
        f"top-{params.top_n} overlap >= {params.min_top_overlap} "
        "(scaled below that many ranked generals)"
    )
    if sens.baseline_table is None:
        reason = sens.baseline_error or "baseline refit unavailable"
        return _gate("ranking_stability", False, None, threshold, reason)
    actual: dict[str, Any] = {}
    failing: list[str] = []
    for v in sens.variants:
        if v.status == STATUS_NO_OP:
            continue
        if v.status != STATUS_OK or v.vs_baseline is None:
            actual[v.spec.name] = v.status
            failing.append(f"{v.spec.name} ({v.status})")
            continue
        cmp = v.vs_baseline["score"]
        required = min_required_overlap(
            cmp["top_n_effective"], params.top_n, params.min_top_overlap
        )
        actual[v.spec.name] = {"top_overlap": cmp["top_overlap"], "required": required}
        if cmp["top_overlap"] < required:
            failing.append(f"{v.spec.name} (overlap {cmp['top_overlap']} < {required})")
    message = "every refitted variant keeps the top set" if not failing else "; ".join(failing)
    floor = _noise_floor_overlap(sens.noise_floor)
    if floor is not None:
        actual["noise_floor"] = {"top_overlap": floor}
        message += f"; noise floor top-{params.top_n} overlap {floor}"
    elif sens.noise_floor is not None:
        actual["noise_floor"] = sens.noise_floor.get("status")
        message += f"; noise floor {sens.noise_floor.get('status')}"
    return _gate("ranking_stability", not failing, actual, threshold, message)


def _noise_floor_overlap(noise_floor: Mapping[str, Any] | None) -> int | None:
    """Top-N overlap of the noise-floor refit with the baseline, if it ran."""
    if noise_floor is None or noise_floor.get("status") != STATUS_OK:
        return None
    return int(noise_floor["score"]["top_overlap"])


def build_gates(
    cv: CVResult,
    sens: SensitivityResult,
    named_gate: Mapping[str, Any],
    params: EvaluateParams,
) -> list[dict[str, Any]]:
    """The five evaluation gates, in spec order.

    Args:
        cv: Cross-validation result.
        sens: Sensitivity result.
        named_gate: The ``named_checks`` gate from the expert comparison.
        params: Evaluate parameters.

    Returns:
        Gate dicts with ``name``, ``passed``, ``actual``, ``threshold``, ``message``.
    """
    binary = cv.metrics["binary"]
    n = int(binary["n"])
    return [
        _metric_gate(
            "held_out_accuracy",
            _finite(binary["accuracy"]),
            n,
            above=True,
            bound=ACCURACY_FLOOR,
            label="accuracy",
        ),
        _metric_gate(
            "calibration",
            _finite(binary["brier"]),
            n,
            above=False,
            bound=BRIER_CEILING,
            label="Brier score",
        ),
        _cv_converged_gate(cv.folds),
        _stability_gate(sens, params),
        dict(named_gate),
    ]


def _arm_section(
    arm_folds: Sequence[FoldRecord], metrics: Mapping[str, Any], baselines: Any
) -> dict[str, Any]:
    return {
        **metrics,
        "baselines": baselines,
        "folds": [
            {
                "fold": f.fold,
                "n_held_out": f.n_held_out,
                "converged": f.converged,
                "worst": f.worst,
                "error": f.error,
                "seconds": f.seconds,
            }
            for f in arm_folds
        ],
    }


def _cv_section(cv: CVResult) -> dict[str, Any]:
    return {
        "imputed": _arm_section(cv.folds, cv.metrics, cv.baselines),
        "joint": _arm_section(cv.joint.folds, cv.joint.metrics, cv.joint.baselines),
    }


def build_report(
    *,
    published: PublishedRun,
    params: EvaluateParams,
    settings: Mapping[str, Any],
    counts: Mapping[str, Any],
    created_at: str,
    cv: CVResult,
    sens: SensitivityResult,
    named_checks: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    ppc: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    figures: Sequence[str],
    timings: Mapping[str, float],
) -> dict[str, Any]:
    """Assemble the report dictionary (not yet JSON-safe; see ``write_json``).

    Args:
        published: The evaluated run.
        params: Evaluate parameters.
        settings: Model and sampler settings used by the fits.
        counts: Design and input counts.
        created_at: ISO timestamp.
        cv: Cross-validation result.
        sens: Sensitivity result.
        named_checks: Named-check results.
        references: External list comparisons.
        ppc: Posterior predictive section.
        gates: Output of :func:`build_gates`.
        figures: Figure paths written.
        timings: Seconds per section.

    Returns:
        The report.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "model_run_id": published.run_id,
        "imputation_run_id": published.imputation_run_id,
        "created_at": created_at,
        "counts": dict(counts),
        "settings": {
            "evaluate": {**params.raw, "eval_min_ess_bulk": params.eval_min_ess_bulk},
            **settings,
        },
        "disclosures": [*DISCLOSURES, _orientation_disclosure(cv.metrics["binary"])],
        "cv": _cv_section(cv),
        "baselines": cv.baselines,
        "sensitivity": sens.summary(),
        "named_checks": list(named_checks),
        "references": list(references),
        "ppc": dict(ppc),
        "figures": list(figures),
        "gates": list(gates),
        "timings_seconds": dict(timings),
    }


def _cv_columns(
    cv: CVResult, battle_ids: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray
) -> dict[str, Any]:
    keep = cv.predicted
    probs = cv.probs[keep]
    win = probs[:, 3] + probs[:, 4]
    decided = win + probs[:, 0] + probs[:, 1]
    p_win = np.divide(win, decided, out=np.full(win.shape, 0.5), where=decided > 0)
    columns: dict[str, Any] = {"battle_id": battle_ids[keep], "fold": cv.fold_of[keep]}
    for level in range(N_LEVELS):
        columns[f"p_level{level}"] = probs[:, level]
    columns.update(k_lo=k_lo[keep], k_hi=k_hi[keep], p_a_win_binary=p_win)
    return columns


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def write_outputs(
    directory: Path,
    report_path: Path,
    report: Mapping[str, Any],
    *,
    cv: CVResult,
    battle_ids: np.ndarray,
    k_lo: np.ndarray,
    k_hi: np.ndarray,
    sens: SensitivityResult,
    published: PublishedRun,
) -> list[Path]:
    """Write the report, the CV predictions and the sensitivity rankings.

    Args:
        directory: ``{output_root}/run{N}``.
        report_path: Where the latest report is copied.
        report: The report dictionary.
        cv: Cross-validation result.
        battle_ids: Raw battle id per design battle.
        k_lo: Lowest compatible outcome level per battle.
        k_hi: Highest compatible outcome level per battle.
        sens: Sensitivity result.
        published: The evaluated run, for names and Q-ids.

    Returns:
        The paths written, the copy at ``report_path`` last.
    """
    written = [
        write_columns_csv(directory / "cv_predictions.csv", _cv_columns(cv, battle_ids, k_lo, k_hi))
    ]
    names = dict(zip(published.general_ids.tolist(), published.names, strict=True))
    qids = dict(zip(published.general_ids.tolist(), published.wikidata_ids, strict=True))
    tables = {"baseline": sens.baseline_table} | {
        v.spec.name: v.table for v in [*sens.variants, sens.imputed_subset] if v and v.table
    }
    for name, table in tables.items():
        if table is None:
            continue
        ids = table.ids.tolist()
        written.append(
            write_columns_csv(
                directory / "sensitivity" / f"{_safe_name(name)}.csv",
                {
                    "general_id": ids,
                    "name": [names.get(g, "") for g in ids],
                    "wikidata_id": [qids.get(g) or "" for g in ids],
                    "score": table.score,
                    "rank": table.rank,
                    "skill_mean": table.skill_mean,
                },
            )
        )
    main = write_json(directory / REPORT_NAME, report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(main, report_path)
    return [*written, main, report_path]
