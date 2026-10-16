"""Flat and nested exports of the ranking: the data behind every other output."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pipeline.reporters.atomic import write_text_atomic
from pipeline.reporters.disclosures import interval_method_note
from pipeline.reporters.records import BattleDetail
from pipeline.reporters.tables import RankedGeneral
from pipeline.reporters.view import ReportView

__all__ = [
    "CSV_COLUMNS",
    "SCHEMA_VERSION",
    "build_payload",
    "read_csv_rows",
    "write_rankings_csv",
    "write_rankings_json",
]

SCHEMA_VERSION: Final[int] = 2

# CSV text columns a spreadsheet could read as a formula; numbers are never prefixed.
TEXT_COLUMNS: Final[frozenset[str]] = frozenset({"name", "wikidata_id"})
FORMULA_PREFIXES: Final[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")

CSV_COLUMNS: Final[tuple[str, ...]] = (
    "rank",
    "general_id",
    "name",
    "wikidata_id",
    "headline",
    "provisional",
    "n_battles",
    "effective_battles",
    "n_uncredited_battles",
    "first_year",
    "last_year",
    "total_war",
    "war_per_battle",
    "war_sd",
    "war_ci_lo",
    "war_ci_hi",
    "war_interval_method",
    "p_above_next_rank",
    "overlaps_next_rank",
    "skill_mean",
    "skill_median",
    "skill_sd",
    "skill_ci95_lo",
    "skill_ci95_hi",
    "skill_ci90_lo",
    "skill_ci90_hi",
    "component_id",
)

LABELS: Final[dict[str, str]] = {
    "total_war": "Expected WAR (total)",
    "war_per_battle": "Expected WAR per battle",
    "war_ci": (
        "95% credible interval on total WAR; war_interval_method says whether it is the stored "
        "posterior interval or, for older runs, a normal approximation (total WAR plus or "
        "minus z times war_sd)"
    ),
    "p_above_next_rank": (
        "Share of posterior draws in which WAR exceeds that of the next-ranked general"
    ),
    "expected_score_model": "Expected score (model)",
    "expected_score_replacement": "Expected score (replacement)",
    "actual_outcome": "Actual outcome (1 win, 0.5 draw, 0 loss)",
}


def _scalars(g: RankedGeneral, provisional: bool) -> dict[str, Any]:
    """The per-general fields shared by the CSV and the JSON, full precision."""
    e, r = g.estimate, g.rubin
    return {
        "rank": g.rank,
        "general_id": e.general_id,
        "name": e.name,
        "wikidata_id": e.wikidata_id,
        "headline": g.is_headline,
        "provisional": provisional,
        "n_battles": e.n_battles,
        "effective_battles": r.effective_battles,
        "n_uncredited_battles": r.n_uncredited_battles,
        "first_year": r.first_year,
        "last_year": r.last_year,
        "total_war": e.total_war,
        "war_per_battle": e.war_per_battle,
        "war_sd": e.war_sd,
        "war_ci_lo": g.war_lo,
        "war_ci_hi": g.war_hi,
        "war_interval_method": (
            None
            if g.war_lo is None
            else ("normal_approx" if g.war_interval_approx else "posterior")
        ),
        "p_above_next_rank": g.p_above_next,
        "overlaps_next_rank": g.overlaps_next,
        "skill_mean": e.skill_mean,
        "skill_median": e.skill_median,
        "skill_sd": e.skill_sd,
        "skill_ci95_lo": e.skill_ci_lo,
        "skill_ci95_hi": e.skill_ci_hi,
        "skill_ci90_lo": e.skill_ci90_lo,
        "skill_ci90_hi": e.skill_ci90_hi,
        "component_id": r.component_id,
    }


def _cell(value: Any, precision: int, *, text: bool = False) -> str:
    if value is None:
        return ""
    if text:
        plain = str(value)
        return f"'{plain}" if plain.startswith(FORMULA_PREFIXES) else plain
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, precision))
    return str(value)


def write_rankings_csv(path: Path, view: ReportView) -> int:
    """Write every headline and under-determined general, one row each.

    Under-determined rows have an empty ``rank``. Floats are rounded to
    ``export_float_precision``; the JSON keeps them whole. A text cell that
    starts with ``=``, ``+``, ``-`` or ``@`` gets a leading single quote so a
    spreadsheet does not run it as a formula.

    Args:
        path: Destination file.
        view: The report view.

    Returns:
        The number of data rows written.
    """
    rows = (*view.headline, *view.under_determined)
    precision = view.params.export_float_precision
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for g in rows:
        fields = _scalars(g, view.provisional)
        writer.writerow([_cell(fields[c], precision, text=c in TEXT_COLUMNS) for c in CSV_COLUMNS])
    write_text_atomic(path, buffer.getvalue())
    return len(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read an exported CSV back, for the stage's own consistency gate.

    Args:
        path: The CSV.

    Returns:
        One dict per data row.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _battle(b: BattleDetail) -> dict[str, Any]:
    return {
        "battle_id": b.battle_id,
        "battle": b.battle_name,
        "year": b.year,
        "polity": b.polity,
        "side_outcome": b.side_outcome,
        "command_role": b.command_role,
        "attribution_weight": b.attribution_weight,
        "expected_war": b.expected_war,
        "expected_score_model": b.expected_score_model,
        "expected_score_replacement": b.expected_score_replacement,
        "actual_outcome": b.actual_outcome,
    }


def _sensitivity(view: ReportView, ids: set[int]) -> dict[str, Any] | None:
    ev = view.evaluation
    if ev is None:
        return None
    return {
        "variants": [
            {
                "name": v.name,
                "status": v.status,
                "message": v.message,
                "spearman_rho": v.spearman_rho,
                "top_overlap": v.top_overlap,
            }
            for v in ev.variants
        ],
        "ranks": {
            name: {str(g): r for g, r in table.items() if g in ids}
            for name, table in ev.sensitivity_ranks.items()
        },
    }


def build_payload(view: ReportView, *, limit: int | None = None) -> dict[str, Any]:
    """The nested export: generals with per-battle detail, plus context.

    Each general carries every one of its battles, so the contributions sum to
    the stored total. ``rankings.json`` holds every general; only the explorer
    page passes a ``limit``.

    Args:
        view: The report view.
        limit: Cap on generals (headline first), or None for all of them.

    Returns:
        A JSON-serialisable dictionary.
    """
    everyone = (*view.headline, *view.under_determined)
    chosen = everyone if limit is None else everyone[:limit]
    ids = {g.estimate.general_id for g in chosen}
    generals = [
        {
            **_scalars(g, view.provisional),
            "battles": [_battle(b) for b in view.battles.get(g.estimate.general_id, ())],
        }
        for g in chosen
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_run_id": view.model_run_id,
        "evaluation_run_id": view.evaluation.model_run_id if view.evaluation else None,
        "provisional": view.provisional,
        "provisional_reasons": list(view.reasons),
        "ci_level": view.params.ci_level,
        "min_battles": view.params.min_battles,
        "top_n": view.params.top_n,
        "z": view.z,
        "overlap_threshold": view.params.overlap_threshold,
        "overlap_method": interval_method_note(uses_normal_approx=view.uses_normal_approx),
        "labels": LABELS,
        "disclosures": list(view.disclosures),
        "n_headline": len(view.headline),
        "n_under_determined": len(view.under_determined),
        "n_generals_total": len(everyone),
        "n_generals_shown": len(chosen),
        "generals": generals,
        "sensitivity": _sensitivity(view, ids),
        "evaluation_unavailable": view.evaluation_reason or None,
    }


def write_rankings_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write the nested export.

    Args:
        path: Destination file.
        payload: Output of :func:`build_payload`.

    Returns:
        The path.
    """
    return write_text_atomic(
        path, json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False)
    )
