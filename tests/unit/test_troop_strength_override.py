"""The ``troop_strength`` override kind: validation and how extract applies it."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.extractors import BattleExtraction, BattleFacts, MissingField, SideExtraction
from pipeline.extractors.override_apply import apply_extract_overrides
from pipeline.extractors.records import TroopReport
from pipeline.overrides import KINDS_BY_STAGE, OverrideError, OverrideReport, load_overrides
from pipeline.reconcilers.claims import classify_regime

CITATION = "Chandler, The Campaigns of Napoleon (1966), p. 412"
TARGET = {"battle": "borodino", "side": "French Empire"}


def _entry(n: int, value: Any, target: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": "troop_strength",
        "target": target or dict(TARGET),
        "value": value,
        "source": CITATION,
        "note": "",
        "reviewed": {"by": "lee", "on": "2026-10-23"},
    }


def _write(tmp_path: Path, *entries: dict[str, Any]) -> Path:
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "entries": list(entries)}), encoding="utf-8")
    return path


def _battle() -> BattleExtraction:
    return BattleExtraction(
        slug="borodino",
        name="Battle of Borodino",
        facts=BattleFacts(name="Battle of Borodino"),
        sides=[
            SideExtraction(
                label="French Empire",
                troop_reports=[
                    TroopReport(
                        side_label="French Empire",
                        reported_value=250_000.0,
                        provenance=_infobox(),
                    )
                ],
            ),
            SideExtraction(label="Russian Empire"),
        ],
        missing=[
            MissingField(field_name="troop_total", side_label="French Empire"),
            MissingField(field_name="troop_total", side_label="Russian Empire"),
        ],
    )


def _infobox() -> Any:
    from pipeline.extractors.records import Provenance

    return Provenance(source_type="wikipedia_infobox", extraction_method="infobox_parser")


def _apply(
    tmp_path: Path, battle: BattleExtraction, *entries: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    overrides = load_overrides(_write(tmp_path, *entries))
    report = OverrideReport("extract", overrides)
    apply_extract_overrides([battle], overrides.for_stage("extract"), report)
    report.write(tmp_path)
    payload = json.loads((tmp_path / "overrides_extract.json").read_text(encoding="utf-8"))
    return overrides.for_stage("extract"), payload


# ─── Validation ──────────────────────────────────────────────────────────────


def test_the_kind_belongs_to_the_extract_stage() -> None:
    assert "troop_strength" in KINDS_BY_STAGE["extract"]


@pytest.mark.parametrize(
    "value",
    [
        {"point": 50_000, "scope": "engaged"},
        {"low": 40_000, "high": 60_000, "scope": "available"},
        {"point": 1, "scope": "theatre_strength"},
        {"low": 1, "high": 2, "scope": "on_paper"},
    ],
)
def test_valid_values_load_unchanged(tmp_path: Path, value: dict[str, Any]) -> None:
    loaded = load_overrides(_write(tmp_path, _entry(1, value)))

    assert loaded.entries[0].value == value
    assert loaded.entries[0].stage == "extract"
    assert dict(loaded.entries[0].target) == TARGET


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"scope": "engaged"}, "point"),
        ({"point": 5_000, "low": 4_000, "high": 6_000, "scope": "engaged"}, "exactly one"),
        ({"low": 4_000, "scope": "engaged"}, "high"),
        ({"high": 6_000, "scope": "engaged"}, "low"),
        ({"point": 0, "scope": "engaged"}, "positive"),
        ({"point": -5, "scope": "engaged"}, "positive"),
        ({"point": 50_000.5, "scope": "engaged"}, "integer"),
        ({"point": "50000", "scope": "engaged"}, "integer"),
        ({"point": True, "scope": "engaged"}, "integer"),
        ({"low": 0, "high": 5, "scope": "engaged"}, "positive"),
        ({"low": 6_000, "high": 6_000, "scope": "engaged"}, "less than"),
        ({"low": 7_000, "high": 6_000, "scope": "engaged"}, "less than"),
        ({"point": 50_000}, "scope"),
        ({"point": 50_000, "scope": "deployed"}, "scope"),
        ({"point": 50_000, "scope": "population"}, "scope"),
        ({"point": 50_000, "scope": "unknown"}, "scope"),
        ({"point": 50_000, "scope": "engaged", "extra": 1}, "keys"),
        (50_000, "mapping"),
        ([50_000], "mapping"),
    ],
)
def test_invalid_values_are_rejected_naming_the_entry(
    tmp_path: Path, value: Any, match: str
) -> None:
    with pytest.raises(OverrideError, match=rf"ov-0001.*{match}"):
        load_overrides(_write(tmp_path, _entry(1, value)))


def test_the_target_needs_battle_and_side(tmp_path: Path) -> None:
    value = {"point": 50_000, "scope": "engaged"}

    with pytest.raises(OverrideError, match="target"):
        load_overrides(_write(tmp_path, _entry(1, value, {"battle": "borodino"})))


def test_two_entries_for_one_side_are_a_duplicate_target(tmp_path: Path) -> None:
    value = {"point": 50_000, "scope": "engaged"}
    second = _entry(2, value, {"battle": " Borodino ", "side": "FRENCH empire"})

    with pytest.raises(OverrideError, match="ov-0002.*duplicate target"):
        load_overrides(_write(tmp_path, _entry(1, value), second))


# ─── Application at extract ──────────────────────────────────────────────────


def test_a_point_entry_adds_one_modern_total_row_with_the_citation(tmp_path: Path) -> None:
    battle = _battle()

    _, report = _apply(tmp_path, battle, _entry(1, {"point": 50_000, "scope": "engaged"}))

    side = battle.sides[0]
    row = side.troop_reports[-1]
    assert len(side.troop_reports) == 2  # the infobox total stays; reconcile supersedes it
    assert row.reported_value == 50_000
    assert row.branch == "total"
    assert row.scope == "engaged"
    assert row.is_estimate is True
    assert not row.is_lower_bound and not row.is_upper_bound
    assert classify_regime(row.extracted_context, value=row.reported_value) == "modern_scholarly"
    assert row.provenance.source_type == "manual_entry"
    assert row.provenance.extraction_method == "manual"
    assert row.provenance.source_title == CITATION
    assert report["applied"]["troop_strength"] == 1 and report["unmatched"] == []
    assert "override ov-0001 applied (troop_strength)" in battle.notes


def test_a_range_entry_adds_a_lower_and_upper_row_sharing_one_context(tmp_path: Path) -> None:
    battle = _battle()

    _apply(tmp_path, battle, _entry(1, {"low": 40_000, "high": 60_000, "scope": "available"}))

    rows = battle.sides[0].troop_reports[1:]
    low, high = sorted(rows, key=lambda r: r.reported_value)
    assert (low.reported_value, high.reported_value) == (40_000, 60_000)
    assert low.is_lower_bound and not low.is_upper_bound
    assert high.is_upper_bound and not high.is_lower_bound
    assert low.extracted_context == high.extracted_context
    assert classify_regime(low.extracted_context) == "modern_scholarly"
    assert low.scope == high.scope == "available"
    assert low.provenance == high.provenance


def test_the_side_label_matches_after_strip_and_casefold(tmp_path: Path) -> None:
    battle = _battle()
    entry = _entry(
        1, {"point": 50_000, "scope": "engaged"}, {"battle": "Borodino ", "side": " french EMPIRE"}
    )

    _, report = _apply(tmp_path, battle, entry)

    assert report["unmatched"] == []
    assert len(battle.sides[0].troop_reports) == 2


def test_the_sides_missing_troop_total_gap_is_cleared_and_the_other_side_keeps_its_own(
    tmp_path: Path,
) -> None:
    battle = _battle()

    _apply(tmp_path, battle, _entry(1, {"point": 50_000, "scope": "engaged"}))

    assert [(m.field_name, m.side_label) for m in battle.missing] == [
        ("troop_total", "Russian Empire")
    ]


def test_a_missing_side_or_battle_is_reported_unmatched_and_changes_nothing(
    tmp_path: Path,
) -> None:
    battle = _battle()
    value = {"point": 50_000, "scope": "engaged"}

    _, report = _apply(
        tmp_path,
        battle,
        _entry(1, value, {"battle": "borodino", "side": "Prussia"}),
        _entry(2, value, {"battle": "nowhere", "side": "French Empire"}),
    )

    assert {u["id"]: u["reason"] for u in report["unmatched"]} == {
        "ov-0001": "side 'Prussia' not found in battle (sides: ['French Empire', 'Russian Empire'])",
        "ov-0002": "battle not found",
    }
    assert report["applied"]["troop_strength"] == 0
    assert len(battle.sides[0].troop_reports) == 1
    assert len(battle.missing) == 2


def test_the_context_carries_the_figure_so_distinct_entries_stay_distinct(tmp_path: Path) -> None:
    first, second = _battle(), _battle()
    _apply(tmp_path, first, _entry(1, {"point": 50_000, "scope": "engaged"}))
    _apply(tmp_path, second, _entry(1, {"point": 55_000, "scope": "engaged"}))

    contexts = {
        first.sides[0].troop_reports[-1].extracted_context,
        second.sides[0].troop_reports[-1].extracted_context,
    }
    assert len(contexts) == 2
    assert math.isfinite(first.sides[0].troop_reports[-1].reported_value)
