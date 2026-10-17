"""Classify manual overrides: matching, validation, reporting and request suppression."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from pipeline.classifiers import ClassifyCounts, CommanderRow, SideGroup
from pipeline.classifiers.overrides import (
    ManualBattleType,
    ManualSide,
    match_attribution,
    match_battle_types,
    overridden_side_ids,
)
from pipeline.overrides import OverrideEntry, OverrideReport, OverrideSet
from pipeline.stages import classify as classify_stage

SLUGS: dict[int, str | None] = {1: "battle_of_actium", 2: None}


def _entry(
    kind: str, target: dict[str, str], value: Any, entry_id: str = "ov-0001"
) -> OverrideEntry:
    return OverrideEntry(
        id=entry_id,
        kind=kind,
        stage="classify",
        target=MappingProxyType(target),
        value=value,
        source="Plutarch, Life of Antony",
        note="",
        reviewed_by="lee",
        reviewed_on="2026-10-17",
    )


def _item(
    general: str, role: str, rank: int, weight: float, reports_to: str | None = None
) -> dict[str, Any]:
    return {
        "general": general,
        "role": role,
        "rank": rank,
        "weight": weight,
        "reports_to": reports_to,
    }


def _side(side_id: int = 10, battle_id: int = 1, label: str = "Octavian") -> SideGroup:
    return SideGroup(
        battle_id=battle_id,
        side_id=side_id,
        side_label=label,
        battle_name="Battle of Actium",
        year_astronomical=-30,
        commanders=(
            CommanderRow(
                bc_id=100,
                battle_id=battle_id,
                side_id=side_id,
                general_id=1,
                name="Marcus Agrippa",
                wikidata_id="Q211101",
                attribution_method="default_split",
            ),
            CommanderRow(
                bc_id=101,
                battle_id=battle_id,
                side_id=side_id,
                general_id=2,
                name="Augustus",
                wikidata_id="Q1405",
                role_evidence="listed in infobox",
            ),
        ),
    )


def _attribution(value: list[dict[str, Any]], side: str = " octavian ") -> OverrideEntry:
    return _entry("attribution", {"battle": "Battle_Of_Actium", "side": side}, value)


GOOD_VALUE = [
    _item("q211101", "field_commander", 0, 0.85),
    _item("augustus", "sovereign", 1, 0.15, reports_to="Q211101"),
]


def test_attribution_applies_and_builds_manual_decisions() -> None:
    match = match_attribution(_attribution(GOOD_VALUE), [_side()], SLUGS)

    assert isinstance(match, ManualSide)
    by_bc = {d.bc_id: d for d in match.decisions}
    assert by_bc[100].command_role == "field_commander"
    assert by_bc[100].attribution_weight == pytest.approx(0.85)
    assert by_bc[100].attribution_method == "manual"
    assert by_bc[101].hierarchy_rank == 1
    assert by_bc[101].reports_to_bc_id == 100
    assert "ov-0001" in by_bc[101].reason
    assert match.evidence[101].startswith("[manual ov-0001] Plutarch")
    assert match.evidence[101].endswith("listed in infobox")


def test_evidence_prefix_is_replaced_not_stacked() -> None:
    first = match_attribution(_attribution(GOOD_VALUE), [_side()], SLUGS)
    assert isinstance(first, ManualSide)
    stored = first.evidence[101]
    side = _side()
    reloaded = replace(
        side, commanders=tuple(replace(c, role_evidence=stored) for c in side.commanders)
    )

    again = match_attribution(_attribution(GOOD_VALUE), [reloaded], SLUGS)

    assert isinstance(again, ManualSide)
    assert again.evidence[101] == stored


def test_listed_set_must_equal_linked_set() -> None:
    value = [_item("Q211101", "field_commander", 0, 1.0)]

    reason = match_attribution(_attribution(value), [_side()], SLUGS)

    assert isinstance(reason, str)
    assert "Augustus" in reason and "linked but not listed" in reason


def test_an_extra_general_is_named() -> None:
    value = [*GOOD_VALUE, _item("Mark Antony", "subordinate", 1, 0.0, reports_to="Q211101")]

    reason = match_attribution(_attribution(value), [_side()], SLUGS)

    assert isinstance(reason, str)
    assert "Mark Antony" in reason and "listed but not linked" in reason


def test_validator_failure_makes_the_entry_unmatched() -> None:
    value = [
        _item("Q211101", "field_commander", 0, 0.5),
        _item("Q1405", "sovereign", 1, 0.4, reports_to="Q211101"),
    ]

    reason = match_attribution(_attribution(value), [_side()], SLUGS)

    assert isinstance(reason, str)
    assert "role rules" in reason and "sum" in reason


def test_unknown_side_and_side_without_commanders_are_unmatched() -> None:
    # A side with no linked commanders is not loaded as a group at all.
    assert isinstance(match_attribution(_attribution(GOOD_VALUE), [], SLUGS), str)
    other = _attribution(GOOD_VALUE, side="Antony")
    assert isinstance(match_attribution(other, [_side()], SLUGS), str)


def test_battle_types_match_by_slug_and_report_missing_ones() -> None:
    found = _entry("battle_type", {"battle": " BATTLE_of_Actium"}, "naval")
    missing = _entry("battle_type", {"battle": "nowhere"}, "field", "ov-0002")

    results = dict((e.id, m) for e, m in match_battle_types([found, missing], SLUGS))

    assert results["ov-0001"] == ManualBattleType(found, 1, "naval")
    assert isinstance(results["ov-0002"], str)


def test_ambiguous_slug_is_unmatched() -> None:
    entry = _entry("battle_type", {"battle": "dup"}, "field")

    [(_, match)] = match_battle_types([entry], {1: "Dup", 2: "dup"})

    assert isinstance(match, str) and "ambiguous" in match


def test_report_covers_every_entry_exactly_once(tmp_path: Path) -> None:
    entries = [
        _entry("battle_type", {"battle": "battle_of_actium"}, "naval"),
        _entry("battle_type", {"battle": "nowhere"}, "field", "ov-0002"),
        replace(_attribution(GOOD_VALUE), id="ov-0003"),
        _entry(
            "attribution", {"battle": "battle_of_actium", "side": "ghost"}, GOOD_VALUE, "ov-0004"
        ),
    ]
    overrides = OverrideSet(entries=tuple(entries), path=tmp_path / "overrides.yaml")
    report = OverrideReport("classify", overrides)

    classify_stage._apply_battle_type_overrides(entries, SLUGS, report)
    applied = classify_stage._apply_attribution_overrides(entries, [_side()], SLUGS, report)
    report.write(tmp_path)

    payload = json.loads((tmp_path / "overrides_classify.json").read_text(encoding="utf-8"))
    assert len(applied) == 1
    assert payload["applied"] == {"battle_type": 1, "attribution": 1}
    assert sorted(u["id"] for u in payload["unmatched"]) == ["ov-0002", "ov-0004"]


def test_overridden_side_generates_no_llm_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two commanders with no roles: the deterministic rule falls back to
    # default_split, which is what normally earns an LLM request.
    side = _side()
    overrides_file = tmp_path / "overrides.yaml"
    overrides_file.write_text(
        "version: 1\nentries:\n"
        "  - id: ov-0001\n    kind: attribution\n"
        "    target: {battle: battle_of_actium, side: Octavian}\n"
        "    value:\n"
        "      - {general: Q211101, role: field_commander, rank: 0, weight: 0.85}\n"
        "      - {general: Augustus, role: sovereign, rank: 1, weight: 0.15, reports_to: Q211101}\n"
        "    source: Plutarch\n    note: ''\n    reviewed: {by: lee, on: 2026-10-17}\n",
        encoding="utf-8",
    )
    spec: dict[str, Any] = {"params": {"overrides_path": str(overrides_file)}}
    monkeypatch.setattr(classify_stage, "load_side_groups", lambda _conn: [side])
    monkeypatch.setattr(classify_stage, "load_battle_wikipedia_urls", lambda _conn: {1: None})
    monkeypatch.setattr(classify_stage, "load_battle_source_slugs", lambda _conn: SLUGS)
    monkeypatch.setattr(
        classify_stage, "build_role_request", lambda *_a, **_k: pytest.fail("request built")
    )

    assert classify_stage.build_pending_role_requests(None, tmp_path, spec) == []

    spec_without = {"params": {"overrides_path": str(tmp_path / "none.yaml")}}
    with pytest.raises(Exception, match="not found"):
        classify_stage.build_pending_role_requests(None, tmp_path, spec_without)


def test_overridden_side_ids_ignores_unmatched_entries() -> None:
    good = _attribution(GOOD_VALUE)
    bad = _attribution([_item("Q211101", "field_commander", 0, 1.0)], side="Octavian")
    bad = replace(bad, id="ov-0002")

    assert overridden_side_ids([good], [_side()], SLUGS) == {10}
    assert overridden_side_ids([bad], [_side()], SLUGS) == set()


def test_rule_pass_skips_battles_a_manual_type_covers(monkeypatch: pytest.MonkeyPatch) -> None:
    battle = {
        "battle_id": 1,
        "name": "Battle of Actium",
        "wikipedia_url": None,
        "source_slug": None,
        "current_type": "unknown",
        "span_days": None,
        "date_precision": "day",
    }
    monkeypatch.setattr(classify_stage, "load_battles_needing_type", lambda _conn: [battle])
    counts = ClassifyCounts()

    skipped = classify_stage._infer_battle_types(None, Path("."), counts, frozenset({1}))

    assert skipped == []
    assert counts.battles_typed == 0
