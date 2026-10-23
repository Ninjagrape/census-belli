"""Loading, validating and reporting manual overrides."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.overrides import (
    BATTLE_TYPES,
    COMMAND_ROLES,
    DEFAULT_PATH,
    KINDS_BY_STAGE,
    OUTCOME_LEVELS,
    OverrideError,
    OverrideReport,
    clear_report,
    file_sha256,
    load_overrides,
    overrides_path,
    target_key,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "config" / "schema.sql"
REVIEWED = {"by": "lee", "on": "2026-10-17"}


def _entry(n: int, kind: str, target: dict[str, str], value: Any) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": kind,
        "target": target,
        "value": value,
        "source": "Chandler (1966), p. 807",
        "note": "",
        "reviewed": dict(REVIEWED),
    }


def _all_kinds() -> list[dict[str, Any]]:
    return [
        _entry(
            1, "outcome", {"battle": "borodino"}, {"French Empire": "victory", "Russia": "defeat"}
        ),
        _entry(
            2,
            "battle_date",
            {"battle": "borodino"},
            {"start": "1812-09-07", "end": None, "precision": "day"},
        ),
        _entry(3, "side_polity", {"battle": "borodino", "side": "French"}, "French Empire"),
        _entry(
            4,
            "commander_link",
            {"battle": "borodino", "side": "French", "mention": "Ney"},
            "Q131",
        ),
        _entry(5, "general_merge", {"general": "Ney"}, "Q131"),
        _entry(6, "battle_type", {"battle": "borodino"}, "field"),
        _entry(
            7,
            "attribution",
            {"battle": "borodino", "side": "French"},
            [
                {"general": "Q517", "role": "supreme_commander", "rank": 0, "weight": 0.6},
                {
                    "general": "Ney",
                    "role": "subordinate",
                    "rank": 1,
                    "weight": 0.4,
                    "reports_to": "Q517",
                },
            ],
        ),
        _entry(
            8,
            "battle_split",
            {"battle": "borodino"},
            [
                {"key": "shevardino", "name": "Battle of Shevardino", "heading": "Shevardino"},
                {"key": "main", "name": "Battle of the Moskva", "heading": "Main"},
            ],
        ),
        _entry(
            9,
            "troop_strength",
            {"battle": "borodino", "side": "French"},
            {"point": 130_000, "scope": "engaged"},
        ),
    ]


def _write(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "entries": entries}), encoding="utf-8")
    return path


def _expect_error(tmp_path: Path, entries: list[dict[str, Any]], match: str) -> None:
    with pytest.raises(OverrideError, match=match):
        load_overrides(_write(tmp_path, entries))


def test_enums_match_schema() -> None:
    sql = SCHEMA.read_text(encoding="utf-8")

    def labels(name: str) -> set[str]:
        body = re.search(rf"CREATE TYPE {name} AS ENUM \((.*?)\);", sql, re.DOTALL)
        assert body is not None
        return set(re.findall(r"'([^']+)'", re.sub(r"--[^\n]*", "", body.group(1))))

    assert labels("outcome_level") == OUTCOME_LEVELS
    assert labels("battle_type") == BATTLE_TYPES
    assert labels("command_role") == COMMAND_ROLES
    assert "victory" in OUTCOME_LEVELS
    assert "campaign" in BATTLE_TYPES


def test_full_valid_file_covers_every_kind(tmp_path: Path) -> None:
    loaded = load_overrides(_write(tmp_path, _all_kinds()))

    assert len(loaded) == 9
    assert {e.kind for e in loaded.entries} == {k for ks in KINDS_BY_STAGE.values() for k in ks}
    assert [e.kind for e in loaded.for_stage("extract")] == [
        "outcome",
        "battle_date",
        "side_polity",
        "battle_split",
        "troop_strength",
    ]
    assert [e.id for e in loaded.of_kind("attribution")] == ["ov-0007"]
    assert loaded.for_stage("nothing") == []
    attribution = loaded.of_kind("attribution")[0].value
    assert attribution[0]["reports_to"] is None
    assert attribution[1]["reports_to"] == "Q517"
    assert loaded.entries[0].target["battle"] == "borodino"
    with pytest.raises(TypeError):
        loaded.entries[0].target["battle"] = "x"  # type: ignore[index]


def test_empty_file_and_absent_file(tmp_path: Path) -> None:
    assert len(load_overrides(_write(tmp_path, []))) == 0
    assert len(load_overrides(tmp_path / "missing.yaml")) == 0


def test_shipped_file_is_valid() -> None:
    known_kinds = {k for ks in KINDS_BY_STAGE.values() for k in ks}

    loaded = load_overrides(DEFAULT_PATH, required=True)

    assert all(e.kind in known_kinds for e in loaded.entries)


def test_default_path_is_anchored_to_the_repo_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = load_overrides(DEFAULT_PATH)
    monkeypatch.chdir(tmp_path)

    assert DEFAULT_PATH == ROOT / "config" / "overrides.yaml"
    loaded = load_overrides()
    assert len(loaded) == len(expected)
    assert [e.id for e in loaded.entries] == [e.id for e in expected.entries]


def test_required_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(OverrideError, match="not found"):
        load_overrides(tmp_path / "missing.yaml", required=True)


@pytest.mark.parametrize(
    "document",
    [
        {"entries": []},
        {"version": 1},
        {"version": 2, "entries": []},
        {"version": 1, "entries": [], "extra": 1},
        {"version": 1, "entries": {}},
        ["not", "a", "mapping"],
    ],
)
def test_bad_top_level_fails(tmp_path: Path, document: Any) -> None:
    path = tmp_path / "o.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(OverrideError):
        load_overrides(path)


def test_unparseable_yaml_fails(tmp_path: Path) -> None:
    path = tmp_path / "o.yaml"
    path.write_text("version: [1\n", encoding="utf-8")
    with pytest.raises(OverrideError):
        load_overrides(path)


def test_reviewed_on_accepts_a_yaml_date_and_an_iso_string(tmp_path: Path) -> None:
    path = tmp_path / "o.yaml"
    path.write_text(
        "version: 1\nentries:\n"
        "- {id: ov-0001, kind: battle_type, target: {battle: b}, value: field,"
        " source: s, note: '', reviewed: {by: lee, on: 2026-10-17}}\n"
        "- {id: ov-0002, kind: battle_type, target: {battle: c}, value: field,"
        " source: s, note: '', reviewed: {by: lee, on: '2026-10-18'}}\n",
        encoding="utf-8",
    )

    entries = load_overrides(path).entries

    assert [e.reviewed_on for e in entries] == ["2026-10-17", "2026-10-18"]
    assert entries[0].reviewed_by == "lee"


def test_bad_reviewed_on_fails(tmp_path: Path) -> None:
    entry = _entry(1, "battle_type", {"battle": "b"}, "field")
    entry["reviewed"]["on"] = "yesterday"
    _expect_error(tmp_path, [entry], "ov-0001.*reviewed.on")


def test_battle_date_coerces_yaml_dates_to_strings(tmp_path: Path) -> None:
    path = tmp_path / "o.yaml"
    path.write_text(
        "version: 1\nentries:\n"
        "- {id: ov-0001, kind: battle_date, target: {battle: b},"
        " value: {start: 1812-09-07, end: null, precision: day},"
        " source: s, note: '', reviewed: {by: lee, on: 2026-10-17}}\n",
        encoding="utf-8",
    )

    assert load_overrides(path).entries[0].value["start"] == "1812-09-07"


def _mutated(index: int, mutate: Any) -> list[dict[str, Any]]:
    entries = copy.deepcopy(_all_kinds())
    mutate(entries[index])
    return entries


@pytest.mark.parametrize(
    ("index", "mutate", "match"),
    [
        (0, lambda e: e.update(id="ov-1"), "ov-1.*id must match"),
        (0, lambda e: e.update(source="   "), "ov-0001.*source"),
        (0, lambda e: e.update(note=3), "ov-0001.*note"),
        (0, lambda e: e.update(kind="nope"), "ov-0001.*unknown kind"),
        (0, lambda e: e.update(extra=1), "ov-0001.*unknown"),
        (0, lambda e: e.pop("note"), "ov-0001.*missing"),
        (0, lambda e: e["reviewed"].update(by=""), "ov-0001.*reviewed.by"),
        (0, lambda e: e["reviewed"].update(extra=1), "ov-0001.*reviewed"),
        (0, lambda e: e.update(target={"battle": "b", "side": "x"}), "ov-0001.*target keys"),
        (0, lambda e: e.update(target={"battle": ""}), "ov-0001.*target.battle"),
        (0, lambda e: e.update(value={"French": "victory"}), "ov-0001.*at least 2"),
        (0, lambda e: e.update(value={"A": "victory", "B": "won"}), "ov-0001.*outcome_level"),
        (1, lambda e: e["value"].update(precision="eon"), "ov-0002.*precision"),
        (1, lambda e: e["value"].pop("end"), "ov-0002.*value keys"),
        (1, lambda e: e["value"].update(start=""), "ov-0002.*start"),
        (2, lambda e: e.update(value=""), "ov-0003.*value"),
        (3, lambda e: e.update(value="Q0"), "ov-0004.*Q-id"),
        (3, lambda e: e.update(value="Ney"), "ov-0004.*Q-id"),
        (4, lambda e: e.update(value="Q1", target={"general": "Q1"}), "ov-0005.*itself"),
        (4, lambda e: e.update(value="Ney"), "ov-0005.*Q-id"),
        (5, lambda e: e.update(value="skirmish"), "ov-0006.*battle_type"),
        (6, lambda e: e.update(value=[]), "ov-0007.*non-empty list"),
        (6, lambda e: e["value"][0].update(role="king"), "ov-0007.*command_role"),
        (6, lambda e: e["value"][0].update(rank=-1), "ov-0007.*rank"),
        (6, lambda e: e["value"][0].update(rank=True), "ov-0007.*rank"),
        (6, lambda e: e["value"][0].update(weight=1.5), "ov-0007.*weight"),
        (6, lambda e: e["value"][0].pop("weight"), "ov-0007.*missing"),
        (6, lambda e: e["value"][0].update(color="red"), "ov-0007.*unknown"),
        (6, lambda e: e["value"][1].update(general="Q517"), "ov-0007.*twice"),
        (6, lambda e: e["value"][1].update(reports_to="Nobody"), "ov-0007.*reports_to"),
        (6, lambda e: e["value"][1].update(reports_to="Ney"), "ov-0007.*reports_to"),
    ],
)
def test_rule_violation_names_entry_and_problem(
    tmp_path: Path, index: int, mutate: Any, match: str
) -> None:
    _expect_error(tmp_path, _mutated(index, mutate), match)


SPLIT = 7


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda e: e.update(value=[e["value"][0]]), "ov-0008.*at least 2 parts"),
        (lambda e: e.update(value="jena"), "ov-0008.*at least 2 parts"),
        (lambda e: e["value"][0].update(extra=1), "ov-0008.*value.0. keys"),
        (lambda e: e["value"][0].pop("heading"), "ov-0008.*value.0. keys"),
        (lambda e: e["value"][0].update(key="Jena Part"), "ov-0008.*lowercase slug-safe"),
        (lambda e: e["value"][0].update(key="a#b"), "ov-0008.*lowercase slug-safe"),
        (lambda e: e["value"][0].update(name=" "), "ov-0008.*value.0..name"),
        (lambda e: e["value"][1].update(key=e["value"][0]["key"]), "ov-0008.*duplicate part key"),
        (
            lambda e: e["value"][1].update(heading=" " + e["value"][0]["heading"].upper()),
            "ov-0008.*duplicate part heading",
        ),
        (lambda e: e.update(target={"battle": "b", "side": "x"}), "ov-0008.*target keys"),
    ],
)
def test_battle_split_violations_name_the_entry(tmp_path: Path, mutate: Any, match: str) -> None:
    _expect_error(tmp_path, _mutated(SPLIT, mutate), match)


def test_battle_split_normalises_its_parts(tmp_path: Path) -> None:
    loaded = load_overrides(_write(tmp_path, _all_kinds()))

    (split,) = loaded.of_kind("battle_split")

    assert split.stage == "extract"
    assert split.value[0] == {
        "key": "shevardino",
        "name": "Battle of Shevardino",
        "heading": "Shevardino",
    }


def test_duplicate_id_fails(tmp_path: Path) -> None:
    first = _entry(1, "battle_type", {"battle": "a"}, "field")
    second = _entry(1, "battle_type", {"battle": "b"}, "naval")
    _expect_error(tmp_path, [first, second], "ov-0001.*duplicate id")


def test_duplicate_target_fails_after_normalising(tmp_path: Path) -> None:
    first = _entry(1, "battle_type", {"battle": "Borodino"}, "field")
    second = _entry(2, "battle_type", {"battle": " borodino "}, "naval")
    _expect_error(tmp_path, [first, second], "ov-0002.*duplicate target.*ov-0001")


def test_same_target_in_different_kinds_is_allowed(tmp_path: Path) -> None:
    first = _entry(1, "battle_type", {"battle": "b"}, "field")
    second = _entry(2, "outcome", {"battle": "b"}, {"A": "victory", "B": "defeat"})
    assert len(load_overrides(_write(tmp_path, [first, second]))) == 2


def test_overrides_path_prefers_the_spec_param(tmp_path: Path) -> None:
    assert overrides_path({}) == (DEFAULT_PATH, False)
    assert overrides_path({"overrides_path": None}) == (DEFAULT_PATH, False)
    assert overrides_path({"overrides_path": str(tmp_path / "x.yaml")}) == (
        tmp_path / "x.yaml",
        True,
    )


def test_report_shape_lists_every_kind_even_at_zero(tmp_path: Path) -> None:
    loaded = load_overrides(_write(tmp_path, _all_kinds()))
    report = OverrideReport("extract", loaded)
    outcome, battle_date, side_polity, split, strength = loaded.for_stage("extract")
    report.applied(outcome)
    report.applied(battle_date)
    report.applied(split)
    report.applied(strength)
    report.unmatched(side_polity, "no such side")

    path = report.write(tmp_path / "out")

    assert path == tmp_path / "out" / "overrides_extract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "extract"
    assert payload["file"] == str(loaded.path)
    assert payload["entries"] == 5
    assert payload["applied"] == {
        "outcome": 1,
        "battle_date": 1,
        "side_polity": 0,
        "battle_split": 1,
        "troop_strength": 1,
    }
    assert payload["entry_ids"] == ["ov-0001", "ov-0002", "ov-0003", "ov-0008", "ov-0009"]
    assert payload["file_sha256"] == file_sha256(loaded.path)
    assert payload["unmatched"] == [
        {"id": "ov-0003", "kind": "side_polity", "reason": "no such side"}
    ]
    assert payload["limited"] is False
    assert "T" in payload["written_at"]


def test_report_limited_and_unknown_stage(tmp_path: Path) -> None:
    loaded = load_overrides(tmp_path / "absent.yaml")
    report = OverrideReport("classify", loaded, limited=True)

    payload = json.loads(report.write(tmp_path).read_text(encoding="utf-8"))

    assert payload["limited"] is True
    assert payload["applied"] == {"battle_type": 0, "attribution": 0}
    assert payload["entries"] == 0
    with pytest.raises(ValueError, match="unknown stage"):
        OverrideReport("report", loaded)


def test_orchestrator_rejects_a_bad_overrides_file_before_any_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import orchestrator

    bad = tmp_path / "overrides.yaml"
    bad.write_text("version: 2\nentries: []\n", encoding="utf-8")
    monkeypatch.setattr("pipeline.overrides.DEFAULT_PATH", bad)
    ran: list[str] = []
    monkeypatch.setattr(orchestrator, "run_stage", lambda *a, **k: ran.append("x"))

    with pytest.raises(SystemExit) as raised:
        orchestrator.run_pipeline(["extract"])

    assert raised.value.code == 1
    assert ran == []


# ─── Report coverage ─────────────────────────────────────────────────────────


def test_write_names_entries_never_reported(tmp_path: Path) -> None:
    loaded = load_overrides(_write(tmp_path, _all_kinds()))
    report = OverrideReport("extract", loaded)
    report.applied(loaded.for_stage("extract")[0])

    with pytest.raises(ValueError, match=r"ov-0002.*ov-0003"):
        report.write(tmp_path / "out")
    assert not (tmp_path / "out" / "overrides_extract.json").exists()


def test_write_names_entries_reported_twice(tmp_path: Path) -> None:
    loaded = load_overrides(_write(tmp_path, _all_kinds()))
    report = OverrideReport("resolve", loaded)
    commander_link, merge = loaded.for_stage("resolve")
    report.applied(commander_link)
    report.applied(commander_link)
    report.unmatched(merge, "gone")

    with pytest.raises(ValueError, match="more than once.*ov-0004"):
        report.write(tmp_path)


def test_reporting_an_entry_of_another_stage_is_a_value_error(tmp_path: Path) -> None:
    loaded = load_overrides(_write(tmp_path, _all_kinds()))
    report = OverrideReport("extract", loaded)
    foreign = loaded.for_stage("classify")[0]

    with pytest.raises(ValueError, match="ov-0006.*classify"):
        report.applied(foreign)
    with pytest.raises(ValueError, match="ov-0006.*classify"):
        report.unmatched(foreign, "x")


def test_report_hash_is_null_for_an_absent_file(tmp_path: Path) -> None:
    report = OverrideReport("classify", load_overrides(tmp_path / "absent.yaml"))

    payload = json.loads(report.write(tmp_path).read_text(encoding="utf-8"))

    assert payload["file_sha256"] is None
    assert payload["entry_ids"] == []


def test_clear_report_removes_a_stale_report(tmp_path: Path) -> None:
    report = OverrideReport("classify", load_overrides(tmp_path / "absent.yaml"))
    path = report.write(tmp_path)

    assert clear_report("classify", tmp_path) is True
    assert not path.exists()
    assert clear_report("classify", tmp_path) is False


# ─── Validation ──────────────────────────────────────────────────────────────


def _attribution(*items: dict[str, Any]) -> dict[str, Any]:
    return _entry(1, "attribution", {"battle": "b", "side": "s"}, list(items))


def _person(name: str, reports_to: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"general": name, "role": "subordinate", "rank": 1, "weight": 0.1}
    if reports_to is not None:
        item["reports_to"] = reports_to
    return item


def test_attribution_rejects_a_two_cycle(tmp_path: Path) -> None:
    entry = _attribution(_person("A", "B"), _person("B", "A"))
    _expect_error(tmp_path, [entry], "ov-0001.*cycle")


def test_attribution_rejects_a_longer_cycle(tmp_path: Path) -> None:
    entry = _attribution(_person("A", "B"), _person("B", "C"), _person("C", "A"), _person("D"))
    _expect_error(tmp_path, [entry], "ov-0001.*cycle")


def test_attribution_accepts_a_chain(tmp_path: Path) -> None:
    entry = _attribution(_person("A"), _person("B", "A"), _person("C", "B"))
    assert len(load_overrides(_write(tmp_path, [entry]))) == 1


def test_attribution_generals_compare_after_strip_and_casefold(tmp_path: Path) -> None:
    _expect_error(tmp_path, [_attribution(_person("Ney"), _person(" ney "))], "twice")
    resolved = _attribution(_person("Ney"), _person("Soult", " NEY "))
    assert len(load_overrides(_write(tmp_path, [resolved]))) == 1
    _expect_error(tmp_path, [_attribution(_person("q517"), _person("Q517"))], "twice")


def test_general_merge_rejects_a_chain(tmp_path: Path) -> None:
    first = _entry(1, "general_merge", {"general": "Ney"}, "Q131")
    second = _entry(2, "general_merge", {"general": "Q131"}, "Q517")
    _expect_error(tmp_path, [first, second], "ov-0001.*itself merged by ov-0002")


def test_general_merge_rejects_a_cycle(tmp_path: Path) -> None:
    first = _entry(1, "general_merge", {"general": "Q1"}, "Q2")
    second = _entry(2, "general_merge", {"general": "q2"}, "Q1")
    _expect_error(tmp_path, [first, second], "itself merged")


def test_general_merge_self_check_ignores_case_of_the_qid(tmp_path: Path) -> None:
    _expect_error(
        tmp_path, [_entry(1, "general_merge", {"general": " q1 "}, "Q1")], "ov-0001.*itself"
    )


def test_general_merge_to_distinct_final_qids_is_allowed(tmp_path: Path) -> None:
    first = _entry(1, "general_merge", {"general": "Ney"}, "Q131")
    second = _entry(2, "general_merge", {"general": "Soult"}, "Q131")
    assert len(load_overrides(_write(tmp_path, [first, second]))) == 2


@pytest.mark.parametrize(
    ("index", "mutate", "match"),
    [
        (0, lambda e: e.update(value={"A": [1], "B": "defeat"}), "ov-0001.*outcome_level"),
        (0, lambda e: e.update(value={"A": "victory", " A ": "defeat"}), "ov-0001.*twice"),
        (1, lambda e: e["value"].update(precision=["day"]), "ov-0002.*precision"),
        (5, lambda e: e.update(value=["field"]), "ov-0006.*battle_type"),
        (6, lambda e: e["value"][0].update(role={"x": 1}), "ov-0007.*command_role"),
        (1, lambda e: e["value"].update(start="1812-09-08", end="1812-09-07"), "ov-0002.*after"),
        (1, lambda e: e["value"].update(start="1812-09-31", end="1812-10-01"), "ov-0002.*real"),
    ],
)
def test_more_rule_violations(tmp_path: Path, index: int, mutate: Any, match: str) -> None:
    _expect_error(tmp_path, _mutated(index, mutate), match)


def test_battle_date_rejects_datetimes_and_passes_other_formats(tmp_path: Path) -> None:
    path = tmp_path / "o.yaml"
    path.write_text(
        "version: 1\nentries:\n"
        "- {id: ov-0001, kind: battle_date, target: {battle: b},"
        " value: {start: 1812-09-07 10:00:00, end: null, precision: day},"
        " source: s, note: '', reviewed: {by: lee, on: 2026-10-17}}\n",
        encoding="utf-8",
    )
    with pytest.raises(OverrideError, match="ov-0001.*datetime"):
        load_overrides(path)

    free = _entry(
        1,
        "battle_date",
        {"battle": "b"},
        {"start": "c. 490 BC", "end": "1812", "precision": "year"},
    )
    assert load_overrides(_write(tmp_path, [free])).entries[0].value["start"] == "c. 490 BC"
    same_day = _entry(
        2,
        "battle_date",
        {"battle": "c"},
        {"start": "1812-09-07", "end": "1812-09-07", "precision": "day"},
    )
    assert len(load_overrides(_write(tmp_path, [same_day]))) == 1


def test_reviewed_rejects_both_true_and_on_keys(tmp_path: Path) -> None:
    path = tmp_path / "o.yaml"
    path.write_text(
        "version: 1\nentries:\n"
        "- {id: ov-0001, kind: battle_type, target: {battle: b}, value: field,"
        " source: s, note: '', reviewed: {by: lee, 'on': 2026-10-17, true: 2026-10-18}}\n",
        encoding="utf-8",
    )
    with pytest.raises(OverrideError, match="ov-0001.*both"):
        load_overrides(path)


def test_target_key_strips_casefolds_and_uppercases_general_qids(tmp_path: Path) -> None:
    loaded = load_overrides(
        _write(
            tmp_path,
            [
                _entry(1, "side_polity", {"battle": " Borodino ", "side": "FRENCH"}, "x"),
                _entry(2, "general_merge", {"general": " q517 "}, "Q1"),
                _entry(3, "general_merge", {"general": " Ney "}, "Q2"),
            ],
        )
    )
    side, qid, name = loaded.entries

    assert target_key(side) == ("side_polity", "borodino", "french")
    assert target_key(qid) == ("general_merge", "Q517")
    assert target_key(name) == ("general_merge", "ney")
