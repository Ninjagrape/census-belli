"""Unit tests for applying manual overrides to the extract stage's records."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.extractors import BattleExtraction, BattleFacts, MissingField, SideExtraction
from pipeline.extractors.outcome import OUTCOME_STAGE, OutcomePass
from pipeline.extractors.override_apply import (
    apply_extract_overrides,
    carries_override,
    outcome_override_slugs,
)
from pipeline.extractors.records import Disagreement, ExtractionFailure
from pipeline.llm import CallStatus, LLMRequest, LLMResponse, LLMService
from pipeline.overrides import OverrideEntry, OverrideReport, load_overrides
from pipeline.stages import extract

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"


def _entry(n: int, kind: str, target: dict[str, str], value: Any) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": kind,
        "target": target,
        "value": value,
        "source": f"Citation {n}",
        "note": "",
        "reviewed": {"by": "lee", "on": "2026-10-17"},
    }


def _load(tmp_path: Path, *raw: dict[str, Any]) -> list[OverrideEntry]:
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "entries": list(raw)}), encoding="utf-8")
    return load_overrides(path).for_stage("extract")


def _battle(slug: str = "borodino") -> BattleExtraction:
    return BattleExtraction(
        slug=slug,
        name="Battle of Borodino",
        facts=BattleFacts(
            date_start="1812-09-08",
            date_precision="day",
            date_circa=True,
            outcome_level="victory",
        ),
        sides=[
            SideExtraction(label="French Empire", outcome="victory"),
            SideExtraction(label="Russian Empire", outcome="defeat"),
        ],
        missing=[
            MissingField(field_name="outcome", side_label="French Empire"),
            MissingField(field_name="outcome_level"),
            MissingField(field_name="date_start"),
            MissingField(field_name="location_name"),
        ],
        failures=[
            ExtractionFailure(
                battle_name="x", source_ref="s", status="unparsed_date", error="bad date"
            )
        ],
    )


def _apply(
    tmp_path: Path, battles: list[BattleExtraction], *raw: dict[str, Any], limited: bool = False
) -> tuple[dict[str, OverrideEntry], dict[str, Any]]:
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "entries": list(raw)}), encoding="utf-8")
    overrides = load_overrides(path)
    report = OverrideReport("extract", overrides, limited=limited)
    applied = apply_extract_overrides(battles, overrides.for_stage("extract"), report)
    report.write(tmp_path)
    payload = json.loads((tmp_path / "overrides_extract.json").read_text(encoding="utf-8"))
    return applied, payload


_OUTCOME = {"French Empire": "pyrrhic_victory", "Russian Empire": "defeat"}


def test_outcome_override_sets_every_side_and_marks_the_method(tmp_path: Path) -> None:
    battle = _battle()

    applied, report = _apply(
        tmp_path, [battle], _entry(1, "outcome", {"battle": "Borodino "}, _OUTCOME)
    )

    assert [s.outcome for s in battle.sides] == ["pyrrhic_victory", "defeat"]
    assert battle.outcome_method == "manual"
    assert battle.facts.outcome_level == "pyrrhic_victory"
    assert battle.notes == ["override ov-0001 applied (outcome)"]
    assert list(applied) == ["borodino"]
    assert report["applied"]["outcome"] == 1 and report["unmatched"] == []


def test_outcome_override_drops_outcome_gap_rows_only(tmp_path: Path) -> None:
    battle = _battle()

    _apply(tmp_path, [battle], _entry(1, "outcome", {"battle": "borodino"}, _OUTCOME))

    assert [m.field_name for m in battle.missing] == ["date_start", "location_name"]


def test_outcome_override_matches_side_labels_after_strip_and_casefold(tmp_path: Path) -> None:
    battle = _battle()
    value = {" french EMPIRE ": "victory", "russian empire": "defeat"}

    _, report = _apply(tmp_path, [battle], _entry(1, "outcome", {"battle": "borodino"}, value))

    assert report["unmatched"] == []
    assert battle.sides[0].outcome == "victory"


def test_outcome_override_with_a_different_side_set_changes_nothing(tmp_path: Path) -> None:
    battle = _battle()
    value = {"French Empire": "victory", "Prussia": "defeat"}

    applied, report = _apply(
        tmp_path, [battle], _entry(1, "outcome", {"battle": "borodino"}, value)
    )

    assert applied == {}
    assert [s.outcome for s in battle.sides] == ["victory", "defeat"]
    assert battle.outcome_method == ""
    assert battle.notes == []
    (miss,) = report["unmatched"]
    assert "Prussia" in miss["reason"] and "Russian Empire" in miss["reason"]


def test_override_on_an_unknown_battle_is_unmatched(tmp_path: Path) -> None:
    _, report = _apply(
        tmp_path,
        [_battle()],
        _entry(1, "outcome", {"battle": "waterloo"}, _OUTCOME),
        _entry(2, "battle_date", {"battle": "waterloo"}, _date("1815-06-18", None, "day")),
        _entry(3, "side_polity", {"battle": "waterloo", "side": "x"}, "France"),
    )

    assert {u["reason"] for u in report["unmatched"]} == {"battle not found"}
    assert len(report["unmatched"]) == 3


def _date(start: str, end: str | None, precision: str) -> dict[str, Any]:
    return {"start": start, "end": end, "precision": precision}


def test_battle_date_override_sets_the_facts_and_clears_circa(tmp_path: Path) -> None:
    battle = _battle()

    _, report = _apply(
        tmp_path,
        [battle],
        _entry(1, "battle_date", {"battle": "borodino"}, _date("1812-09-07", "1812-09-08", "day")),
    )

    facts = battle.facts
    assert (facts.date_start, facts.date_end, facts.date_precision, facts.date_circa) == (
        "1812-09-07",
        "1812-09-08",
        "day",
        None,
    )
    assert report["unmatched"] == []


def test_battle_date_override_drops_the_date_gap_and_unparsed_failure(tmp_path: Path) -> None:
    battle = _battle()

    _apply(
        tmp_path,
        [battle],
        _entry(1, "battle_date", {"battle": "borodino"}, _date("1812-09-07", None, "day")),
    )

    assert "date_start" not in [m.field_name for m in battle.missing]
    assert battle.failures == []


def test_battle_date_override_keeps_a_bc_date_as_a_postgres_literal(tmp_path: Path) -> None:
    battle = _battle()

    _apply(
        tmp_path,
        [battle],
        _entry(
            1,
            "battle_date",
            {"battle": "borodino"},
            _date("0480-09-01 BC", "0480-09-30 BC", "month"),
        ),
    )

    assert battle.facts.date_start == "0480-09-01 BC"
    assert battle.facts.date_end == "0480-09-30 BC"
    assert battle.facts.date_precision == "month"


@pytest.mark.parametrize(
    ("start", "end", "reason"),
    [
        ("0480-09-30 BC", "0480-09-01 BC", "before start"),
        ("not a date", None, "not a date"),
        ("1812-09-07", "garbage", "not a date"),
        ("21-22 May 1809", None, "range"),
        ("1812-09-07", "1812", "full day"),
    ],
)
def test_battle_date_override_the_extractor_cannot_represent_is_unmatched(
    tmp_path: Path, start: str, end: str | None, reason: str
) -> None:
    battle = _battle()

    _, report = _apply(
        tmp_path,
        [battle],
        _entry(1, "battle_date", {"battle": "borodino"}, _date(start, end, "day")),
    )

    (miss,) = report["unmatched"]
    assert reason in miss["reason"]
    assert battle.facts.date_start == "1812-09-08"
    assert battle.notes == []


def test_bc_end_before_start_compares_by_astronomical_year(tmp_path: Path) -> None:
    # 100 BC is later than 200 BC, so 200 BC to 100 BC is in order.
    battle = _battle()

    _, report = _apply(
        tmp_path,
        [battle],
        _entry(
            1,
            "battle_date",
            {"battle": "borodino"},
            _date("0200-01-01 BC", "0100-01-01 BC", "year"),
        ),
    )

    assert report["unmatched"] == []
    assert battle.facts.date_end == "0100-01-01 BC"


def test_a_circa_start_text_keeps_the_circa_flag(tmp_path: Path) -> None:
    battle = _battle()

    _apply(
        tmp_path,
        [battle],
        _entry(1, "battle_date", {"battle": "borodino"}, _date("c. 1812", None, "year")),
    )

    assert battle.facts.date_circa is True
    assert battle.facts.date_precision == "year"


def test_side_polity_override_sets_the_polity_and_keeps_the_label(tmp_path: Path) -> None:
    battle = _battle()

    _, report = _apply(
        tmp_path,
        [battle],
        _entry(1, "side_polity", {"battle": "borodino", "side": "RUSSIAN empire"}, "Russia"),
    )

    assert [(s.label, s.polity) for s in battle.sides] == [
        ("French Empire", None),
        ("Russian Empire", "Russia"),
    ]
    assert report["unmatched"] == []


def test_side_polity_override_on_an_unknown_side_is_unmatched(tmp_path: Path) -> None:
    battle = _battle()

    _, report = _apply(
        tmp_path,
        [battle],
        _entry(1, "side_polity", {"battle": "borodino", "side": "Prussia"}, "Prussia"),
    )

    (miss,) = report["unmatched"]
    assert "Prussia" in miss["reason"] and "French Empire" in miss["reason"]
    assert all(s.polity is None for s in battle.sides)


def test_the_report_covers_every_entry_and_is_limited_on_a_limited_run(tmp_path: Path) -> None:
    _, report = _apply(
        tmp_path,
        [_battle()],
        _entry(1, "outcome", {"battle": "borodino"}, _OUTCOME),
        _entry(2, "battle_date", {"battle": "absent"}, _date("1812-09-07", None, "day")),
        _entry(3, "side_polity", {"battle": "borodino", "side": "French Empire"}, "France"),
        limited=True,
    )

    assert report["entry_ids"] == ["ov-0001", "ov-0002", "ov-0003"]
    assert report["applied"] == {
        "outcome": 1,
        "battle_date": 0,
        "side_polity": 1,
        "battle_split": 0,
    }
    assert [u["id"] for u in report["unmatched"]] == ["ov-0002"]
    assert report["limited"] is True


def test_outcome_override_slugs_are_normalised(tmp_path: Path) -> None:
    entries = _load(
        tmp_path,
        _entry(1, "outcome", {"battle": " Borodino "}, _OUTCOME),
        _entry(2, "side_polity", {"battle": "other", "side": "x"}, "y"),
    )

    assert outcome_override_slugs(entries) == frozenset({"borodino"})


class _Recorder:
    """A provider that records every request and answers with a winner."""

    name = "anthropic"
    model = "claude-sonnet-5-5"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            status=CallStatus.OK,
            provider=self.name,
            model=self.model,
            request_hash="h",
            data={"winner": "Kingdom of Varra", "evidence": "held the field"},
        )


def _contested(tmp_path: Path) -> extract.RawBattle:
    articles = tmp_path / "battles_html"
    articles.mkdir()
    target = articles / "contested.wikitext"
    shutil.copy(FIXTURES / "contested_field.wikitext", target)
    return extract.RawBattle(slug="contested", article_path=target)


def _outcome_pass(provider: _Recorder) -> OutcomePass:
    return OutcomePass(
        service=LLMService(OUTCOME_STAGE, provider, db_conn=None),
        system="sys",
        template="{battle_name} {infobox_result} {sides} {lead_chars} {lead_excerpt}",
        schema={"type": "object"},
    )


def test_an_overridden_battle_issues_no_outcome_request(tmp_path: Path) -> None:
    provider = _Recorder()
    raw = _contested(tmp_path)
    kwargs: dict[str, Any] = {
        "service": None,
        "system": "s",
        "template": "t",
        "schema": {},
        "outcome_pass": _outcome_pass(provider),
    }

    skipped = extract.extract_battle(raw, outcome_overridden=True, **kwargs)
    assert provider.requests == []
    assert all(s.outcome is None for s in skipped.sides)

    asked = extract.extract_battle(raw, **kwargs)
    assert len(provider.requests) == 1
    assert any(s.outcome for s in asked.sides)


def _disputed() -> BattleExtraction:
    battle = _battle()
    battle.disagreements = [
        Disagreement("victor", "France", "wikipedia", {"dbpedia": "Russia"}),
        Disagreement("outcome_level", "victory", "wikipedia", {"dbpedia": "defeat"}),
        Disagreement("date_start", "1812-09-08", "wikipedia", {"dbpedia": "1812-09-07"}),
        Disagreement("wikidata_id", "Q1", "wikipedia", {"dbpedia": "Q2"}),
    ]
    return battle


OUTCOME_ENTRY = _entry(
    1, "outcome", {"battle": "borodino"}, {"French Empire": "victory", "Russian Empire": "defeat"}
)
DATE_ENTRY = _entry(
    2,
    "battle_date",
    {"battle": "borodino"},
    {"start": "1812-09-07", "end": None, "precision": "day"},
)


def test_outcome_override_drops_outcome_disagreements_only(tmp_path: Path) -> None:
    battle = _disputed()

    _apply(tmp_path, [battle], OUTCOME_ENTRY)

    assert {d.field_name for d in battle.disagreements} == {"date_start", "wikidata_id"}
    assert "victor" not in battle.review_notes


def test_battle_date_override_drops_date_disagreements_only(tmp_path: Path) -> None:
    battle = _disputed()
    battle.failures = []

    _apply(tmp_path, [battle], DATE_ENTRY)

    assert {d.field_name for d in battle.disagreements} == {
        "victor",
        "outcome_level",
        "wikidata_id",
    }
    assert "date_start" not in battle.review_notes


def test_an_override_that_settles_the_only_disagreement_clears_needs_review(
    tmp_path: Path,
) -> None:
    battle = _battle()
    battle.failures = []
    battle.disagreements = [Disagreement("victor", "France", "wikipedia", {"dbpedia": "Russia"})]
    assert battle.needs_review

    _apply(tmp_path, [battle], OUTCOME_ENTRY)

    assert not battle.needs_review
    assert battle.review_notes == ""


def test_an_unmatched_override_keeps_its_disagreements(tmp_path: Path) -> None:
    battle = _disputed()
    bad = _entry(3, "outcome", {"battle": "borodino"}, {"Nobody": "victory", "Nemo": "defeat"})

    _apply(tmp_path, [battle], bad)

    assert len(battle.disagreements) == 4


def test_carries_override_flags_only_applied_battles(tmp_path: Path) -> None:
    hit, miss = _battle(), _battle("other")

    _apply(tmp_path, [hit, miss], OUTCOME_ENTRY)

    assert carries_override(hit) is True
    assert carries_override(miss) is False


class _FailingConn:
    """A connection whose savepoint opens but whose statements fail."""

    def begin_nested(self) -> Any:
        return self

    def __enter__(self) -> _FailingConn:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("insert failed")


def test_a_failed_provenance_lookup_leaves_the_source_null_and_the_outcome(
    tmp_path: Path,
) -> None:
    battle = _battle()
    applied, _ = _apply(tmp_path, [battle], OUTCOME_ENTRY)

    extract._attach_outcome_provenance(_FailingConn(), [battle], applied)

    assert all(s.outcome_source_id is None for s in battle.sides)
    assert [s.outcome for s in battle.sides] == ["victory", "defeat"]
