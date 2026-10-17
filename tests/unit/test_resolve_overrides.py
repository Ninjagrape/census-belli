"""
Manual overrides in the resolve stage: ``commander_link`` and ``general_merge``.

The behaviour worth pinning is what a correction must *not* go through: the
lifespan guard, the gap split, the matcher and the LLM. Each test builds the
overrides file the way an editor would, loads it with the strict loader, and
drives ``resolve_groups`` exactly as the stage does. No database, no network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from structlog.testing import capture_logs

from pipeline.config import load_agent_spec
from pipeline.overrides import OverrideReport, load_overrides
from pipeline.resolvers import (
    Candidate,
    Decision,
    LocalCandidateSource,
    Mention,
    MentionGroup,
    ResolveCounts,
    candidate_keys,
    group_mentions,
)
from pipeline.resolvers.disambiguate import prompt_parts
from pipeline.resolvers.overrides import ResolveOverrides
from pipeline.resolvers.records import METHODS, BattleContext
from pipeline.stages.resolve import (
    LazyService,
    _build_identities,
    _log_row,
    _unresolved_mentions,
    resolve_groups,
)
from tests.unit.test_resolve import _entity, _StubService

HANNIBAL = Candidate(qid="Q1", label="Hannibal", birth_year=-247, death_year=-183)
SCIPIO_A = Candidate(qid="Q101", label="Scipio")
SCIPIO_B = Candidate(qid="Q102", label="Scipio")
TARGET = Candidate(qid="Q7", label="Publius Cornelius Scipio Africanus", birth_year=-235)


def _mention(
    name: str, battle: str = "Battle of Zama", year: int | None = -201, side: str = "Rome"
) -> Mention:
    slug = battle.replace(" ", "_")
    return Mention(
        battle_slug=slug,
        battle_name=battle,
        side_label=side,
        name=name,
        context=BattleContext(slug=slug, name=battle, year=year, date_text=str(year)),
    )


class _Source:
    """Serves candidates by folded name, and entities by Q-id."""

    def __init__(self, found: list[Candidate], by_qid: list[Candidate] | None = None) -> None:
        self.found = found
        self.by_qid = {c.qid: c for c in by_qid or []}

    def prefetch(self, groups: Any) -> None:
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        keys = set(group.keys)
        return [c for c in self.found if keys & set(candidate_keys(c))]

    def non_person_kind(self, group: MentionGroup) -> str | None:
        return None

    def candidate_by_qid(self, qid: str) -> Candidate | None:
        return self.by_qid.get(qid)


def _entry(n: int, kind: str, target: dict[str, str], value: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": kind,
        "target": target,
        "value": value,
        "source": "A cited source, p. 1",
        "note": "test",
        "reviewed": {"by": "lee", "on": "2026-10-17"},
        **extra,
    }


def _link(n: int, mention: str, value: str, *, battle: str = "Battle_of_Zama") -> dict[str, Any]:
    return _entry(
        n, "commander_link", {"battle": battle, "side": "Rome", "mention": mention}, value
    )


def _merge(n: int, general: str, value: str) -> dict[str, Any]:
    return _entry(n, "general_merge", {"general": general}, value)


class _Run:
    """The result of one ``resolve_groups`` pass with overrides."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        mentions: list[Mention],
        source: _Source,
        tmp_path: Path,
        service: _StubService | None = None,
    ) -> None:
        path = tmp_path / "overrides.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "entries": entries}), encoding="utf-8")
        overrides = load_overrides(path, required=True)
        self.report = OverrideReport("resolve", overrides)
        self.root = tmp_path
        system, template, schema = prompt_parts(load_agent_spec("resolve"))
        self.service = service or _StubService({"match": "NEW_ENTITY", "confidence": 0.5})
        self.groups = group_mentions(mentions)
        with capture_logs() as self.logs:
            self.decisions = resolve_groups(
                self.groups,
                source,
                LazyService(lambda: self.service),  # type: ignore[arg-type,return-value]
                threshold=85.0,
                margin=6.0,
                counts=ResolveCounts(),
                system=system,
                template=template,
                schema=schema,
                overrides=ResolveOverrides(overrides.for_stage("resolve"), self.report),
            )

    def decision_of(self, name: str, battle: str | None = None) -> Decision:
        for group in self.groups:
            for mention in group.mentions:
                if mention.name == name and (battle is None or mention.battle_name == battle):
                    return self.decisions[group.key]
        raise AssertionError(f"no group holds {name!r}")

    def written(self) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(
            self.report.write(self.root).read_text(encoding="utf-8")
        )
        return payload


def test_manual_is_a_known_method() -> None:
    assert "manual" in METHODS


def test_a_link_to_a_qid_bypasses_the_lifespan_guard_and_the_gap_split(tmp_path: Path) -> None:
    mentions = [
        _mention("Hannibal", "Battle of Zama", -201),
        _mention("Hannibal", "Battle of Lissa", 1811),
    ]
    entry = _link(1, "Hannibal", "Q1", battle="Battle_of_Lissa")

    plain = _Run([], mentions, _Source([HANNIBAL]), tmp_path)
    assert plain.decision_of("Hannibal", "Battle of Lissa").status != "linked"

    run = _Run([entry], mentions, _Source([HANNIBAL], [HANNIBAL]), tmp_path)
    lissa = run.decision_of("Hannibal", "Battle of Lissa")
    zama = run.decision_of("Hannibal", "Battle of Zama")
    assert (lissa.status, lissa.qid, lissa.method, lissa.confidence) == (
        "linked",
        "Q1",
        "manual",
        1.0,
    )
    assert lissa.override_id == "ov-0001"
    assert lissa.candidate is HANNIBAL
    assert zama.qid == "Q1" and zama.method != "manual"
    assert run.written()["applied"]["commander_link"] == 1


def test_the_bypass_is_logged_with_what_the_guard_would_have_done(tmp_path: Path) -> None:
    run = _Run(
        [_link(1, "Hannibal", "Q1", battle="Battle_of_Lissa")],
        [_mention("Hannibal", "Battle of Lissa", 1811)],
        _Source([], [HANNIBAL]),
        tmp_path,
    )
    events = [e for e in run.logs if e["event"] == "resolve_manual_link"]
    assert len(events) == 1
    assert events[0]["id"] == "ov-0001"
    assert events[0]["qid"] == "Q1"
    assert events[0]["battle"] == "Battle_of_Lissa"
    assert events[0]["mention"] == "Hannibal"
    assert events[0]["guard_would_reject"] is True


def test_the_target_matches_after_strip_and_casefold(tmp_path: Path) -> None:
    entry = _entry(
        1,
        "commander_link",
        {"battle": " battle_of_ZAMA ", "side": "rome", "mention": "  HANNIBAL "},
        "Q1",
    )
    run = _Run([entry], [_mention("Hannibal")], _Source([], [HANNIBAL]), tmp_path)
    assert run.decision_of("Hannibal").method == "manual"


def test_a_link_to_none_leaves_the_mention_unresolved(tmp_path: Path) -> None:
    mentions = [_mention("Hannibal"), _mention("Hannibal", "Battle of Cannae", -215)]
    run = _Run([_link(1, "Hannibal", "none")], mentions, _Source([HANNIBAL]), tmp_path)

    decision = run.decision_of("Hannibal", "Battle of Zama")
    assert (decision.status, decision.method, decision.override_id) == (
        "unresolved",
        "manual",
        "ov-0001",
    )
    unresolved = _unresolved_mentions(run.groups, run.decisions)
    assert [(m.battle_name, m.name) for m in unresolved] == [("Battle of Zama", "Hannibal")]
    assert run.decision_of("Hannibal", "Battle of Cannae").qid == "Q1"
    assert _build_identities(run.groups, run.decisions)[0].mentions == [
        m for g in run.groups for m in g.mentions if m.battle_name == "Battle of Cannae"
    ]


def test_a_merge_moves_a_corpus_local_name_to_the_target(tmp_path: Path) -> None:
    mentions = [_mention("Scipio"), _mention("Scipio", "Battle of Ilipa", -205)]
    run = _Run([_merge(1, " scipio ", "Q7")], mentions, _Source([], [TARGET]), tmp_path)

    for decision in run.decisions.values():
        assert (decision.status, decision.qid, decision.method) == ("linked", "Q7", "manual")
        assert decision.canonical_name == TARGET.label
        assert decision.override_id == "ov-0001"
    assert run.written()["applied"]["general_merge"] == 1


def test_a_merge_moves_a_qid_link_to_the_target(tmp_path: Path) -> None:
    run = _Run(
        [_merge(1, "q101", "Q7")],
        [_mention("Scipio")],
        _Source([SCIPIO_A], [TARGET]),
        tmp_path,
    )
    decision = run.decision_of("Scipio")
    assert (decision.qid, decision.method) == ("Q7", "manual")
    assert decision.candidate is TARGET


def test_a_commander_link_beats_a_merge(tmp_path: Path) -> None:
    mentions = [
        _mention("Scipio", "Battle of Zama"),
        _mention("Scipio Maior", "Battle of Ilipa", -205),
    ]
    other = Candidate(qid="Q9", label="Scipio Maior")
    entries = [_link(1, "Scipio", "Q9"), _merge(2, "Q9", "Q7")]
    run = _Run(entries, mentions, _Source([other], [other, TARGET]), tmp_path)

    manual = run.decision_of("Scipio", "Battle of Zama")
    natural = run.decision_of("Scipio Maior", "Battle of Ilipa")
    assert (manual.qid, manual.override_id) == ("Q9", "ov-0001")
    assert (natural.qid, natural.override_id) == ("Q7", "ov-0002")
    assert run.written()["unmatched"] == []


def test_an_overridden_mention_never_reaches_the_llm(tmp_path: Path) -> None:
    service = _StubService({"match": "Q102", "confidence": 0.9})
    mentions = [_mention("Scipio")]
    run = _Run(
        [_link(1, "Scipio", "Q101")],
        mentions,
        _Source([SCIPIO_A, SCIPIO_B], [SCIPIO_A]),
        tmp_path,
        service,
    )
    assert service.calls == []
    assert run.decision_of("Scipio").qid == "Q101"

    control = _StubService({"match": "Q102", "confidence": 0.9})
    _Run([], mentions, _Source([SCIPIO_A, SCIPIO_B]), tmp_path, control)
    assert len(control.calls) == 1


def test_a_manual_identity_is_not_a_corpus_anchor(tmp_path: Path) -> None:
    mentions = [
        _mention("Publius Cornelius Scipio Africanus"),
        _mention("Scipio Africanus", "Battle of Ilipa", -205),
    ]
    run = _Run(
        [_link(1, "Publius Cornelius Scipio Africanus", "Q7")],
        mentions,
        _Source([], [TARGET]),
        tmp_path,
    )
    assert run.decision_of("Publius Cornelius Scipio Africanus").qid == "Q7"
    other = run.decision_of("Scipio Africanus")
    assert (other.status, other.method) == ("new", "new_entity")


def test_a_target_naming_no_mention_is_unmatched_with_a_reason(tmp_path: Path) -> None:
    run = _Run(
        [_link(1, "Nobody", "Q1")], [_mention("Hannibal")], _Source([], [HANNIBAL]), tmp_path
    )
    unmatched = run.written()["unmatched"]
    assert [u["id"] for u in unmatched] == ["ov-0001"]
    assert "Nobody" in unmatched[0]["reason"]


def test_an_unfetchable_link_target_leaves_the_mention_to_the_matcher(tmp_path: Path) -> None:
    run = _Run([_link(1, "Hannibal", "Q1")], [_mention("Hannibal")], _Source([HANNIBAL]), tmp_path)
    assert run.decision_of("Hannibal").method != "manual"
    unmatched = run.written()["unmatched"]
    assert len(unmatched) == 1 and "Q1" in unmatched[0]["reason"]


def test_merges_with_no_group_or_no_target_entity_are_unmatched(tmp_path: Path) -> None:
    entries = [_merge(1, "Q555", "Q7"), _merge(2, "Scipio", "Q8")]
    run = _Run(entries, [_mention("Scipio")], _Source([], [TARGET]), tmp_path)
    reasons = {u["id"]: u["reason"] for u in run.written()["unmatched"]}
    assert set(reasons) == {"ov-0001", "ov-0002"}
    assert "Q555" in reasons["ov-0001"]
    assert "Q8" in reasons["ov-0002"]
    assert run.decision_of("Scipio").method == "new_entity"


def test_every_entry_is_reported_exactly_once(tmp_path: Path) -> None:
    entries = [_link(1, "Hannibal", "Q1"), _link(2, "Nobody", "none"), _merge(3, "Q404", "Q7")]
    run = _Run(entries, [_mention("Hannibal")], _Source([], [HANNIBAL, TARGET]), tmp_path)
    report = run.written()
    assert report["entries"] == 3
    assert sorted(report["entry_ids"]) == ["ov-0001", "ov-0002", "ov-0003"]
    assert sum(report["applied"].values()) + len(report["unmatched"]) == 3


def test_the_resolution_log_row_carries_the_override_id(tmp_path: Path) -> None:
    run = _Run(
        [_link(1, "Hannibal", "Q1")], [_mention("Hannibal")], _Source([], [HANNIBAL]), tmp_path
    )
    group = run.groups[0]
    row = _log_row(group, run.decisions[group.key], None)
    assert (row["method"], row["override_id"], row["wikidata_id"]) == ("manual", "ov-0001", "Q1")

    plain = _Run([], [_mention("Hannibal")], _Source([HANNIBAL]), tmp_path)
    assert "override_id" not in _log_row(
        plain.groups[0], plain.decisions[plain.groups[0].key], None
    )


def test_the_local_source_fetches_an_entity_by_qid(tmp_path: Path) -> None:
    (tmp_path / "Q7.json").write_text(
        json.dumps(_entity("Q7", "Publius Cornelius Scipio Africanus")), encoding="utf-8"
    )
    source = LocalCandidateSource(tmp_path)
    found = source.candidate_by_qid("Q7")
    assert found is not None and found.label == "Publius Cornelius Scipio Africanus"
    assert source.candidate_by_qid("Q8") is None


def _qid_body(*rows: tuple[str, str]) -> str:
    return json.dumps(
        {
            "results": {
                "bindings": [
                    {
                        "person": {"value": f"http://www.wikidata.org/entity/{qid}"},
                        "personLabel": {"value": label},
                        "birth": {"value": "1769-08-15T00:00:00Z"},
                    }
                    for qid, label in rows
                ]
            }
        }
    )


def test_the_sparql_fetch_by_qid_binds_the_item_and_drops_unlabelled_ones() -> None:
    from pipeline.resolvers.candidates import fetch_candidates_by_qid
    from tests.unit.test_resolve import _StubFetcher

    fetcher = _StubFetcher(_qid_body(("Q517", "Napoleon"), ("Q9", "Q9")))
    found = asyncio.run(
        fetch_candidates_by_qid(fetcher, ["q517", "Q9", "Q517"])  # type: ignore[arg-type]
    )

    assert set(found) == {"Q517"}
    assert found["Q517"].birth_year == 1769
    assert len(fetcher.queries) == 1
    assert "VALUES ?person { wd:Q517 wd:Q9 }" in fetcher.queries[0]


def test_a_failed_qid_batch_yields_nothing_rather_than_raising() -> None:
    from pipeline.resolvers.candidates import fetch_candidates_by_qid
    from tests.unit.test_resolve import _StubFetcher

    found = asyncio.run(fetch_candidates_by_qid(_StubFetcher(None), ["Q517"]))  # type: ignore[arg-type]
    assert found == {}
