"""Unit tests for the targeted outcome request (fix R1).

The request is made only for a battle the deterministic merge left without a
winner, is byte-stable, goes to its own batch file and never lets the model
choose the outcome level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.extractors import (
    BattleExtraction,
    BattleFacts,
    Provenance,
    SideExtraction,
    SourceExtraction,
    merge_battle,
)
from pipeline.extractors.outcome import (
    OUTCOME_LEAD_MAX_CHARS,
    OUTCOME_METHOD,
    OUTCOME_STAGE,
    OutcomePass,
    apply_outcome_answer,
    lead_excerpt,
    needs_outcome_request,
    render_outcome_prompt,
    resolve_outcome,
)
from pipeline.extractors.records import CommanderMention, MissingField
from pipeline.llm import CallStatus, LLMRequest, LLMResponse, LLMService
from pipeline.llm.offline import OfflineProvider, build_chunks

_SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string"}, "evidence": {"type": "string"}},
    "required": ["winner", "evidence"],
}
_TEMPLATE = (
    "Battle: {battle_name}\nInfobox result: {infobox_result}\n{sides}\n"
    "Excerpt ({lead_chars}):\n{lead_excerpt}"
)
_INFOBOX = Provenance(source_type="wikipedia_infobox", extraction_method="test", source_ref="t")


class _Recorder:
    """A provider that records every request and answers with a fixed object."""

    name = "anthropic"
    model = "claude-sonnet-5-5"

    def __init__(self, answer: dict[str, str] | None) -> None:
        self.requests: list[LLMRequest] = []
        self._answer = answer

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            status=CallStatus.OK if self._answer else CallStatus.API_ERROR,
            provider=self.name,
            model=self.model,
            request_hash="h",
            data=self._answer,
            error=None if self._answer else "boom",
        )


def _pass(provider: _Recorder) -> OutcomePass:
    return OutcomePass(
        service=LLMService(OUTCOME_STAGE, provider, db_conn=None),
        system="sys",
        template=_TEMPLATE,
        schema=_SCHEMA,
    )


def _commander(name: str, order: int) -> CommanderMention:
    return CommanderMention(
        side_label="Kingdom of Varra", name=name, provenance=_INFOBOX, infobox_order=order
    )


def _undecided(result: str = "Contested") -> BattleExtraction:
    source = SourceExtraction(
        provenance=_INFOBOX,
        facts=BattleFacts(name="Battle of Tests", outcome_evidence=result),
        sides=[
            SideExtraction(
                label="Kingdom of Varra",
                commanders=[_commander("Zed", 1), _commander("Aldric", 0)],
            ),
            SideExtraction(label="Duchy of Selm"),
        ],
    )
    return merge_battle("tests", [source])


def _decided() -> BattleExtraction:
    source = SourceExtraction(
        provenance=_INFOBOX,
        facts=BattleFacts(victor="Varran", outcome_level="victory"),
        sides=[SideExtraction(label="Varra"), SideExtraction(label="Selm")],
    )
    return merge_battle("decided", [source])


def test_no_request_is_made_when_the_deterministic_path_decided() -> None:
    provider = _Recorder({"winner": "Varra", "evidence": "x"})
    merged = _decided()

    assert not needs_outcome_request(merged)
    assert resolve_outcome(merged, _pass(provider), lead="") == "skipped"
    assert provider.requests == []


def test_an_indecisive_battle_is_not_asked_about() -> None:
    source = SourceExtraction(
        provenance=_INFOBOX,
        facts=BattleFacts(outcome_level="indecisive"),
        sides=[SideExtraction(label="A"), SideExtraction(label="B")],
    )

    assert not needs_outcome_request(merge_battle("x", [source]))


def test_a_single_sided_battle_is_not_asked_about() -> None:
    source = SourceExtraction(provenance=_INFOBOX, sides=[SideExtraction(label="A")])

    assert not needs_outcome_request(merge_battle("x", [source]))


def test_an_undecided_battle_makes_one_request_with_the_infobox_text_verbatim() -> None:
    provider = _Recorder({"winner": "unclear", "evidence": ""})
    merged = _undecided("Contested; see Aftermath")

    assert needs_outcome_request(merged)
    assert resolve_outcome(merged, _pass(provider), lead="The lead.") == "unclear"

    (request,) = provider.requests
    assert "Contested; see Aftermath" in request.user
    assert request.user.index("Side 1: Kingdom of Varra") < request.user.index("Side 2")
    assert request.user.index("Aldric") < request.user.index("Zed")
    assert "The lead." in request.user


def test_the_prompt_is_byte_stable() -> None:
    first = render_outcome_prompt(_TEMPLATE, _undecided(), "lead", 10)
    second = render_outcome_prompt(_TEMPLATE, _undecided(), "lead", 10)

    assert first == second


def test_the_lead_is_capped_and_stops_at_the_first_heading() -> None:
    wikitext = "The battle began.\n" + "word " * 2000 + "\n\n== Later ==\n\nSecond section."

    excerpt = lead_excerpt(wikitext, is_html=False, max_chars=OUTCOME_LEAD_MAX_CHARS)

    assert 0 < len(excerpt) <= OUTCOME_LEAD_MAX_CHARS
    assert "Second section" not in excerpt


def test_a_named_winner_gets_the_level_the_result_text_grades() -> None:
    merged = _undecided("Decisive Varran victory")

    assert apply_outcome_answer(merged, {"winner": "Kingdom of Varra", "evidence": "held"})

    outcomes = {s.label: s.outcome for s in merged.sides}
    assert outcomes == {"Kingdom of Varra": "decisive_victory", "Duchy of Selm": "decisive_defeat"}
    assert merged.outcome_method == OUTCOME_METHOD
    assert merged.facts.outcome_level == "decisive_victory"


def test_an_ungraded_result_gives_a_plain_victory() -> None:
    merged = _undecided("Contested")

    assert apply_outcome_answer(merged, {"winner": "Duchy of Selm", "evidence": "q"})

    assert {s.label: s.outcome for s in merged.sides}["Duchy of Selm"] == "victory"


@pytest.mark.parametrize("winner", ["unclear", "Varra", "", "Selm"])
def test_unclear_or_unknown_winners_write_nothing(winner: str) -> None:
    merged = _undecided()

    assert not apply_outcome_answer(merged, {"winner": winner, "evidence": ""})

    assert all(s.outcome is None for s in merged.sides)
    assert merged.outcome_method == ""


def test_an_indecisive_answer_marks_every_side() -> None:
    merged = _undecided()

    assert apply_outcome_answer(merged, {"winner": "indecisive", "evidence": "draw"})

    assert {s.outcome for s in merged.sides} == {"indecisive"}


@pytest.mark.parametrize(
    "winner",
    [
        "Side 1: Kingdom of Varra",
        "side 1 - Kingdom of Varra",
        "  kingdom of varra  ",
        "Side 1:Kingdom of Varra",
    ],
)
def test_the_winner_is_normalised_before_matching_a_side_label(winner: str) -> None:
    merged = _undecided()

    assert apply_outcome_answer(merged, {"winner": winner, "evidence": "e"})

    assert {s.label: s.outcome for s in merged.sides}["Kingdom of Varra"] == "victory"


def test_a_partial_label_is_still_not_matched() -> None:
    merged = _undecided()

    assert not apply_outcome_answer(merged, {"winner": "Side 1: Kingdom", "evidence": ""})


@pytest.mark.parametrize("level", ["victory", "decisive_victory", "pyrrhic_victory"])
def test_an_indecisive_answer_is_rejected_when_the_sources_graded_a_victory(level: str) -> None:
    merged = _undecided()
    merged.facts.outcome_level = level

    assert not apply_outcome_answer(merged, {"winner": "indecisive", "evidence": "draw"})

    assert all(s.outcome is None for s in merged.sides)
    assert merged.outcome_method == ""
    assert merged.facts.outcome_level == level


def test_an_applied_answer_retracts_the_outcome_gaps() -> None:
    merged = _undecided()
    merged.missing = [
        MissingField(field_name="outcome", side_label="Kingdom of Varra"),
        MissingField(field_name="outcome", side_label="Duchy of Selm"),
        MissingField(field_name="troop_total", side_label="Duchy of Selm"),
    ]
    provider = _Recorder({"winner": "Kingdom of Varra", "evidence": "held the field"})

    assert resolve_outcome(merged, _pass(provider), lead="") == "applied"

    assert [m.field_name for m in merged.missing] == ["troop_total"]


def test_a_failed_request_flags_the_battle_and_keeps_the_gaps() -> None:
    merged = _undecided()
    merged.missing = [MissingField(field_name="outcome", side_label="Duchy of Selm")]

    assert resolve_outcome(merged, _pass(_Recorder(None)), lead="") == "failed"

    assert merged.failures and merged.missing


def test_outcome_requests_go_to_their_own_sonnet_batch_file(tmp_path: Path) -> None:
    provider = OfflineProvider("anthropic", "claude-sonnet-5-5", output_dir=tmp_path)
    service = LLMService(OUTCOME_STAGE, provider, db_conn=None)
    outcome_pass = OutcomePass(service, "sys", _TEMPLATE, _SCHEMA)
    for slug in ("one", "two"):
        merged = _undecided()
        merged.name = f"Battle {slug}"
        assert resolve_outcome(merged, outcome_pass, lead="") == "deferred"
        assert merged.deferred_failures

    path = service.flush_pending()

    assert path is not None and path.parent == tmp_path
    assert path.name.startswith("extract_outcome_")
    lines = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()]
    assert {line["stage"] for line in lines} == {OUTCOME_STAGE}
    assert {line["agent_model"] for line in lines} == {"sonnet"}
    chunks = build_chunks(path, chunk_size=15)
    assert len(chunks) == 1 and chunks[0]["agent_model"] == "sonnet"
    assert len(chunks[0]["requests"]) == 2
