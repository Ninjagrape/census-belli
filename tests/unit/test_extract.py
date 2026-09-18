"""
Unit tests for the extract stage.

No network, database or API key is used anywhere in this module. The LLM is
a fake provider driven through the real :class:`~pipeline.llm.LLMService`,
so the cache, schema validation and failure classification under test are
the production ones rather than doubles of them.

The fixtures are real-shaped battle articles chosen to cover the infobox
variants the parser has to tell apart: Austerlitz for a land conflict with a
coalition on one side, Actium for a naval one where the strength fields
carry mixed scope, Alesia for a siege, where the template names the besieged
force differently and does not say which direction the siege ran, and Cannae
as rendered HTML, which is the form the crawl stage actually stores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.crawlers.wikipedia import article_filename
from pipeline.extractors import (
    BattleFacts,
    Provenance,
    SideExtraction,
    SourceExtraction,
    TroopReport,
    build_passages,
    campaign_names,
    clean_article_text,
    detect_variant,
    infer_scope,
    iter_batches,
    map_entity,
    merge_battle,
    parse_infobox,
    parse_infobox_html,
    parse_quantities,
    render_template,
    resolve_scope,
    wikidata_time_to_date,
)
from pipeline.extractors.article import extract_from_passages
from pipeline.llm.base import CallStatus, LLMRequest, LLMResponse, TokenUsage
from pipeline.llm.parsing import parse_structured
from pipeline.llm.service import LLMService
from pipeline.stages import extract as extract_stage
from pipeline.stages.base import StageContext, check_conforms

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"


def _fixture(name: str) -> str:
    """Read one extract fixture.

    Args:
        name: File name inside tests/fixtures/extract.

    Returns:
        The file's text.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    """Load the real agents/extract.yaml.

    Returns:
        The spec as the stage sees it.
    """
    from pipeline.config import load_agent_spec

    return load_agent_spec("extract")


# ─── Fake LLM provider ───────────────────────────────────────────────────────


class ScriptedProvider:
    """A provider replaying canned raw responses through the real parser.

    Mirrors what the vendor clients do: a response is parsed and validated
    against the request's schema, and only a valid object comes back as OK.
    That way a malformed payload in a test produces the same PARSE_ERROR the
    pipeline would see in production.
    """

    name = "fake"
    model = "fake-extract-1"

    def __init__(self, responses: list[str | CallStatus]) -> None:
        self.requests: list[LLMRequest] = []
        self._responses = list(responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next scripted response.

        Args:
            request: The request being sent.

        Returns:
            An OK response for a schema-valid payload, otherwise a failure.
        """
        self.requests.append(request)
        scripted = self._responses.pop(0) if self._responses else "{}"
        usage = TokenUsage(input_tokens=900, output_tokens=120)

        if isinstance(scripted, CallStatus):
            return LLMResponse(
                status=scripted,
                provider=self.name,
                model=self.model,
                request_hash="fakehash",
                usage=usage,
                error=f"provider returned {scripted.value}",
            )

        data, error = parse_structured(scripted, request.json_schema)
        if data is None:
            return LLMResponse(
                status=CallStatus.PARSE_ERROR,
                provider=self.name,
                model=self.model,
                request_hash="fakehash",
                raw_text=scripted,
                usage=usage,
                error=error,
            )
        return LLMResponse(
            status=CallStatus.OK,
            provider=self.name,
            model=self.model,
            request_hash="fakehash",
            data=data,
            raw_text=scripted,
            usage=usage,
        )


def _service(responses: list[str | CallStatus]) -> LLMService:
    """Build an LLMService over a scripted provider with no database.

    Args:
        responses: Raw payloads or statuses, consumed in order.

    Returns:
        A service safe to use with no connection and no API key.
    """
    return LLMService(stage="extract", provider=ScriptedProvider(responses), db_conn=None)


_GOOD_PAYLOAD = json.dumps(
    {
        "battle_name": "Battle of Austerlitz",
        "commanders": [
            {
                "name": "Napoleon I",
                "side": "French Empire",
                "apparent_role": "field_commander",
                "role_evidence": "Napoleon directed the battle in person",
            }
        ],
        "troop_reports": [
            {
                "side": "Russian Empire",
                "branch": "total",
                "value": 110000,
                "scope": "engaged",
                "is_estimate": True,
                "context_quote": "Kutuzov could call upon some 110,000 men in the theatre",
            }
        ],
        "casualty_reports": [
            {"side": "French Empire", "casualty_type": "killed", "value": 1305}
        ],
        "terrain": ["ridge", "fog"],
        "outcome": {"victor": "French Empire", "outcome_level": "decisive_victory"},
    }
)


# ─── textnorm: quantities, bounds and scope ──────────────────────────────────


def test_a_range_becomes_a_lower_and_an_upper_bound() -> None:
    quantities = parse_quantities("between 15,000 and 20,000 engaged")

    assert [q.value for q in quantities] == [15000.0, 20000.0]
    assert quantities[0].is_lower_bound is True and quantities[0].is_upper_bound is False
    assert quantities[1].is_upper_bound is True and quantities[1].is_lower_bound is False


def test_a_dash_range_is_read_as_a_range() -> None:
    quantities = parse_quantities("65,000-75,000 men")

    assert [q.value for q in quantities] == [65000.0, 75000.0]


def test_approximation_wording_sets_is_estimate() -> None:
    (quantity,) = parse_quantities("about 30,000 men")

    assert quantity.is_estimate is True


def test_at_least_and_up_to_set_single_bounds() -> None:
    (lower,) = parse_quantities("at least 4,000 cavalry")
    (upper,) = parse_quantities("up to 9,000 infantry")

    assert lower.is_lower_bound is True and lower.is_upper_bound is False
    assert upper.is_upper_bound is True and upper.is_lower_bound is False


def test_scaled_and_grouped_numbers_are_read() -> None:
    assert parse_quantities("1.5 million men")[0].value == 1_500_000.0
    assert parse_quantities("12,500 men")[0].value == 12500.0


def test_formation_counts_are_not_troop_numbers() -> None:
    """"12 legions" counts units, not men, and must not reach troop_reports."""
    quantities = parse_quantities("12 legions, around 60,000 men engaged")

    assert [q.value for q in quantities] == [60000.0]


def test_years_are_not_mistaken_for_troop_counts() -> None:
    assert parse_quantities("fought in 1805 near the village") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30,000 men were engaged in the battle", "engaged"),
        ("Napoleon had 110,000 men in the theatre", "theatre_strength"),
        ("the establishment strength was 40,000", "on_paper"),
        ("he could call upon 60,000 men", "available"),
        ("the army numbered 20,000", "unknown"),
    ],
)
def test_scope_is_read_from_the_wording(text: str, expected: str) -> None:
    assert infer_scope(text) == expected


def test_scope_is_not_defaulted_to_engaged_without_evidence() -> None:
    """Silence about scope is not evidence that troops were on the field."""
    (quantity,) = parse_quantities("the army numbered 20,000")

    assert quantity.scope == "unknown"


def test_wording_overrides_a_declared_scope() -> None:
    """A model's label loses to the sentence it quoted."""
    settled = resolve_scope("engaged", "40,000 men were available in the theatre")

    assert settled == "theatre_strength"


def test_a_declared_scope_survives_silent_wording() -> None:
    assert resolve_scope("on_paper", "the army numbered 40,000") == "on_paper"


def test_an_unknown_declared_scope_falls_back_to_the_callers_default() -> None:
    assert resolve_scope(None, "the army numbered 40,000", "engaged") == "engaged"


# ─── Infobox parsing ─────────────────────────────────────────────────────────


def test_land_infobox_yields_two_sides_with_commanders_and_reports() -> None:
    extraction = parse_infobox(_fixture("austerlitz.wikitext"), source_ref="f", source_title="t")

    assert extraction is not None
    assert [s.label for s in extraction.sides] == ["French Empire", "Russian Empire"]
    # The generic template says nothing about domain, so battle_type is left
    # for classify. See test_generic_template_does_not_assert_a_land_battle.
    assert extraction.facts.battle_type is None
    assert extraction.facts.outcome_level == "decisive_victory"
    assert extraction.facts.victor == "French"

    french, coalition = extraction.sides
    assert [c.name for c in french.commanders] == [
        "Napoleon I",
        "Jean Lannes",
        "Louis-Nicolas Davout",
    ]
    assert coalition.aliases == ["Austrian Empire"]
    assert any(c.note == "KIA" for c in coalition.commanders)


def test_infobox_strength_ranges_become_two_bounded_reports() -> None:
    extraction = parse_infobox(_fixture("austerlitz.wikitext"))

    assert extraction is not None
    french = extraction.sides[0]
    assert [r.reported_value for r in french.troop_reports] == [65000.0, 75000.0]
    assert french.troop_reports[0].is_lower_bound is True
    assert french.troop_reports[1].is_upper_bound is True
    assert all(r.page_or_section == "infobox:strength1" for r in french.troop_reports)


def test_casualty_types_are_read_per_line() -> None:
    extraction = parse_infobox(_fixture("austerlitz.wikitext"))

    assert extraction is not None
    types = [(c.casualty_type, c.reported_value) for c in extraction.sides[0].casualty_reports]
    assert types == [("killed", 1305.0), ("wounded", 6940.0), ("captured", 573.0)]


def test_killed_or_wounded_is_not_claimed_for_either_bucket() -> None:
    extraction = parse_infobox(_fixture("austerlitz.wikitext"))

    assert extraction is not None
    coalition = extraction.sides[1]
    assert coalition.casualty_reports[0].casualty_type == "total"
    assert coalition.casualty_reports[0].is_estimate is True


def test_naval_variant_maps_its_own_strength_field_to_the_naval_branch() -> None:
    extraction = parse_infobox(_fixture("actium_naval.wikitext"))

    assert extraction is not None
    assert extraction.facts.battle_type == "naval"
    ships = [r for r in extraction.sides[0].troop_reports if r.branch == "naval"]
    assert [r.reported_value for r in ships] == [400.0]


def test_naval_infobox_preserves_differing_scopes_within_one_field() -> None:
    """Two figures in one field can mean different things; both are kept."""
    extraction = parse_infobox(_fixture("actium_naval.wikitext"))

    assert extraction is not None
    antony = extraction.sides[1]
    by_scope = {(r.reported_value, r.scope) for r in antony.troop_reports}
    assert (120000.0, "on_paper") in by_scope
    assert (20000.0, "engaged") in by_scope


def test_generic_template_does_not_assert_a_land_battle() -> None:
    """The universal template must not label naval battles as land engagements.

    Wikipedia uses {{Infobox military conflict}} for almost everything: Actium,
    Trafalgar and Midway all use the plain template rather than a naval one,
    and none of them carries a ships= or vessels= field to fall back on. An
    earlier version mapped that template to a "land" variant asserting
    battle_type="field", which labelled every naval battle a land engagement.

    battle_type is a covariate in agents/model.yaml, so that error fell
    precisely on the commanders whose ranking depends on it.
    """
    naval_article_using_the_generic_template = """
    {{Infobox military conflict
    | conflict = Battle of Actium
    | partof = the Final War of the Roman Republic
    | date = 2 September 31 BC
    | place = Ionian Sea, near Actium
    | result = Octavian victory
    | combatant1 = Octavian's forces
    | combatant2 = Forces of Antony and Cleopatra
    | commander1 = [[Marcus Vipsanius Agrippa|Agrippa]]
    | commander2 = [[Mark Antony]]
    | strength1 = 400 ships, 19,000 marines
    | strength2 = 290 ships, 20,000 marines
    }}
    """

    extraction = parse_infobox(
        naval_article_using_the_generic_template, source_ref="f", source_title="t"
    )

    assert extraction is not None
    assert extraction.facts.battle_type is None, (
        "the generic template carries no evidence of domain; guessing 'field' "
        "mislabels every naval battle"
    )
    assert extraction.notes and "battle_type" in extraction.notes[0]


def test_siege_variant_reads_its_garrison_field_and_leaves_battle_type_unset() -> None:
    extraction = parse_infobox(_fixture("alesia_siege.wikitext"))

    assert extraction is not None
    # The template says a siege happened but not from whose side, and the
    # schema distinguishes offensive from defensive.
    assert extraction.facts.battle_type is None
    assert extraction.notes and "battle_type" in extraction.notes[0]

    gauls = extraction.sides[1]
    garrison = [r for r in gauls.troop_reports if r.page_or_section == "infobox:garrison2"]
    assert [r.reported_value for r in garrison] == [80000.0]
    assert extraction.facts.fortified is True


def test_siege_relief_force_range_is_recorded_as_available_not_engaged() -> None:
    extraction = parse_infobox(_fixture("alesia_siege.wikitext"))

    assert extraction is not None
    relief = [
        r
        for r in extraction.sides[1].troop_reports
        if r.page_or_section == "infobox:strength2"
    ]
    assert {r.scope for r in relief} == {"available"}


def test_campaignboxes_are_read() -> None:
    assert campaign_names(_fixture("austerlitz.wikitext")) == ["War of the Third Coalition"]


def test_partof_and_campaignbox_are_not_duplicated() -> None:
    extraction = parse_infobox(_fixture("austerlitz.wikitext"))

    assert extraction is not None
    assert extraction.facts.part_of == ["War of the Third Coalition"]


def test_variant_detection_covers_the_families_and_rejects_others() -> None:
    assert detect_variant("Infobox military conflict") is not None
    assert detect_variant("Infobox naval battle") is not None
    assert detect_variant("Infobox siege") is not None
    assert detect_variant("Infobox person") is None


def test_an_article_without_a_conflict_infobox_returns_none() -> None:
    assert parse_infobox("{{Infobox person\n| name = Someone\n}}\n\nProse.") is None


# ─── Rendered infoboxes, which is what the crawl stage actually stores ───────


def test_rendered_infobox_reads_side_indexed_rows() -> None:
    """A rendered infobox states a quantity as a header row plus one cell per side."""
    extraction = parse_infobox_html(
        _fixture("cannae_rendered.html"), source_ref="f", source_title="Battle of Cannae"
    )

    assert extraction is not None
    assert [s.label for s in extraction.sides] == ["Roman Republic", "Carthage"]
    assert extraction.facts.part_of == ["Second Punic War"]
    assert extraction.facts.victor == "Carthaginian"
    assert extraction.facts.outcome_level == "decisive_victory"


def test_rendered_infobox_splits_commanders_on_line_breaks() -> None:
    extraction = parse_infobox_html(_fixture("cannae_rendered.html"))

    assert extraction is not None
    romans = extraction.sides[0]
    assert [c.name for c in romans.commanders] == [
        "Lucius Aemilius Paullus",
        "Gaius Terentius Varro",
    ]
    assert romans.commanders[0].note == "KIA"


def test_rendered_infobox_drops_reference_superscripts_from_numbers() -> None:
    extraction = parse_infobox_html(_fixture("cannae_rendered.html"))

    assert extraction is not None
    romans = extraction.sides[0]
    assert [r.reported_value for r in romans.troop_reports] == [86000.0]
    assert romans.troop_reports[0].scope == "engaged"


def test_rendered_infobox_leaves_battle_type_for_a_later_stage() -> None:
    """Rendered HTML carries row labels, not a template name."""
    extraction = parse_infobox_html(_fixture("cannae_rendered.html"))

    assert extraction is not None
    assert extraction.facts.battle_type is None
    assert extraction.notes and "battle_type" in extraction.notes[0]


def test_a_rendered_page_without_an_infobox_returns_none() -> None:
    assert parse_infobox_html("<html><body><p>Prose only.</p></body></html>") is None


# ─── Article cleaning and chunking ───────────────────────────────────────────


def test_html_cleaning_keeps_headings_and_drops_scripts_and_references() -> None:
    text = clean_article_text(_fixture("austerlitz.html"))

    assert "== Prelude ==" in text
    assert "tracking" not in text
    assert "[1]" not in text


def test_wikitext_cleaning_keeps_headings_and_drops_templates() -> None:
    text = clean_article_text(_fixture("austerlitz.wikitext"))

    assert "== Prelude ==" in text
    assert "Infobox" not in text
    assert "The Battle of Austerlitz" in text


def test_passages_skip_reference_sections() -> None:
    passages = build_passages("Battle of Austerlitz", _fixture("austerlitz.wikitext"))

    assert [p.section for p in passages] == ["lead", "Prelude", "The battle"]
    assert all(p.total == len(passages) for p in passages)


def test_long_sections_are_chunked_under_the_limit() -> None:
    body = "\n\n".join(f"Paragraph {i} about the battle." * 6 for i in range(20))
    passages = build_passages("Test", f"== Body ==\n\n{body}", max_chars=400, is_html=False)

    assert len(passages) > 1
    assert all(len(p.text) <= 400 for p in passages)


def test_iter_batches_honours_the_batch_size() -> None:
    assert [len(b) for b in iter_batches(list(range(7)), 3)] == [3, 3, 1]
    assert [len(b) for b in iter_batches(list(range(3)), 0)] == [1, 1, 1]


def test_render_template_fills_placeholders_and_tolerates_unknown_ones() -> None:
    rendered = render_template(
        "Battle: {battle_name}\n{text_passage}\n{mystery}",
        {"battle_name": "Actium", "text_passage": "A {brace} in the text"},
    )

    assert "Actium" in rendered
    assert "A {brace} in the text" in rendered
    assert "{mystery}" in rendered


# ─── Wikidata mapping ────────────────────────────────────────────────────────


def test_wikidata_entity_maps_onto_schema_columns() -> None:
    payload = json.loads(_fixture("actium_wikidata.json"))
    extraction = map_entity(payload, source_ref="wd")

    assert extraction is not None
    facts = extraction.facts
    assert facts.wikidata_id == "Q193320"
    assert facts.latitude == pytest.approx(38.9333)
    assert facts.wikipedia_url == "https://en.wikipedia.org/wiki/Battle_of_Actium"
    assert facts.date_precision == "day"


def test_bc_dates_survive_as_a_postgres_literal() -> None:
    """datetime.date cannot hold 31 BC; a quarter of the corpus is BC."""
    payload = json.loads(_fixture("actium_wikidata.json"))
    extraction = map_entity(payload, source_ref="wd")

    assert extraction is not None
    assert extraction.facts.date_start == "0031-09-02 BC"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"time": "+1805-12-02T00:00:00Z", "precision": 11}, ("1805-12-02", "day")),
        ({"time": "-0052-00-00T00:00:00Z", "precision": 9}, ("0052-01-01 BC", "year")),
        ({"time": "+1805-09-00T00:00:00Z", "precision": 10}, ("1805-09-01", "month")),
        ({"time": "nonsense", "precision": 11}, (None, None)),
    ],
)
def test_wikidata_time_conversion(value: dict[str, Any], expected: tuple[Any, Any]) -> None:
    assert wikidata_time_to_date(value) == expected


def test_wikidata_participants_are_noted_not_invented_into_sides() -> None:
    payload = json.loads(_fixture("actium_wikidata.json"))
    extraction = map_entity(payload, source_ref="wd")

    assert extraction is not None
    assert extraction.sides == []
    assert any("participants" in note for note in extraction.notes)


def test_a_payload_without_an_entity_returns_none() -> None:
    assert map_entity({"entities": {}}, source_ref="wd") is None


# ─── The LLM pass ────────────────────────────────────────────────────────────


def test_well_formed_llm_json_becomes_records(spec: dict[str, Any]) -> None:
    service = _service([_GOOD_PAYLOAD])
    passages = build_passages("Battle of Austerlitz", "== Body ==\n\nSome prose.")

    result = extract_from_passages(
        service,
        passages,
        system_prompt=spec["prompt"]["system"],
        user_template=spec["prompt"]["user_template"],
        json_schema=spec["prompt"]["output_schema"],
        source_ref="article",
    )

    assert result.failures == []
    (extraction,) = result.extractions
    labels = {side.label for side in extraction.sides}
    assert labels == {"French Empire", "Russian Empire"}
    assert extraction.facts.outcome_level == "decisive_victory"


def test_the_prompt_sent_is_the_one_in_the_spec(spec: dict[str, Any]) -> None:
    """The stage must load the prompt from the spec, never carry a copy."""
    provider = ScriptedProvider([_GOOD_PAYLOAD])
    service = LLMService(stage="extract", provider=provider, db_conn=None)

    extract_from_passages(
        service,
        build_passages("Battle of Actium", "== Body ==\n\nProse about the fleet."),
        system_prompt=spec["prompt"]["system"],
        user_template=spec["prompt"]["user_template"],
        json_schema=spec["prompt"]["output_schema"],
    )

    (request,) = provider.requests
    assert request.system == spec["prompt"]["system"]
    assert "Battle of Actium" in request.user
    assert "Prose about the fleet." in request.user


def test_llm_scope_is_overridden_by_the_quote_it_came_from(spec: dict[str, Any]) -> None:
    """The fixture payload claims 'engaged' over a quote saying 'theatre'."""
    service = _service([_GOOD_PAYLOAD])

    result = extract_from_passages(
        service,
        build_passages("Battle of Austerlitz", "== Body ==\n\nProse."),
        system_prompt=spec["prompt"]["system"],
        user_template=spec["prompt"]["user_template"],
        json_schema=spec["prompt"]["output_schema"],
    )

    (extraction,) = result.extractions
    russians = next(s for s in extraction.sides if s.label == "Russian Empire")
    assert russians.troop_reports[0].scope == "theatre_strength"


def test_malformed_json_marks_for_review_without_raising(spec: dict[str, Any]) -> None:
    service = _service(["{not json at all"])

    result = extract_from_passages(
        service,
        build_passages("Battle of Austerlitz", "== Body ==\n\nProse."),
        system_prompt=spec["prompt"]["system"],
        user_template=spec["prompt"]["user_template"],
        json_schema=spec["prompt"]["output_schema"],
        source_ref="article",
    )

    assert result.extractions == []
    (failure,) = result.failures
    assert failure.status == "parse_error"
    assert failure.error


def test_schema_violating_json_marks_for_review(spec: dict[str, Any]) -> None:
    """Valid JSON that does not match the spec's schema is still a failure."""
    service = _service([json.dumps({"commanders": []})])

    result = extract_from_passages(
        service,
        build_passages("Battle of Austerlitz", "== Body ==\n\nProse."),
        system_prompt=spec["prompt"]["system"],
        user_template=spec["prompt"]["user_template"],
        json_schema=spec["prompt"]["output_schema"],
    )

    assert result.extractions == []
    assert result.failures[0].status == "parse_error"


def test_a_refusal_marks_for_review_without_raising(spec: dict[str, Any]) -> None:
    service = _service([CallStatus.REFUSAL])

    result = extract_from_passages(
        service,
        build_passages("Battle of Austerlitz", "== Body ==\n\nProse."),
        system_prompt=spec["prompt"]["system"],
        user_template=spec["prompt"]["user_template"],
        json_schema=spec["prompt"]["output_schema"],
    )

    assert result.extractions == []
    assert result.failures[0].status == "refusal"


def test_one_failed_passage_does_not_stop_the_others(spec: dict[str, Any]) -> None:
    service = _service(["{broken", _GOOD_PAYLOAD, CallStatus.API_ERROR])
    passages = build_passages(
        "Battle of Austerlitz",
        "== One ==\n\nFirst.\n\n== Two ==\n\nSecond.\n\n== Three ==\n\nThird.",
    )

    result = extract_from_passages(
        service,
        passages,
        system_prompt=spec["prompt"]["system"],
        user_template=spec["prompt"]["user_template"],
        json_schema=spec["prompt"]["output_schema"],
    )

    assert len(result.extractions) == 1
    assert {f.status for f in result.failures} == {"parse_error", "api_error"}


# ─── Merging ─────────────────────────────────────────────────────────────────


def _source(
    source_type: str,
    *,
    facts: BattleFacts | None = None,
    sides: list[SideExtraction] | None = None,
    ref: str = "",
) -> SourceExtraction:
    """Build a source extraction for merge tests.

    Args:
        source_type: A source_type enum value.
        facts: Battle-level scalars this source reports.
        sides: The sides this source reports.
        ref: A source reference, distinguishing documents of the same type.

    Returns:
        The assembled extraction.
    """
    return SourceExtraction(
        provenance=Provenance(
            source_type=source_type,
            extraction_method="infobox_parser",
            source_ref=ref or source_type,
        ),
        facts=facts or BattleFacts(),
        sides=sides or [],
    )


def _troops(value: float, source_type: str, ref: str, scope: str = "engaged") -> TroopReport:
    """Build a troop report for merge tests.

    Args:
        value: The reported number.
        source_type: The reporting source's type.
        ref: The reporting document.
        scope: What the number refers to.

    Returns:
        The assembled report.
    """
    return TroopReport(
        side_label="French Empire",
        reported_value=value,
        provenance=Provenance(
            source_type=source_type, extraction_method="llm_extraction", source_ref=ref
        ),
        scope=scope,
    )


def test_structured_sources_win_scalar_fields_over_llm_extraction() -> None:
    merged = merge_battle(
        "actium",
        [
            _source("wikipedia_body", facts=BattleFacts(date_start="3 September 31 BC")),
            _source("wikidata", facts=BattleFacts(date_start="0031-09-02 BC")),
        ],
    )

    assert merged.facts.date_start == "0031-09-02 BC"


def test_a_scalar_disagreement_is_recorded_rather_than_hidden() -> None:
    merged = merge_battle(
        "actium",
        [
            _source("wikipedia_body", facts=BattleFacts(date_start="3 September 31 BC")),
            _source("wikidata", facts=BattleFacts(date_start="0031-09-02 BC")),
        ],
    )

    (disagreement,) = merged.disagreements
    assert disagreement.field_name == "date_start"
    assert disagreement.chosen_source == "wikidata"
    assert disagreement.rejected == {"wikipedia_body": "3 September 31 BC"}
    assert merged.needs_review is True
    assert "date_start" in merged.review_notes


def test_every_conflicting_troop_number_survives_as_its_own_report() -> None:
    """The reconcile stage models source disagreement; it needs every report."""
    merged = merge_battle(
        "austerlitz",
        [
            _source(
                "wikipedia_infobox",
                sides=[
                    SideExtraction(
                        label="French Empire",
                        troop_reports=[_troops(65000, "wikipedia_infobox", "infobox")],
                    )
                ],
            ),
            _source(
                "wikipedia_body",
                sides=[
                    SideExtraction(
                        label="French Empire",
                        troop_reports=[_troops(72000, "wikipedia_body", "body")],
                    )
                ],
            ),
            _source(
                "web_secondary",
                ref="chandler",
                sides=[
                    SideExtraction(
                        label="French Empire",
                        troop_reports=[_troops(65000, "web_secondary", "chandler")],
                    )
                ],
            ),
        ],
    )

    (side,) = merged.sides
    # Three reports, including two that agree: agreement between independent
    # sources is itself evidence and must not be collapsed.
    assert sorted(r.reported_value for r in side.troop_reports) == [65000.0, 65000.0, 72000.0]


def test_the_same_claim_repeated_in_one_document_is_counted_once() -> None:
    merged = merge_battle(
        "austerlitz",
        [
            _source(
                "wikipedia_body",
                sides=[
                    SideExtraction(
                        label="French Empire",
                        troop_reports=[
                            _troops(65000, "wikipedia_body", "body"),
                            _troops(65000, "wikipedia_body", "body"),
                        ],
                    )
                ],
            )
        ],
    )

    assert len(merged.sides[0].troop_reports) == 1


def test_sides_are_matched_across_sources_by_label() -> None:
    """Common polity suffixes are stripped so one side does not become two."""
    merged = merge_battle(
        "austerlitz",
        [
            _source("wikipedia_infobox", sides=[SideExtraction(label="French Empire")]),
            _source("wikipedia_body", sides=[SideExtraction(label="the French forces")]),
        ],
    )

    assert len(merged.sides) == 1
    assert "the French forces" in merged.sides[0].aliases


def test_unrelated_sides_are_not_merged_together() -> None:
    """Matching is deliberately conservative: merging real belligerents is worse."""
    merged = merge_battle(
        "austerlitz",
        [
            _source("wikipedia_infobox", sides=[SideExtraction(label="French Empire")]),
            _source("wikipedia_body", sides=[SideExtraction(label="Russian Empire")]),
        ],
    )

    assert len(merged.sides) == 2


def test_a_stated_role_replaces_unclear_when_the_same_person_recurs() -> None:
    infobox_side = SideExtraction(label="Octavian")
    body_side = SideExtraction(label="Octavian")
    from pipeline.extractors import CommanderMention

    provenance = Provenance(source_type="wikipedia_infobox", extraction_method="infobox_parser")
    infobox_side.commanders.append(
        CommanderMention(side_label="Octavian", name="Agrippa", provenance=provenance)
    )
    body_side.commanders.append(
        CommanderMention(
            side_label="Octavian",
            name="Agrippa",
            provenance=Provenance(
                source_type="wikipedia_body", extraction_method="llm_extraction"
            ),
            apparent_role="field_commander",
            role_evidence="Agrippa commanded the fleet",
        )
    )

    merged = merge_battle(
        "actium",
        [
            _source("wikipedia_infobox", sides=[infobox_side]),
            _source("wikipedia_body", sides=[body_side]),
        ],
    )

    (commander,) = merged.sides[0].commanders
    assert commander.apparent_role == "field_commander"
    assert commander.role_evidence == "Agrippa commanded the fleet"
    assert commander.mentions == 2


def test_outcome_is_mirrored_onto_the_losing_side() -> None:
    extraction = parse_infobox(_fixture("austerlitz.wikitext"))
    assert extraction is not None

    merged = merge_battle("austerlitz", [extraction])

    outcomes = {s.label: s.outcome for s in merged.sides}
    assert outcomes["French Empire"] == "decisive_victory"
    assert outcomes["Russian Empire"] == "decisive_defeat"


def test_an_ambiguous_victor_label_leaves_outcomes_unset() -> None:
    merged = merge_battle(
        "somewhere",
        [
            _source(
                "wikipedia_infobox",
                facts=BattleFacts(victor="Allied", outcome_level="victory"),
                sides=[
                    SideExtraction(label="Allied Powers"),
                    SideExtraction(label="Allied Expeditionary Corps"),
                ],
            )
        ],
    )

    assert all(side.outcome is None for side in merged.sides)


def test_missing_fields_are_logged_not_guessed() -> None:
    merged = merge_battle("obscure", [_source("wikipedia_infobox", facts=BattleFacts(name="X"))])

    missing = {m.field_name for m in merged.missing}
    assert {"date_start", "location_name", "outcome_level", "battle_sides"} <= missing


def test_all_unknown_scope_is_flagged_as_unusable_for_a_force_ratio() -> None:
    merged = merge_battle(
        "austerlitz",
        [
            _source(
                "wikipedia_body",
                sides=[
                    SideExtraction(
                        label="French Empire",
                        troop_reports=[_troops(65000, "wikipedia_body", "body", scope="unknown")],
                    )
                ],
            )
        ],
    )

    scope_gaps = [m for m in merged.missing if m.field_name == "troop_scope"]
    assert scope_gaps and scope_gaps[0].side_label == "French Empire"


def test_extraction_failures_travel_with_the_battle() -> None:
    from pipeline.extractors import ExtractionFailure

    merged = merge_battle(
        "austerlitz",
        [_source("wikipedia_infobox")],
        failures=[
            ExtractionFailure(
                battle_name="Battle of Austerlitz",
                source_ref="article",
                status="refusal",
                error="declined",
            )
        ],
    )

    assert merged.needs_review is True
    assert "refusal" in merged.review_notes


# ─── The stage ───────────────────────────────────────────────────────────────


def test_the_stage_module_satisfies_the_runner_contract() -> None:
    check_conforms(extract_stage)


def test_run_with_no_raw_input_logs_and_returns(
    spec: dict[str, Any], tmp_path: Path
) -> None:
    """A stage run before the crawl stage must not crash the pipeline."""
    overridden = {
        **spec,
        "params": {**spec["params"], "raw_root": str(tmp_path / "raw")},
    }

    extract_stage.run(overridden, StageContext(dry_run=True))


ACTIUM_URL = "https://en.wikipedia.org/wiki/Battle_of_Actium"
AUSTERLITZ_URL = "https://en.wikipedia.org/wiki/Battle_of_Austerlitz"


def _slug(url: str) -> str:
    """Return the stem the crawl stage stores an article under.

    Args:
        url: The article URL.

    Returns:
        The filename stem: sanitised title plus a digest of the URL.
    """
    return Path(article_filename(url)).stem


def _write_raw_corpus(root: Path) -> None:
    """Lay down a two-battle raw corpus in the crawl stage's own layout.

    Filenames come from the crawl stage's own article_filename, and the
    Wikidata entity is keyed by Q-id, so this exercises the sitelink join
    rather than a convenient naming coincidence.

    Args:
        root: The ``data/raw`` directory to populate.
    """
    (root / "battles_html").mkdir(parents=True)
    (root / "wikidata").mkdir(parents=True)
    (root / "citations").mkdir(parents=True)

    (root / "battles_html" / f"{_slug(AUSTERLITZ_URL)}.wikitext").write_text(
        _fixture("austerlitz.wikitext"), encoding="utf-8"
    )
    (root / "battles_html" / f"{_slug(ACTIUM_URL)}.wikitext").write_text(
        _fixture("actium_naval.wikitext"), encoding="utf-8"
    )
    (root / "wikidata" / "Q193320.json").write_text(
        _fixture("actium_wikidata.json"), encoding="utf-8"
    )

    records = [
        {
            "citation_id": "c1",
            "url": "https://example.org/actium",
            "battle_url": ACTIUM_URL,
            "content": "<h2>Account</h2><p>Antony fielded 230 ships at Actium.</p>",
        },
        {
            "citation_id": "c2",
            "url": "https://blocked.example/actium",
            "battle_url": ACTIUM_URL,
            "skipped": "domain_not_allowed",
        },
    ]
    (root / "citations" / f"{_slug(ACTIUM_URL)}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_discover_battles_joins_wikidata_through_the_sitelink(tmp_path: Path) -> None:
    """Wikidata files are keyed by Q-id, so the join has to run through enwiki."""
    _write_raw_corpus(tmp_path)

    discovered = {b.slug: b for b in extract_stage.discover_battles(tmp_path)}

    assert set(discovered) == {_slug(ACTIUM_URL), _slug(AUSTERLITZ_URL)}
    actium = discovered[_slug(ACTIUM_URL)]
    assert actium.wikidata_path is not None
    assert actium.wikidata_path.name == "Q193320.json"
    assert len(actium.citation_paths) == 1
    assert discovered[_slug(AUSTERLITZ_URL)].wikidata_path is None


def test_a_battle_name_drops_the_crawl_stages_url_digest(tmp_path: Path) -> None:
    _write_raw_corpus(tmp_path)

    discovered = {b.slug: b for b in extract_stage.discover_battles(tmp_path)}

    assert discovered[_slug(ACTIUM_URL)].name == "Battle of Actium"


def test_run_writes_the_four_processed_files(spec: dict[str, Any], tmp_path: Path) -> None:
    """A dry run skips the LLM entirely but still produces structured output."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _write_raw_corpus(raw)

    overridden = {
        **spec,
        "params": {
            **spec["params"],
            "raw_root": str(raw),
            "processed_root": str(processed),
            "batch_size": 1,
        },
    }

    extract_stage.run(overridden, StageContext(dry_run=True))

    for name in (
        "battles.jsonl",
        "commanders_raw.jsonl",
        "troop_reports.jsonl",
        "casualty_reports.jsonl",
    ):
        assert (processed / name).exists(), name

    battles = [
        json.loads(line)
        for line in (processed / "battles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {b["name"] for b in battles} == {"Battle of Actium", "Battle of Austerlitz"}

    actium = next(b for b in battles if b["name"] == "Battle of Actium")
    assert actium["wikidata_id"] == "Q193320"
    assert actium["date_start"] == "0031-09-02 BC"

    troops = [
        json.loads(line)
        for line in (processed / "troop_reports.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert troops and {t["scope"] for t in troops} >= {"engaged", "on_paper"}


def test_extract_battle_runs_the_llm_pass_when_a_service_is_given(
    spec: dict[str, Any], tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    _write_raw_corpus(raw)
    (battle,) = [
        b for b in extract_stage.discover_battles(raw) if b.slug == _slug(AUSTERLITZ_URL)
    ]

    provider = ScriptedProvider([_GOOD_PAYLOAD, _GOOD_PAYLOAD, _GOOD_PAYLOAD])
    service = LLMService(stage="extract", provider=provider, db_conn=None)

    merged = extract_stage.extract_battle(
        battle,
        service,
        system=spec["prompt"]["system"],
        template=spec["prompt"]["user_template"],
        schema=spec["prompt"]["output_schema"],
    )

    assert provider.requests, "the body pass should have been run"
    assert merged.n_sources >= 2
    french = next(s for s in merged.sides if s.label == "French Empire")
    assert any(c.apparent_role == "field_commander" for c in french.commanders)


def test_a_stage_spec_without_a_prompt_is_rejected(spec: dict[str, Any]) -> None:
    """The prompt is loaded from the spec; improvising one is not an option."""
    with pytest.raises(ValueError, match="prompt.system"):
        extract_stage.run({**spec, "prompt": None}, StageContext(dry_run=True))
