"""
Unit tests for the extract stage's plumbing: request volume, batch reuse,
provenance URLs, citation parsing and limited runs.

No network, database or API key is used here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.crawlers import citations
from pipeline.crawlers.wikipedia import article_filename
from pipeline.extractors.article import (
    build_passages,
    drop_repeated_passages,
    is_boilerplate,
    split_sections,
)
from pipeline.llm.base import LLMRequest
from pipeline.llm.offline import OfflineProvider, find_reusable_batch
from pipeline.stages import extract as extract_stage
from pipeline.stages.base import StageContext

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"x": {"type": "string"}}}
PROVIDER = "anthropic"
MODEL = "claude-haiku-5-5"
ACTIUM_URL = "https://en.wikipedia.org/wiki/Battle_of_Actium"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"


def _request(user: str) -> LLMRequest:
    return LLMRequest(system="s", user=user, json_schema=SCHEMA, schema_name="t")


# ─── Citation parsing (item 5) ───────────────────────────────────────────────


def test_a_citation_with_a_unicode_line_separator_stays_one_record(tmp_path: Path) -> None:
    """U+2028 sits unescaped in page text; splitlines() used to cut the record there."""
    record = {"url": "https://example.org/a", "content": "first half second half\x85end"}
    other = {"url": "https://example.org/b", "content": "plain"}
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n" + json.dumps(other) + "\n",
        encoding="utf-8",
    )

    records = extract_stage._citation_records(path)

    assert [r["url"] for r in records] == ["https://example.org/a", "https://example.org/b"]
    assert records[0]["content"] == "first half second half\x85end"


@pytest.mark.parametrize("separator", ["\x85", " ", " "])
def test_writer_output_with_a_separator_in_page_text_is_read_back_whole(
    tmp_path: Path, separator: str
) -> None:
    """The pilot's 11,337 warnings: 11,170 NEL, 37 LS and 30 PS inside page text.

    ``json.dumps(ensure_ascii=False)`` leaves these raw, so the writer's line
    is valid JSON; a reader that splits with ``str.splitlines`` cut it in two.
    """
    records = [
        {"url": "https://example.org/a", "content": f"before{separator}after", "n": 1},
        {"url": "https://example.org/b", "content": f"x{separator}y{separator}z", "n": 2},
    ]
    path = citations.write_citation_records(tmp_path, ACTIUM_URL, records)

    assert extract_stage._citation_records(path) == records


def test_a_genuinely_malformed_citation_line_is_still_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"url": "ok"}\n{not json\n', encoding="utf-8")

    assert extract_stage._citation_records(path) == [{"url": "ok"}]


# ─── Request volume (item 4) ─────────────────────────────────────────────────


def test_repeated_passages_are_sent_once_per_battle_across_documents() -> None:
    seen: set[str] = set()
    first = build_passages("B", "Troops  marched.\n\nThey fought.", source_type="web_secondary")
    again = build_passages("B", "Troops marched.\n\nThey fought.", source_type="web_secondary")

    kept_first = drop_repeated_passages(first, seen)
    kept_again = drop_repeated_passages(again, seen)

    assert len(kept_first) == 1
    assert kept_again == []


def test_dropping_repeats_renumbers_the_surviving_passages() -> None:
    seen = {"already asked"}
    passages = build_passages(
        "B", "already asked\n\nnew text here", source_type="web_secondary", max_chars=14
    )

    kept = drop_repeated_passages(passages, seen)

    assert [p.text for p in kept] == ["new text here"]
    assert (kept[0].index, kept[0].total) == (0, 1)


def test_a_rendered_table_of_contents_is_not_a_passage() -> None:
    raw = (
        "<html><body><h2>Contents</h2><p>(Top)</p><h2>Battle</h2><p>The army won.</p></body></html>"
    )

    passages = build_passages("B", raw, is_html=True)

    assert [p.section for p in passages] == ["Battle"]


@pytest.mark.parametrize(
    "title", ["Contents", "Explanatory notes", "Endnotes", "Works cited", "Notes and references"]
)
def test_reference_style_sections_are_skipped(title: str) -> None:
    text = f"== Battle ==\nThe army won.\n\n== {title} ==\n1. A note about a source."

    assert [t for t, _ in split_sections(text)] == ["Battle"]


@pytest.mark.parametrize(
    "text",
    [
        "(Top)\n\n1 Background\n\n2 Battle",
        "↑ See Flanders campaign\n\n↑ Old Style date",
        "PK\x03\x04\x00\x00\x08\x00���\x1f" * 20,
        "   ",
    ],
)
def test_structurally_empty_passages_are_boilerplate(text: str) -> None:
    assert is_boilerplate(text)


def test_ordinary_prose_is_not_boilerplate_even_when_short() -> None:
    assert not is_boilerplate("Antony fielded 230 ships.")


def test_citation_fragments_under_the_minimum_are_dropped() -> None:
    passages = build_passages("B", "Civil War Times", source_type="web_secondary", min_chars=40)

    assert passages == []


# ─── Batch reuse (item 3) ────────────────────────────────────────────────────


def _flush(directory: Path, users: list[str]) -> Path | None:
    provider = OfflineProvider(PROVIDER, MODEL, output_dir=directory)
    for user in users:
        provider.complete(_request(user))
    return provider.flush("extract")


def test_an_identical_pending_set_reuses_the_existing_batch_file(tmp_path: Path) -> None:
    first = _flush(tmp_path, ["a", "b", "c"])
    second = _flush(tmp_path, ["c", "b", "a"])

    assert first is not None
    assert second == first
    assert len(list(tmp_path.glob("extract_*.jsonl"))) == 1


def test_a_changed_pending_set_writes_a_new_file(tmp_path: Path) -> None:
    first = _flush(tmp_path, ["a", "b", "c"])
    # "a" was answered and imported since, so it is a cache hit and not pending.
    second = _flush(tmp_path, ["b", "c"])

    assert first is not None and second is not None
    assert second != first
    assert len(list(tmp_path.glob("extract_*.jsonl"))) == 2


def test_a_superset_file_is_not_reused(tmp_path: Path) -> None:
    _flush(tmp_path, ["a", "b"])

    assert _flush(tmp_path, ["a", "b", "c"]) is not None
    assert len(list(tmp_path.glob("extract_*.jsonl"))) == 2


def test_batches_of_other_stages_are_never_reused(tmp_path: Path) -> None:
    provider = OfflineProvider(PROVIDER, MODEL, output_dir=tmp_path)
    provider.complete(_request("a"))
    resolve_file = provider.flush("resolve")

    assert resolve_file is not None
    assert _flush(tmp_path, ["a"]) != resolve_file


def test_reuse_finds_nothing_in_a_missing_directory(tmp_path: Path) -> None:
    assert find_reusable_batch("extract", {"deadbeef"}, tmp_path / "nope") is None


# ─── Provenance URLs (item 7) ────────────────────────────────────────────────


def _slug(url: str) -> str:
    return Path(article_filename(url)).stem


def _raw_with_actium(root: Path, *, crawl_state: bool) -> None:
    (root / "battles_html").mkdir(parents=True)
    (root / "battles_html" / f"{_slug(ACTIUM_URL)}.wikitext").write_text(
        (FIXTURES / "actium_naval.wikitext").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "wikidata").mkdir()
    (root / "wikidata" / "Q193320.json").write_text(
        (FIXTURES / "actium_wikidata.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    if crawl_state:
        (root / "crawl_state.json").write_text(
            json.dumps({"battle_urls": [ACTIUM_URL]}), encoding="utf-8"
        )


def test_the_article_url_comes_from_the_crawl_state(tmp_path: Path) -> None:
    _raw_with_actium(tmp_path, crawl_state=True)

    battle = extract_stage.discover_battles(tmp_path)[0]

    assert battle.article_ref == ACTIUM_URL
    assert battle.dbpedia_ref == "http://dbpedia.org/resource/Battle_of_Actium"


def test_the_article_url_falls_back_to_the_title_without_crawl_state(tmp_path: Path) -> None:
    _raw_with_actium(tmp_path, crawl_state=False)

    battle = extract_stage.discover_battles(tmp_path)[0]

    assert battle.article_ref == "https://en.wikipedia.org/wiki/Battle_of_Actium"


def test_merged_sources_carry_urls_and_never_local_paths(tmp_path: Path) -> None:
    from pipeline.config import load_agent_spec

    spec = load_agent_spec("extract")
    _raw_with_actium(tmp_path, crawl_state=True)
    battle = extract_stage.discover_battles(tmp_path)[0]
    system, template, schema = extract_stage._prompt_parts(spec)

    merged = extract_stage.extract_battle(
        battle, None, system=system, template=template, schema=schema
    )

    refs = {
        report.provenance.source_ref
        for side in merged.sides
        for report in [*side.troop_reports, *side.casualty_reports]
    }
    assert refs, "fixture should yield reports"
    assert all(ref.startswith(("http://", "https://")) for ref in refs), refs
    assert "https://www.wikidata.org/wiki/Q193320" in refs or ACTIUM_URL in refs


# ─── Limited runs (item 6) ───────────────────────────────────────────────────


def test_a_limited_run_leaves_the_full_outputs_alone(tmp_path: Path) -> None:
    from pipeline.config import load_agent_spec

    raw, processed = tmp_path / "raw", tmp_path / "processed"
    _raw_with_actium(raw, crawl_state=True)
    spec = load_agent_spec("extract")
    spec = {
        **spec,
        "params": {**spec["params"], "raw_root": str(raw), "processed_root": str(processed)},
    }

    extract_stage.run(spec, StageContext(dry_run=True))
    full = (processed / "battles.jsonl").read_text(encoding="utf-8")

    extract_stage.run(spec, StageContext(dry_run=True, limit=1))

    assert (processed / "battles.jsonl").read_text(encoding="utf-8") == full
    assert (processed / "battles.limited.jsonl").exists()
    for name in ("commanders_raw", "troop_reports", "casualty_reports"):
        assert (processed / f"{name}.limited.jsonl").exists()
