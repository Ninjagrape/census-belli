"""
A resolve re-run with answered LLM requests is idempotent.

The failure this guards: SPARQL ``GROUP_CONCAT`` returns occupations and
citizenships in an order the endpoint does not promise, the order is rendered
into the prompt, and the prompt is hashed for the ``llm_calls`` cache. Two runs
over identical mentions then asked different questions, so answered requests
missed the cache and the corpus never converged (handover 9.3).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, text

from pipeline.config import load_agent_spec
from pipeline.db import apply_schema, get_engine
from pipeline.llm.offline import OfflineProvider, import_responses
from pipeline.llm.service import LLMService
from pipeline.resolvers import (
    Mention,
    ResolveCounts,
    group_mentions,
    load_battle_index,
    load_side_index,
    write_identity,
)
from pipeline.resolvers.candidates import _merge_rows
from pipeline.resolvers.disambiguate import prompt_parts
from pipeline.resolvers.records import BattleContext
from pipeline.stages.resolve import LazyService, _build_identities, resolve_groups

BATTLE = "Battle of Zama"
SIDE = "Rome"


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """Provide an engine against a freshly applied schema."""
    eng = get_engine()
    name = (eng.url.database or "").lower()
    if "test" not in name and "general_war" not in name:
        eng.dispose()
        pytest.skip("Refusing to reset a database that does not look like a test database.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    """Give each test an empty corpus, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


class _ShuffledSource:
    """Serves the same two namesakes with each run's own GROUP_CONCAT order."""

    def __init__(self, run: int) -> None:
        occupations = [["general", "politician", "writer"], ["writer", "general", "politician"]]
        countries = [["Rome", "Carthage"], ["Carthage", "Rome"]]
        rows = [
            {
                "person": f"http://www.wikidata.org/entity/Q{qid}",
                "name": "Scipio",
                "personLabel": "Scipio",
                "occupations": ", ".join(occupations[(run + qid) % 2]),
                "countries": ", ".join(countries[(run + qid) % 2]),
            }
            for qid in (101, 102)
        ]
        self.candidates = [c for c, _ in _merge_rows(rows)]

    def prefetch(self, groups: Any) -> None:
        return None

    def search(self, group: Any) -> list[Any]:
        return list(self.candidates)

    def non_person_kind(self, group: Any) -> str | None:
        return None


def _mention() -> Mention:
    return Mention(
        battle_slug="Battle_of_Zama",
        battle_name=BATTLE,
        side_label=SIDE,
        name="Scipio",
        apparent_role="field_commander",
        context=BattleContext(slug="Battle_of_Zama", name=BATTLE, year=-201, date_text="202 BC"),
    )


def _seed(conn: Connection) -> None:
    battle_id = conn.execute(
        text(
            "INSERT INTO battles (name, date_start, date_precision) "
            "VALUES (:n, CAST('0202-10-19 BC' AS DATE), 'day') RETURNING battle_id"
        ),
        {"n": BATTLE},
    ).scalar_one()
    conn.execute(
        text("INSERT INTO battle_sides (battle_id, side_label, polity) VALUES (:b, :s, 'Rome')"),
        {"b": battle_id, "s": SIDE},
    )


def _run(conn: Connection, run: int, out_dir: Path) -> tuple[OfflineProvider, dict[str, str]]:
    """One resolve pass through the offline provider, then the database write."""
    system, template, schema = prompt_parts(load_agent_spec("resolve"))
    provider = OfflineProvider("anthropic", "claude-haiku-5-5", output_dir=out_dir)
    service = LLMService("resolve", provider, db_conn=conn)
    groups = group_mentions([_mention()])
    decisions = resolve_groups(
        groups,
        _ShuffledSource(run),
        LazyService(lambda: service),
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        system=system,
        template=template,
        schema=schema,
    )
    counts = ResolveCounts()
    battles, sides = load_battle_index(conn), load_side_index(conn)
    for identity in _build_identities(groups, decisions):
        write_identity(conn, identity, battles, sides, counts, {})
    return provider, {key: d.status for key, d in decisions.items()}


def _links(conn: Connection) -> list[tuple[Any, ...]]:
    return [
        tuple(row)
        for row in conn.execute(
            text(
                "SELECT g.wikidata_id, g.canonical_name, bc.command_role, bc.battle_id "
                "FROM battle_commanders bc JOIN generals g USING (general_id) "
                "ORDER BY g.wikidata_id, bc.battle_id"
            )
        )
    ]


def test_a_rerun_after_import_defers_nothing_and_writes_the_same_links(
    conn: Connection, tmp_path: Path
) -> None:
    _seed(conn)

    first, statuses = _run(conn, 0, tmp_path)
    assert first.pending == 1
    assert set(statuses.values()) == {"unresolved"}
    request_file = first.flush("resolve")
    assert request_file is not None
    digest = json.loads(request_file.read_text(encoding="utf-8").splitlines()[0])["request_hash"]

    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        json.dumps({"request_hash": digest, "data": {"match": "Q101", "confidence": 0.9}}) + "\n",
        encoding="utf-8",
    )
    result = import_responses(answers, conn, request_files=[request_file])
    assert result.imported == 1

    second, _ = _run(conn, 1, tmp_path)
    links_second = _links(conn)
    third, _ = _run(conn, 2, tmp_path)

    assert second.pending == 0, "the answered request must hit the cache on a re-run"
    assert third.pending == 0
    assert links_second, "the cached answer should have produced a link"
    assert links_second[0][0] == "Q101"
    assert _links(conn) == links_second
