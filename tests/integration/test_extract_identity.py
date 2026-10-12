"""
Integration tests for how extract identifies battles and attributes requests.

Needs a real PostgreSQL instance, like the other integration tests. Covers
same-name battles, the slug column's legacy fallback, the resolve battle
index, the rewritten ``no_empty_extractions`` gate, and the provenance the
offline round trip leaves behind (``llm_calls.battle_id``, ``sources.url``).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text

from pipeline.config import load_agent_spec
from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.extractors import (
    BattleExtraction,
    BattleFacts,
    SideExtraction,
    write_battle,
)
from pipeline.extractors.store import reserve_battle_id
from pipeline.llm import CallStatus, LLMRequest, LLMResponse, TokenUsage
from pipeline.llm.call_log import log_call
from pipeline.llm.offline import import_responses
from pipeline.quality import QualityRunner
from pipeline.resolvers import load_battle_index
from pipeline.stages import extract
from pipeline.stages.base import StageContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"


def _database_available() -> tuple[bool, str]:
    try:
        url = database_url()
    except DatabaseConfigError as exc:
        return False, str(exc)
    try:
        engine = get_engine(url, pool_pre_ping=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        return False, f"cannot connect: {type(exc).__name__}: {exc}"
    return True, ""


_AVAILABLE, _REASON = _database_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _AVAILABLE,
        reason=f"No database available ({_REASON}). Run `make db-up` first.",
    ),
]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(f"Refusing to reset database {db_name!r}: not a test database.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Any]:
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _battle(slug: str, name: str = "Battle of Luetzen", *, date: str | None = None) -> Any:
    return BattleExtraction(
        slug=slug,
        name=name,
        facts=BattleFacts(name=name, date_start=date),
        sides=[SideExtraction(label="A"), SideExtraction(label="B")],
        n_sources=1,
    )


def _scalar(conn: Any, sql: str, **params: Any) -> Any:
    return conn.execute(text(sql), params).scalar_one()


# ─── Same-name battles ───────────────────────────────────────────────────────


def test_two_battles_sharing_a_name_stay_two_rows(conn: Any) -> None:
    first = write_battle(conn, _battle("Battle_of_Luetzen-1111111111", date="1632-11-16"))
    second = write_battle(conn, _battle("Battle_of_Luetzen-2222222222", date="1813-05-02"))

    assert first != second
    assert _scalar(conn, "SELECT count(*) FROM battles WHERE name = 'Battle of Luetzen'") == 2
    assert _scalar(conn, "SELECT count(*) FROM battle_sides") == 4


def test_rewriting_a_same_name_battle_updates_its_own_row(conn: Any) -> None:
    first = write_battle(conn, _battle("Battle_of_Luetzen-1111111111", date="1632-11-16"))
    second = write_battle(conn, _battle("Battle_of_Luetzen-2222222222", date="1813-05-02"))

    again = write_battle(conn, _battle("Battle_of_Luetzen-2222222222", date="1813-05-02"))

    assert again == second
    assert _scalar(conn, "SELECT count(*) FROM battles") == 2
    assert _scalar(conn, "SELECT date_start::text FROM battles WHERE battle_id = :i", i=first) == (
        "1632-11-16"
    )


def test_a_row_written_before_the_slug_column_is_claimed_by_name(conn: Any) -> None:
    legacy = _scalar(
        conn, "INSERT INTO battles (name) VALUES ('Battle of Luetzen') RETURNING battle_id"
    )

    claimed = write_battle(conn, _battle("Battle_of_Luetzen-1111111111"))
    other = write_battle(conn, _battle("Battle_of_Luetzen-2222222222"))

    assert claimed == legacy
    assert other != legacy
    assert _scalar(conn, "SELECT source_slug FROM battles WHERE battle_id = :i", i=legacy) == (
        "Battle_of_Luetzen-1111111111"
    )


def test_the_resolve_index_keys_on_slug_and_drops_ambiguous_names(conn: Any) -> None:
    first = write_battle(conn, _battle("Battle_of_Luetzen-1111111111"))
    second = write_battle(conn, _battle("Battle_of_Luetzen-2222222222"))
    solo = write_battle(conn, _battle("Battle_of_Actium-3333333333", "Battle of Actium"))

    index = load_battle_index(conn)

    assert index["Battle_of_Luetzen-1111111111"] == first
    assert index["Battle_of_Luetzen-2222222222"] == second
    assert "Battle of Luetzen" not in index
    assert index["Battle of Actium"] == solo
    assert index["Battle_of_Actium-3333333333"] == solo


def test_reserving_a_battle_id_is_stable_and_completed_by_the_write(conn: Any) -> None:
    reserved = reserve_battle_id(conn, "Battle_of_Luetzen-1111111111", "Battle of Luetzen")
    assert reserved is not None
    assert reserve_battle_id(conn, "Battle_of_Luetzen-1111111111", "Battle of Luetzen") == reserved

    written = write_battle(conn, _battle("Battle_of_Luetzen-1111111111", date="1632-11-16"))

    assert written == reserved
    assert _scalar(conn, "SELECT count(*) FROM battles") == 1


# ─── The no_empty_extractions gate ───────────────────────────────────────────


def _gate(conn: Any) -> Any:
    spec = load_agent_spec("extract")
    check = next(c for c in spec["quality_checks"] if c["name"] == "no_empty_extractions")
    return QualityRunner(conn).run_one(check)


def test_the_empty_extractions_gate_fails_on_an_empty_database(conn: Any) -> None:
    result = _gate(conn)

    assert "SQL execution failed" not in result.message, result.message
    assert not result.passed
    assert result.actual_value == 1.0


def test_the_empty_extractions_gate_fails_when_extract_wrote_nothing_useful(conn: Any) -> None:
    for i in range(5):
        conn.execute(text("INSERT INTO battles (name) VALUES (:n)"), {"n": f"Bare {i}"})

    assert not _gate(conn).passed


def test_the_empty_extractions_gate_passes_on_a_healthy_corpus(conn: Any) -> None:
    for i in range(30):
        write_battle(conn, _battle(f"Healthy_{i}-0000000000", f"Healthy {i}", date="1700-01-01"))
    write_battle(conn, BattleExtraction(slug="stub-0", name="Stub", facts=BattleFacts(name="Stub")))

    result = _gate(conn)

    assert result.passed, result.message
    assert result.actual_value == pytest.approx(1 / 31)


# ─── llm_calls.battle_id and sources.url ─────────────────────────────────────


def _ok_response() -> LLMResponse:
    return LLMResponse(
        status=CallStatus.OK,
        provider="p",
        model="m",
        request_hash="h" * 64,
        data={"battle_name": "x"},
        usage=TokenUsage(),
        cost_usd=0.0,
    )


def test_a_stale_battle_id_costs_the_attribution_not_the_answer(conn: Any) -> None:
    request = LLMRequest(system="", user="", json_schema={}, metadata={"battle_id": 987654})

    call_id = log_call(conn, "extract", request, _ok_response())

    assert call_id is not None
    assert _scalar(conn, "SELECT battle_id FROM llm_calls WHERE call_id = :i", i=call_id) is None
    # The failed insert must not have poisoned the transaction.
    assert _scalar(conn, "SELECT 1") == 1


def test_a_valid_battle_id_is_recorded(conn: Any) -> None:
    battle_id = write_battle(conn, _battle("Battle_of_Luetzen-1111111111"))
    request = LLMRequest(system="", user="", json_schema={}, metadata={"battle_id": battle_id})

    call_id = log_call(conn, "extract", request, _ok_response())

    assert _scalar(conn, "SELECT battle_id FROM llm_calls WHERE call_id = :i", i=call_id) == (
        battle_id
    )


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    articles = tmp_path / "raw" / "battles_html"
    articles.mkdir(parents=True)
    shutil.copy(
        FIXTURES / "actium_naval.wikitext", articles / "Battle_of_Actium-aaaaaaaaaa.wikitext"
    )
    (tmp_path / "raw" / "crawl_state.json").write_text(
        json.dumps({"battle_urls": []}), encoding="utf-8"
    )
    return tmp_path / "raw"


def test_the_offline_round_trip_attributes_calls_and_stores_urls(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", tmp_path / "llm_requests")
    spec = load_agent_spec(
        "extract", {"raw_root": str(raw_root), "processed_root": str(tmp_path / "processed")}
    )

    outcome = extract.run(spec, StageContext(db_conn=conn))

    assert outcome is not None and outcome.batch_file is not None
    lines = [
        json.loads(raw)
        for raw in outcome.batch_file.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    battle_id = _scalar(conn, "SELECT battle_id FROM battles")
    assert {line["metadata"]["battle_id"] for line in lines} == {battle_id}

    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_hash": line["request_hash"],
                    "stage": "extract",
                    "data": {"battle_name": "x"},
                }
            )
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )
    result = import_responses(answers, conn, request_files=[outcome.batch_file])
    assert result.errors == 0
    assert _scalar(conn, "SELECT count(*) FROM llm_calls WHERE battle_id IS NULL") == 0

    urls = [row[0] for row in conn.execute(text("SELECT url FROM sources")).fetchall()]
    assert urls
    assert all(str(u).startswith(("http://", "https://")) for u in urls), urls

    # A re-run with the same pending set (none here: all answered) must not duplicate rows.
    assert extract.run(spec, StageContext(db_conn=conn)) is None
    assert _scalar(conn, "SELECT count(*) FROM battles") == 1
