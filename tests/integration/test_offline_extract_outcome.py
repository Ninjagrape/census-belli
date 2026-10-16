"""
Integration test: the targeted outcome request in offline mode.

A battle the deterministic merge cannot decide gets an ``extract_outcome``
request in its own batch file. Answering that file must change the outcomes on
the next extract run even though the (much larger) passage requests stay
unanswered, which is the state the real corpus is in.
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
from pipeline.llm.offline import import_responses
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
    pytest.mark.skipif(not _AVAILABLE, reason=f"No database available ({_REASON})."),
]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """Provide an engine against a freshly applied schema."""
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
    """Provide a connection whose writes are rolled back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """Lay down one article whose result field names no victor."""
    articles = tmp_path / "raw" / "battles_html"
    articles.mkdir(parents=True)
    shutil.copy(FIXTURES / "contested_field.wikitext", articles / "contested.wikitext")
    return tmp_path / "raw"


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


def test_outcome_requests_are_deferred_to_their_own_file_and_applied_once_answered(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests_dir = tmp_path / "llm_requests"
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", requests_dir)
    ctx = StageContext(db_conn=conn)

    def spec(processed: Path) -> dict[str, Any]:
        return load_agent_spec(
            "extract", {"raw_root": str(raw_root), "processed_root": str(processed)}
        )

    # Act: first run, nothing cached.
    outcome = extract.run(spec(tmp_path / "first"), ctx)

    # Assert: two batch files, the outcome one named by the stage outcome.
    assert outcome is not None and outcome.batch_file is not None
    assert outcome.batch_file.name.startswith("extract_outcome_")
    outcome_lines = _lines(outcome.batch_file)
    assert len(outcome_lines) == 1
    assert outcome_lines[0]["stage"] == "extract_outcome"
    assert outcome_lines[0]["agent_model"] == "sonnet"
    assert "Contested; see Aftermath" in outcome_lines[0]["user"]
    passage_files = [p for p in requests_dir.glob("extract_*.jsonl") if p != outcome.batch_file]
    assert len(passage_files) == 1
    assert {line["agent_model"] for line in _lines(passage_files[0])} == {"haiku"}
    assert (
        conn.execute(
            text("SELECT count(*) FROM battle_sides WHERE outcome IS NOT NULL")
        ).scalar_one()
        == 0
    )

    # Act: answer only the outcome file.
    answer = tmp_path / "outcome_responses.jsonl"
    answer.write_text(
        json.dumps(
            {
                "request_hash": outcome_lines[0]["request_hash"],
                "stage": "extract_outcome",
                "data": {"winner": "Kingdom of Varra", "evidence": "Varra held the ground"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = import_responses(answer, conn, request_files=[outcome.batch_file])
    assert (result.imported, result.errors) == (1, 0)

    # Act: second run; the passages are still unanswered.
    second_out = tmp_path / "second"
    second = extract.run(spec(second_out), ctx)

    # Assert: the cached answer is applied while the passage requests stay deferred.
    assert second is not None and second.awaiting_llm
    assert second.batch_file is not None
    assert not second.batch_file.name.startswith("extract_outcome_")
    (battle,) = [
        json.loads(raw)
        for raw in (second_out / "battles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {s["label"]: s["outcome"] for s in battle["sides"]} == {
        "Kingdom of Varra": "victory",
        "Duchy of Selm": "defeat",
    }
    assert battle["outcome_method"] == "llm_outcome"
    stored = dict(
        conn.execute(text("SELECT side_label, outcome::text FROM battle_sides")).fetchall()
    )
    assert stored == {"Kingdom of Varra": "victory", "Duchy of Selm": "defeat"}
    assert len(list(requests_dir.glob("extract_outcome_*.jsonl"))) == 1
