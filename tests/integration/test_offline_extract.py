"""
Integration test: the extract stage in offline mode, end to end.

Needs a real PostgreSQL instance, like the other integration tests. The claim
under test is that deferring is invisible once the answers arrive: a first run
defers, writes a batch file and leaves nothing in ``llm_calls`` or
``missing_data_log`` that would be wrong later; after the answers are imported
a second run produces exactly what a run that had the answers from the start
would have produced.
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
from pipeline.llm import CallStatus, LLMRequest, LLMResponse, LLMService, request_hash
from pipeline.llm.offline import import_responses
from pipeline.stages import extract
from pipeline.stages.base import StageContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"
_OUTPUT_FILES = ("battles", "commanders_raw", "troop_reports", "casualty_reports")


def _database_available() -> tuple[bool, str]:
    """Check whether DATABASE_URL points at a reachable database.

    Returns:
        (available, reason). Reason is empty when available.
    """
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
    """Lay down two fixture articles the way the crawl stage would."""
    articles = tmp_path / "raw" / "battles_html"
    articles.mkdir(parents=True)
    shutil.copy(FIXTURES / "actium_naval.wikitext", articles / "actium.wikitext")
    shutil.copy(FIXTURES / "austerlitz.wikitext", articles / "austerlitz.wikitext")
    return tmp_path / "raw"


def _spec(raw_root: Path, processed: Path) -> dict[str, Any]:
    return load_agent_spec("extract", {"raw_root": str(raw_root), "processed_root": str(processed)})


def _answer_for(line: dict[str, Any]) -> dict[str, Any]:
    """A schema-valid synthetic answer for one exported request."""
    return {
        "battle_name": "Synthetic Battle",
        "commanders": [
            {"name": "Synthetic Commander", "side": "Side A", "apparent_role": "field_commander"}
        ],
        "troop_reports": [
            {"side": "Side A", "branch": "total", "value": 1000, "scope": "engaged"},
            {"side": "Side B", "branch": "total", "value": 2000, "scope": "engaged"},
        ],
        "casualty_reports": [{"side": "Side A", "casualty_type": "killed", "value": 10}],
        "outcome": {"victor": "Side A", "outcome_level": "victory", "evidence": line["user"][:20]},
    }


class _ReplayProvider:
    """Answers each request from a precomputed hash-to-data map."""

    name = "anthropic"
    model = "claude-haiku-5-5"

    def __init__(self, answers: dict[str, dict[str, Any]]) -> None:
        self._answers = answers

    def complete(self, request: LLMRequest) -> LLMResponse:
        digest = request_hash(request, self.name, self.model)
        return LLMResponse(
            status=CallStatus.OK,
            provider=self.name,
            model=self.model,
            request_hash=digest,
            data=self._answers[digest],
        )


def _scalar(conn: Any, sql: str) -> int:
    return int(conn.execute(text(sql)).scalar_one())


def _outputs(processed: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            json.loads(line)
            for line in (processed / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in _OUTPUT_FILES
    }


def test_offline_extract_defers_then_matches_a_cached_run(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", tmp_path / "llm_requests")
    ctx = StageContext(db_conn=conn)

    # Arrange / Act: first run, nothing cached.
    first_out = tmp_path / "processed_first"
    outcome = extract.run(_spec(raw_root, first_out), ctx)

    # Assert: deferred, batch file written, nothing persisted that would be wrong later.
    assert outcome is not None and outcome.awaiting_llm
    assert outcome.batch_file is not None and outcome.batch_file.exists()
    lines = [
        json.loads(raw)
        for raw in outcome.batch_file.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    assert len(lines) == outcome.deferred_requests > 0
    assert {line["agent_model"] for line in lines} == {"haiku"}
    assert _scalar(conn, "SELECT count(*) FROM llm_calls") == 0
    assert _scalar(conn, "SELECT count(*) FROM missing_data_log") == 0
    flagged = conn.execute(text("SELECT needs_review, review_notes FROM battles")).fetchall()
    assert flagged and all(row[0] for row in flagged)
    assert all("pending offline processing" in (row[1] or "") for row in flagged)

    # Act: answer every request and import.
    answers = {line["request_hash"]: _answer_for(line) for line in lines}
    response_file = tmp_path / "responses.jsonl"
    response_file.write_text(
        "\n".join(
            json.dumps({"request_hash": h, "stage": "extract", "data": d})
            for h, d in answers.items()
        )
        + "\n",
        encoding="utf-8",
    )
    result = import_responses(response_file, conn, request_files=[outcome.batch_file])
    assert (result.imported, result.errors) == (len(lines), 0)
    assert _scalar(conn, "SELECT count(*) FROM llm_calls WHERE provider = 'claude_code'") == len(
        lines
    )

    # Act: second run, answers are in the cache.
    second_out = tmp_path / "processed_second"
    second = extract.run(_spec(raw_root, second_out), ctx)

    # Assert: nothing deferred, no new batch file.
    assert second is None
    assert len(list((tmp_path / "llm_requests").glob("*.jsonl"))) == 1

    # Act: the same answers served by a provider from the very first run.
    reference_out = tmp_path / "processed_reference"
    monkeypatch.setattr(
        extract,
        "_build_service",
        lambda spec, context: LLMService(
            "extract", _ReplayProvider(answers), default_max_tokens=4096
        ),
    )
    extract.run(_spec(raw_root, reference_out), StageContext())

    # Assert: byte-for-byte the same outputs as if the answers had always been there.
    assert _outputs(second_out) == _outputs(reference_out)
    assert not any("pending offline" in b["review_notes"] for b in _outputs(second_out)["battles"])
    stale = conn.execute(
        text("SELECT count(*) FROM battles WHERE review_notes LIKE '%pending offline%'")
    ).scalar_one()
    assert stale == 0
