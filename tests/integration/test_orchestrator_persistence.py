"""Integration tests: stage writes made through the orchestrator are durable.

The stage integration tests call ``stage.run(spec, StageContext(db_conn=conn))``
directly, so none of them noticed that the CLI path used to call ``run(spec)``
with no context, which made extract and resolve skip every database write.
These tests drive stages through ``run_pipeline`` and then look at the data
from a separate, freshly opened connection, which is the only view that proves
a commit happened.

Needs a real PostgreSQL instance. Each test rebuilds the schema because the
orchestrator commits for real.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text

from pipeline import orchestrator
from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.llm.offline import import_responses
from pipeline.stages.base import StageContext, StageOutcome

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


@pytest.fixture
def engine() -> Iterator[Engine]:
    """Provide an engine against a freshly applied schema, rebuilt per test."""
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(f"Refusing to reset database {db_name!r}: not a test database.")
    apply_schema(eng, drop_existing=True)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """Lay down two fixture articles the way the crawl stage would."""
    articles = tmp_path / "raw" / "battles_html"
    articles.mkdir(parents=True)
    shutil.copy(FIXTURES / "actium_naval.wikitext", articles / "actium.wikitext")
    shutil.copy(FIXTURES / "austerlitz.wikitext", articles / "austerlitz.wikitext")
    return tmp_path / "raw"


def _overrides(raw_root: Path, tmp_path: Path) -> dict[str, Any]:
    return {"raw_root": str(raw_root), "processed_root": str(tmp_path / "processed")}


def _count_from_fresh_connection(engine: Engine, table: str) -> int:
    """Count rows as a separate session sees them, i.e. only what was committed."""
    with engine.connect() as fresh:
        return int(fresh.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())  # noqa: S608


def _answer(line: dict[str, Any]) -> dict[str, Any]:
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


def test_halted_offline_extract_writes_persist_for_other_connections(
    engine: Engine, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", tmp_path / "llm_requests")

    with engine.connect() as conn:
        results = orchestrator.run_pipeline(
            ["extract"], config_overrides=_overrides(raw_root, tmp_path), db_conn=conn
        )

    assert [r.awaiting_llm for r in results] == [True]
    assert _count_from_fresh_connection(engine, "battles") == 2


def test_completed_extract_writes_persist_for_other_connections(
    engine: Engine, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", tmp_path / "llm_requests")
    overrides = _overrides(raw_root, tmp_path)

    with engine.connect() as conn:
        first = orchestrator.run_pipeline(["extract"], config_overrides=overrides, db_conn=conn)
    batch_file = Path(str(first[0].batch_file))
    lines = [
        json.loads(raw)
        for raw in batch_file.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    response_file = tmp_path / "responses.jsonl"
    response_file.write_text(
        "\n".join(
            json.dumps(
                {"request_hash": ln["request_hash"], "stage": "extract", "data": _answer(ln)}
            )
            for ln in lines
        )
        + "\n",
        encoding="utf-8",
    )
    with engine.connect() as importer:
        import_responses(response_file, importer, request_files=[batch_file])
        importer.commit()

    with engine.connect() as conn:
        second = orchestrator.run_pipeline(["extract"], config_overrides=overrides, db_conn=conn)

    assert [r.awaiting_llm for r in second] == [False]
    assert _count_from_fresh_connection(engine, "battles") == 2
    assert _count_from_fresh_connection(engine, "troop_reports") > 0
    assert _count_from_fresh_connection(engine, "battle_sides") > 0


def _install_stage(monkeypatch: pytest.MonkeyPatch, name: str, run: Any, **attrs: Any) -> None:
    module = types.ModuleType(f"pipeline.stages.{name}")
    module.run = run  # type: ignore[attr-defined]
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, f"pipeline.stages.{name}", module)


_INSERT_SOURCE = text(
    "INSERT INTO sources (title, source_type) VALUES ('orchestrator-test', 'manual_entry')"
)


def test_stage_that_raises_leaves_no_partial_writes(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(spec: dict[str, Any], context: StageContext | None = None) -> None:
        assert context is not None and context.db_conn is not None
        context.db_conn.execute(_INSERT_SOURCE)
        raise RuntimeError("boom")

    _install_stage(monkeypatch, "classify", failing)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    with engine.connect() as conn:
        results = orchestrator.run_pipeline(["classify"], db_conn=conn)
        # The failed attempt must not leave the shared connection mid-transaction.
        assert conn.execute(text("SELECT count(*) FROM sources")).scalar_one() == 0

    assert results[0].success is False
    assert _count_from_fresh_connection(engine, "sources") == 0


def test_writes_are_committed_when_stage_returns_awaiting_llm_outcome(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deferring(spec: dict[str, Any], context: StageContext | None = None) -> StageOutcome:
        assert context is not None and context.db_conn is not None
        context.db_conn.execute(_INSERT_SOURCE)
        return StageOutcome(deferred_requests=1)

    _install_stage(monkeypatch, "classify", deferring)

    with engine.connect() as conn:
        results = orchestrator.run_pipeline(["classify"], db_conn=conn)

    assert results[0].awaiting_llm
    assert _count_from_fresh_connection(engine, "sources") == 1


def test_stage_that_owns_its_connection_gets_no_shared_connection(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def owner(spec: dict[str, Any], context: StageContext | None = None) -> None:
        seen.append(context.db_conn if context else "no-context")

    _install_stage(monkeypatch, "reconcile", owner, OWNS_DB_CONNECTION=True)

    with engine.connect() as conn:
        orchestrator.run_pipeline(["reconcile"], db_conn=conn)

    assert seen and all(c is None for c in seen)
