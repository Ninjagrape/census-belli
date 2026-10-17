"""Overrides against a real schema: manual provenance and the 'manual' battle type source."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.overrides import OverrideEntry, manual_source_id


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

_ENTRY = OverrideEntry(
    id="ov-0001",
    kind="battle_type",
    stage="classify",
    target=MappingProxyType({"battle": "b"}),
    value="field",
    source="Chandler, The Campaigns of Napoleon (1966), p. 807",
    note="",
    reviewed_by="lee",
    reviewed_on="2026-10-17",
)


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(
            f"Refusing to reset database {db_name!r}: it does not look like a test database."
        )
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


def test_manual_source_id_is_find_or_create(conn: Any) -> None:
    first = manual_source_id(conn, _ENTRY)
    again = manual_source_id(conn, _ENTRY)
    other = manual_source_id(conn, replace(_ENTRY, source="Another citation"))

    assert first == again
    assert other != first
    row = conn.execute(
        text("SELECT CAST(source_type AS TEXT), title, url FROM sources WHERE source_id = :id"),
        {"id": first},
    ).one()
    assert tuple(row) == ("manual_entry", _ENTRY.source, None)


def test_battle_type_source_accepts_manual_and_still_rejects_others(conn: Any) -> None:
    conn.execute(
        text(
            "INSERT INTO battles (name, source_slug, battle_type, battle_type_source) "
            "VALUES ('B', 'b', 'field', 'manual')"
        )
    )
    stored = conn.execute(text("SELECT battle_type_source FROM battles WHERE source_slug = 'b'"))
    assert stored.scalar() == "manual"

    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO battles (name, source_slug, battle_type_source) "
                "VALUES ('C', 'c', 'someone')"
            )
        )
