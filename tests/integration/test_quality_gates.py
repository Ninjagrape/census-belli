"""
Integration tests for the quality gate runner against a real database.

The thing under test cannot be reproduced with a stub. Postgres aborts the
entire transaction when a statement fails, so a broken check does not merely
fail on its own: every later check on the same connection comes back
"current transaction is aborted, commands ignored until end of transaction
block". A stubbed connection has no transaction to abort, and so reports every
check as fine.

That is not hypothetical. agents/extract.yaml declared a check against
data.commanders_raw, a table in no schema. It failed, and took the valid check
after it down with it, which is how a stage came to report two broken gates
when only one was actually broken.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, text

from pipeline.config import load_agent_spec
from pipeline.db import (
    DatabaseConfigError,
    apply_schema,
    database_url,
    get_engine,
)
from pipeline.quality import QualityRunner, Severity


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
        pytest.skip(
            f"Refusing to reset database {db_name!r}: the name suggests it is not a "
            "test database. Point DATABASE_URL at general_war or a *test* database."
        )

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


_BROKEN: dict[str, Any] = {
    "name": "queries_a_table_that_does_not_exist",
    "check": "SELECT COUNT(*) FROM no_such_table_anywhere",
    "threshold": "< 1",
    "severity": "warning",
}

_VALID: dict[str, Any] = {
    "name": "counts_battles",
    "check": "SELECT COUNT(*) FROM battles",
    "threshold": ">= 0",
    "severity": "warning",
}


def test_a_failing_check_does_not_poison_the_checks_after_it(conn: Any) -> None:
    """The regression the per-check savepoint exists for.

    Without one the second check reports an aborted transaction rather than its
    own result, so a single broken gate invalidates every gate behind it.
    """
    broken, valid = QualityRunner(conn).run_all([_BROKEN, _VALID])

    assert broken.passed is False
    assert "no_such_table_anywhere" in (broken.message or "")

    assert valid.passed is True, valid.message
    assert "aborted" not in (valid.message or "").lower(), (
        "the valid check inherited the broken check's aborted transaction"
    )
    assert valid.actual_value == 0


def test_the_connection_stays_usable_after_a_failed_check(conn: Any) -> None:
    """A failed gate must not leave the caller holding a dead connection.

    The orchestrator keeps using this connection after the gates run, so an
    aborted transaction would otherwise surface far from the check that caused
    it.
    """
    QualityRunner(conn).run_all([_BROKEN])

    assert conn.execute(text("SELECT 1")).scalar() == 1


def test_order_does_not_change_a_checks_verdict(conn: Any) -> None:
    """The same check must report the same result wherever it sits.

    This is the property that actually broke: counts_battles passed alone and
    failed when it happened to run after a broken check.
    """
    alone = QualityRunner(conn).run_all([_VALID])[0]
    after_failure = QualityRunner(conn).run_all([_BROKEN, _VALID])[1]

    assert alone.passed == after_failure.passed
    assert alone.actual_value == after_failure.actual_value


def test_every_declared_sql_gate_executes(conn: Any) -> None:
    """No agent spec may ship SQL the database rejects.

    A gate against a table that does not exist cannot pass, cannot fail
    meaningfully, and reads as a data problem rather than the spec bug it is.
    """
    runner = QualityRunner(conn)
    broken: list[str] = []

    for path in sorted(glob.glob("agents/*.yaml")):
        stage = os.path.splitext(os.path.basename(path))[0]
        spec = load_agent_spec(stage)
        checks = [c for c in (spec.get("quality_checks") or []) if c.get("method") is None]

        for result in runner.run_all(checks):
            if "SQL execution failed" in (result.message or ""):
                first_line = (result.message or "").splitlines()[0]
                broken.append(f"{stage}.{result.name}: {first_line}")

    assert not broken, "agent specs declare SQL the database rejects:\n" + "\n".join(broken)


def test_severity_survives_a_sql_failure(conn: Any) -> None:
    """An ERROR-severity check that fails to execute must stay ERROR.

    Severity decides whether the pipeline halts, so a broken blocking gate must
    not be quietly downgraded to a warning by the failure path.
    """
    check = dict(_BROKEN, name="blocking_but_broken", severity="error")

    result = QualityRunner(conn).run_all([check])[0]

    assert result.passed is False
    assert result.severity is Severity.ERROR
