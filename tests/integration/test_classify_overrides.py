"""Classify manual overrides against a real schema: apply, idempotence, restore, gate."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Connection, Engine, text

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.quality import QualityRunner
from pipeline.stages import classify as classify_stage
from pipeline.stages.base import StageContext
from tests.integration.test_classify import (
    ACTIUM_SIDE,
    _load_spec,
    _seed_actium_llm_call,
    _seed_battle,
    _seed_commander,
    _seed_corpus,
    _seed_general,
    _seed_missing,
    _seed_side,
)

ACTIUM_SLUG = "battle_of_actium"


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

_ATTRIBUTION_YAML = """
  - id: ov-0002
    kind: attribution
    target: {{battle: {slug}, side: {side}}}
    value:
      - {{general: Q48174, role: field_commander, rank: 0, weight: 0.8}}
      - {{general: octavian, role: sovereign, rank: 1, weight: 0.2, reports_to: Q48174}}
    source: Plutarch, Life of Antony
    note: ''
    reviewed: {{by: lee, on: 2026-10-17}}
"""
_TYPE_YAML = """
  - id: ov-0001
    kind: battle_type
    target: {{battle: {slug}}}
    value: field
    source: Plutarch, Life of Antony
    note: ''
    reviewed: {{by: lee, on: 2026-10-17}}
"""


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(f"Refusing to reset database {db_name!r}.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _spec(tmp_path: Path, seed: dict[str, Any], entries_yaml: str) -> dict[str, Any]:
    overrides = tmp_path / "overrides.yaml"
    body = entries_yaml if entries_yaml.strip() else " []"
    overrides.write_text(f"version: 1\nentries:{body}", encoding="utf-8")
    spec = _load_spec(seed["raw_root"])
    spec["params"]["overrides_path"] = str(overrides)
    spec["params"]["processed_root"] = str(tmp_path / "processed")
    return spec


def _seed(conn: Connection, tmp_path: Path, spec_entries: str) -> tuple[dict[str, Any], dict]:
    seed = _seed_corpus(conn, tmp_path)
    conn.execute(
        text("UPDATE battles SET source_slug = :slug WHERE battle_id = :b"),
        {"slug": ACTIUM_SLUG, "b": seed["actium_id"]},
    )
    return seed, _spec(tmp_path, seed, spec_entries)


def _cache_llm_answer(conn: Connection, seed: dict[str, Any]) -> None:
    """Cache an LLM answer for the Octavian side, as an earlier run would have."""
    _seed_actium_llm_call(conn, _load_spec(seed["raw_root"]), seed["raw_root"])


def _run(conn: Connection, spec: dict[str, Any]) -> None:
    classify_stage.run(spec, context=StageContext(db_conn=conn))


def _commanders(conn: Connection) -> list[Any]:
    return conn.execute(
        text(
            "SELECT bc_id, command_role, hierarchy_rank, reports_to_bc_id, attribution_weight, "
            "attribution_method, role_evidence FROM battle_commanders ORDER BY bc_id"
        )
    ).fetchall()


def _gate(conn: Connection, spec: dict[str, Any]) -> Any:
    checks = [c for c in spec["quality_checks"] if c["name"] == "overrides_applied"]
    return QualityRunner(conn, spec["params"]).run_all(checks)[0]


def _entries(slug: str = ACTIUM_SLUG, side: str = ACTIUM_SIDE) -> str:
    return _TYPE_YAML.format(slug=slug) + _ATTRIBUTION_YAML.format(slug=slug, side=side)


def test_overrides_write_manual_rows_and_pass_the_gate(conn: Connection, tmp_path: Path) -> None:
    seed, spec = _seed(conn, tmp_path, _entries())
    _cache_llm_answer(conn, seed)

    _run(conn, spec)

    battle = conn.execute(
        text(
            "SELECT CAST(battle_type AS TEXT), battle_type_source FROM battles WHERE battle_id=:b"
        ),
        {"b": seed["actium_id"]},
    ).one()
    assert tuple(battle) == ("field", "manual")
    rows = {r.bc_id: r for r in _commanders(conn)}
    agrippa, octavian = rows[seed["agrippa_bc"]], rows[seed["octavian_bc"]]
    assert (agrippa.command_role, agrippa.hierarchy_rank, agrippa.attribution_weight) == (
        "field_commander",
        0,
        pytest.approx(0.8),
    )
    assert (octavian.command_role, octavian.hierarchy_rank, octavian.reports_to_bc_id) == (
        "sovereign",
        1,
        seed["agrippa_bc"],
    )
    assert agrippa.attribution_method == octavian.attribution_method == "manual"
    assert "ov-0002" in octavian.role_evidence
    observed = conn.execute(
        text("SELECT CAST(missingness_class AS TEXT) FROM missing_data_log WHERE log_id = :i"),
        {"i": seed["battle_type_log"]},
    ).scalar_one()
    assert observed == "observed"
    gate = _gate(conn, spec)
    assert gate.passed, gate.message


def test_second_run_is_idempotent_and_restores_a_reset_side(
    conn: Connection, tmp_path: Path
) -> None:
    seed, spec = _seed(conn, tmp_path, _entries())
    _cache_llm_answer(conn, seed)
    _run(conn, spec)
    first = _commanders(conn)

    _run(conn, spec)
    assert _commanders(conn) == first

    # What resolve does when it prunes a link: back to defaults.
    conn.execute(
        text(
            "UPDATE battle_commanders SET command_role = 'unknown', hierarchy_rank = NULL, "
            "reports_to_bc_id = NULL, attribution_weight = 0.5, attribution_method = 'equal', "
            "role_evidence = NULL WHERE side_id = :s"
        ),
        {"s": seed["octavian_side"]},
    )
    # And extract re-typing the battle.
    conn.execute(
        text(
            "UPDATE battles SET battle_type = 'naval', battle_type_source = 'extract' "
            "WHERE battle_id = :b"
        ),
        {"b": seed["actium_id"]},
    )

    _run(conn, spec)

    assert _commanders(conn) == first
    assert (
        conn.execute(
            text("SELECT battle_type_source FROM battles WHERE battle_id = :b"),
            {"b": seed["actium_id"]},
        ).scalar_one()
        == "manual"
    )


def test_rule_pass_never_reverts_a_manual_type(conn: Connection, tmp_path: Path) -> None:
    # The Actium article reads as naval; the manual value must win on every run,
    # and a manual 'unknown' must not be re-inferred either.
    seed, spec = _seed(conn, tmp_path, _TYPE_YAML.format(slug=ACTIUM_SLUG))
    _run(conn, spec)
    _run(conn, spec)
    assert (
        conn.execute(
            text("SELECT CAST(battle_type AS TEXT) FROM battles WHERE battle_id = :b"),
            {"b": seed["actium_id"]},
        ).scalar_one()
        == "field"
    )

    # With no override left for it, a manual row stays: nothing else selects it.
    spec = _spec(tmp_path, seed, "")
    _run(conn, spec)
    state = conn.execute(
        text(
            "SELECT CAST(battle_type AS TEXT), battle_type_source FROM battles WHERE battle_id=:b"
        ),
        {"b": seed["actium_id"]},
    ).one()
    assert tuple(state) == ("field", "manual")

    conn.execute(
        text("UPDATE battles SET battle_type = 'unknown' WHERE battle_id = :b"),
        {"b": seed["actium_id"]},
    )
    _run(conn, spec)
    assert (
        conn.execute(
            text("SELECT CAST(battle_type AS TEXT) FROM battles WHERE battle_id = :b"),
            {"b": seed["actium_id"]},
        ).scalar_one()
        == "unknown"
    )


def test_overridden_side_is_not_sent_to_the_llm(conn: Connection, tmp_path: Path) -> None:
    seed, spec = _seed(conn, tmp_path, _ATTRIBUTION_YAML.format(slug=ACTIUM_SLUG, side="Octavian"))
    # No cached answer: without the override this side would be written to the
    # offline batch file.
    pairs = classify_stage.build_pending_role_requests(conn, seed["raw_root"], spec)
    assert seed["octavian_side"] not in {side.side_id for side, _request in pairs}

    _run(conn, spec)
    assert _gate(conn, spec).passed


def test_unmatched_entry_fails_the_gate(conn: Connection, tmp_path: Path) -> None:
    seed, spec = _seed(conn, tmp_path, _entries(side="No Such Side"))
    _cache_llm_answer(conn, seed)

    _run(conn, spec)

    gate = _gate(conn, spec)
    assert not gate.passed
    assert "ov-0002" in gate.message
    # The matching battle_type entry is still applied.
    assert (
        conn.execute(
            text("SELECT battle_type_source FROM battles WHERE battle_id = :b"),
            {"b": seed["actium_id"]},
        ).scalar_one()
        == "manual"
    )


def test_a_side_without_commanders_does_not_crash_classify(
    conn: Connection, tmp_path: Path
) -> None:
    seed, spec = _seed(conn, tmp_path, _entries())
    _cache_llm_answer(conn, seed)
    empty_battle = _seed_battle(conn, "Battle of Nobody", "1700-01-01")
    empty_side = _seed_side(conn, empty_battle, "Ghosts")
    _seed_missing(conn, empty_battle, "commander_general_id", side_id=empty_side)
    # A resolve override that unlinked the only commander leaves exactly this.
    lone = _seed_general(conn, "Lone Commander")
    bc = _seed_commander(conn, empty_battle, empty_side, lone)
    conn.execute(text("DELETE FROM battle_commanders WHERE bc_id = :b"), {"b": bc})

    _run(conn, spec)
    _run(conn, spec)

    assert _gate(conn, spec).passed
    assert (
        conn.execute(
            text("SELECT COUNT(*) FROM battle_commanders WHERE side_id = :s"), {"s": empty_side}
        ).scalar_one()
        == 0
    )


def test_dry_run_writes_nothing_but_reports(conn: Connection, tmp_path: Path) -> None:
    seed, spec = _seed(conn, tmp_path, _entries())
    _cache_llm_answer(conn, seed)
    before = _commanders(conn)

    classify_stage.run(spec, context=StageContext(db_conn=conn, dry_run=True))

    assert _commanders(conn) == before
    assert (
        conn.execute(
            text("SELECT battle_type_source FROM battles WHERE battle_id = :b"),
            {"b": seed["actium_id"]},
        ).scalar_one()
        is None
    )
    assert _gate(conn, spec).passed


def test_yaml_declares_the_overrides_check() -> None:
    spec = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "agents" / "classify.yaml").read_text("utf-8")
    )
    [check] = [c for c in spec["quality_checks"] if c["name"] == "overrides_applied"]
    assert check["method"] == "overrides_gate"
    assert check["params"] == {"stage": "classify"}
    assert check["severity"] == "error"
