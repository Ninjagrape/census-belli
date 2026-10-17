"""
Integration test: manual overrides through the extract stage and into the database.

Runs the real stage on two fixture articles with an overrides file named by the
``overrides_path`` spec param, then checks what reached ``battle_sides`` and
``battles``, that a second run keeps it, that removing an entry withdraws its
provenance, and that a rotted entry fails the ``overrides_applied`` gate.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Engine, text

from pipeline.config import load_agent_spec
from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.extractors import BattleExtraction, BattleFacts, SideExtraction
from pipeline.extractors.override_apply import apply_extract_overrides
from pipeline.overrides import OverrideReport, load_overrides
from pipeline.quality import QualityRunner
from pipeline.stages import extract
from pipeline.stages.base import StageContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"
CITATION = "Hand citation, 1999, p. 1"


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
    """Lay down an undecided battle (contested) and a BC siege (alesia)."""
    articles = tmp_path / "raw" / "battles_html"
    articles.mkdir(parents=True)
    shutil.copy(FIXTURES / "contested_field.wikitext", articles / "contested.wikitext")
    shutil.copy(FIXTURES / "alesia_siege.wikitext", articles / "alesia.wikitext")
    return tmp_path / "raw"


def _entry(n: int, kind: str, target: dict[str, str], value: Any) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": kind,
        "target": target,
        "value": value,
        "source": CITATION,
        "note": "",
        "reviewed": {"by": "lee", "on": "2026-10-17"},
    }


OUTCOME = _entry(
    1,
    "outcome",
    {"battle": "contested"},
    {"Kingdom of Varra": "pyrrhic_victory", "Duchy of Selm": "defeat"},
)
POLITY = _entry(2, "side_polity", {"battle": "contested", "side": "Duchy of Selm"}, "Selm")
BC_DATE = _entry(
    3,
    "battle_date",
    {"battle": "alesia"},
    {"start": "0052-09-20 BC", "end": "0052-10-03 BC", "precision": "day"},
)


def _write_overrides(path: Path, *entries: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump({"version": 1, "entries": list(entries)}), encoding="utf-8")
    return path


def _spec(raw_root: Path, processed: Path, overrides: Path) -> dict[str, Any]:
    return load_agent_spec(
        "extract",
        {
            "raw_root": str(raw_root),
            "processed_root": str(processed),
            "overrides_path": str(overrides),
        },
    )


def _sides(conn: Any) -> dict[str, tuple[str | None, str | None, str | None]]:
    rows = conn.execute(
        text(
            "SELECT s.side_label, s.outcome::text, s.polity, src.title "
            "FROM battle_sides s JOIN battles b ON b.battle_id = s.battle_id "
            "LEFT JOIN sources src ON src.source_id = s.outcome_source "
            "WHERE b.source_slug = 'contested'"
        )
    ).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def _alesia(conn: Any) -> tuple[str, str, str, int]:
    row = conn.execute(
        text(
            "SELECT date_start::text, date_end::text, date_precision, year_astronomical "
            "FROM battles WHERE source_slug = 'alesia'"
        )
    ).fetchone()
    assert row is not None
    return row[0], row[1], row[2], int(row[3])


def _run(
    conn: Any, raw_root: Path, processed: Path, overrides: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", processed.parent / "llm_requests")
    extract.run(_spec(raw_root, processed, overrides), StageContext(db_conn=conn))


def test_overrides_reach_the_database_survive_a_rerun_and_are_withdrawn_with_the_entry(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", OUTCOME, POLITY, BC_DATE)
    processed = tmp_path / "processed"
    expected_sides = {
        "Kingdom of Varra": ("pyrrhic_victory", "Kingdom of Varra", CITATION),
        "Duchy of Selm": ("defeat", "Selm", CITATION),
    }

    # Act: first run.
    _run(conn, raw_root, processed, overrides, monkeypatch)

    # Assert: database and processed file agree on the corrected values.
    assert _sides(conn) == expected_sides
    assert _alesia(conn) == ("0052-09-20 BC", "0052-10-03 BC", "day", -51)
    assert (
        conn.execute(
            text("SELECT count(*) FROM sources WHERE source_type = 'manual_entry'")
        ).scalar_one()
        == 1
    )
    rows = {
        row["slug"]: row
        for row in (
            json.loads(line)
            for line in (processed / "battles.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert rows["contested"]["outcome_method"] == "manual"
    assert {s["label"]: (s["outcome"], s["polity"]) for s in rows["contested"]["sides"]} == {
        "Kingdom of Varra": ("pyrrhic_victory", "Kingdom of Varra"),
        "Duchy of Selm": ("defeat", "Selm"),
    }
    assert "override ov-0001 applied (outcome)" in rows["contested"]["notes"]
    assert rows["alesia"]["date_start"] == "0052-09-20 BC"
    gaps = conn.execute(
        text(
            "SELECT count(*) FROM missing_data_log m JOIN battles b USING (battle_id) "
            "WHERE b.source_slug = 'contested' AND m.field_name IN ('outcome', 'outcome_level')"
        )
    ).scalar_one()
    assert gaps == 0

    # Act and assert: a second run converges on the same rows.
    _run(conn, raw_root, processed, overrides, monkeypatch)
    assert _sides(conn) == expected_sides
    assert _alesia(conn) == ("0052-09-20 BC", "0052-10-03 BC", "day", -51)
    assert (
        conn.execute(
            text("SELECT count(*) FROM sources WHERE source_type = 'manual_entry'")
        ).scalar_one()
        == 1
    )

    # Act and assert: dropping the outcome entry withdraws outcome and provenance.
    overrides = _write_overrides(tmp_path / "overrides.yaml", POLITY, BC_DATE)
    _run(conn, raw_root, processed, overrides, monkeypatch)
    assert _sides(conn) == {
        "Kingdom of Varra": (None, "Kingdom of Varra", None),
        "Duchy of Selm": (None, "Selm", None),
    }


def _gate(spec: dict[str, Any], conn: Any) -> Any:
    (check,) = [c for c in spec["quality_checks"] if c["name"] == "overrides_applied"]
    return QualityRunner(conn, spec["params"]).run_one(check)


def test_the_overrides_gate_passes_a_clean_run_and_fails_a_rotted_entry(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    clean = _write_overrides(tmp_path / "overrides.yaml", OUTCOME, POLITY, BC_DATE)
    _run(conn, raw_root, processed, clean, monkeypatch)
    spec = _spec(raw_root, processed, clean)

    assert _gate(spec, conn).passed is True

    rotted = _write_overrides(
        tmp_path / "rotted.yaml",
        OUTCOME,
        _entry(9, "side_polity", {"battle": "renamed_slug", "side": "Duchy of Selm"}, "Selm"),
    )
    _run(conn, raw_root, processed, rotted, monkeypatch)
    result = _gate(_spec(raw_root, processed, rotted), conn)

    assert result.passed is False
    assert "ov-0009" in result.message and "battle not found" in result.message


def _duplicate(outcomes: tuple[str, str], date_start: str) -> BattleExtraction:
    """Build one of two records that resolve to the same ``battles`` row."""
    return BattleExtraction(
        slug="dup",
        name="Battle of Dup",
        facts=BattleFacts(name="Battle of Dup", date_start=date_start, date_precision="day"),
        sides=[
            SideExtraction(label="Red", outcome=outcomes[0]),
            SideExtraction(label="Blue", outcome=outcomes[1]),
        ],
        n_sources=1,
    )


DUP_OUTCOME = _entry(5, "outcome", {"battle": "dup"}, {"Red": "defeat", "Blue": "victory"})
DUP_DATE = _entry(
    6, "battle_date", {"battle": "dup"}, {"start": "1650-05-05", "end": None, "precision": "day"}
)


@pytest.mark.parametrize("override_on", [0, 1])
def test_an_override_survives_a_second_record_for_the_same_battle(
    conn: Any, tmp_path: Path, override_on: int
) -> None:
    records = [
        _duplicate(("victory", "defeat"), "1700-01-01"),
        _duplicate(("indecisive", "indecisive"), "1701-02-02"),
    ]
    path = _write_overrides(tmp_path / "overrides.yaml", DUP_OUTCOME, DUP_DATE)
    overrides = load_overrides(path)
    report = OverrideReport("extract", overrides, limited=False)
    applied = apply_extract_overrides(
        [records[override_on]], overrides.for_stage("extract"), report
    )

    extract._attach_outcome_provenance(conn, records, applied)
    extract._write_battles(conn, records)

    sides = conn.execute(
        text(
            "SELECT s.side_label, s.outcome::text, src.title FROM battle_sides s "
            "JOIN battles b USING (battle_id) "
            "LEFT JOIN sources src ON src.source_id = s.outcome_source "
            "WHERE b.source_slug = 'dup'"
        )
    ).fetchall()
    assert {r[0]: (r[1], r[2]) for r in sides} == {
        "Red": ("defeat", CITATION),
        "Blue": ("victory", CITATION),
    }
    dates = conn.execute(
        text("SELECT date_start::text, date_precision FROM battles WHERE source_slug = 'dup'")
    ).one()
    assert tuple(dates) == ("1650-05-05", "day")


def test_a_dry_run_leaves_an_existing_report_untouched(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", OUTCOME)
    processed = tmp_path / "processed"
    processed.mkdir()
    report_file = processed / "overrides_extract.json"
    report_file.write_text('{"sentinel": true}', encoding="utf-8")
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", tmp_path / "llm_requests")

    extract.run(_spec(raw_root, processed, overrides), StageContext(db_conn=conn, dry_run=True))

    assert report_file.read_text(encoding="utf-8") == '{"sentinel": true}'
