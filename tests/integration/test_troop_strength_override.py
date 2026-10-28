"""
Integration test: the ``troop_strength`` override through extract and reconcile load.

Runs the real extract stage on a fixture article with an overrides file, then
checks the rows it wrote, the ``overrides_applied`` gate and a second run; then
reads them back through the reconcile loader to see that the side's other
totals are superseded, and that the span guard flags a disagreeing side unless
an override settled it.
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
from pipeline.quality import QualityRunner
from pipeline.reconcilers import store as reconcile_store
from pipeline.reconcilers.load import load_reports
from pipeline.reconcilers.records import ReconcileCounts
from pipeline.stages import extract
from pipeline.stages.base import StageContext

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "extract"
AGENTS = ROOT / "agents"
CITATION = "Hand citation, 1999, p. 1"
SIDE = "Kingdom of Varra"


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
    """Lay down one fixture battle (contested) that carries no strength figures."""
    articles = tmp_path / "raw" / "battles_html"
    articles.mkdir(parents=True)
    shutil.copy(FIXTURES / "contested_field.wikitext", articles / "contested.wikitext")
    return tmp_path / "raw"


def _entry(n: int, value: dict[str, Any], side: str = SIDE) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": "troop_strength",
        "target": {"battle": "contested", "side": side},
        "value": {k: v for k, v in value.items() if k != "scope"},
        "scope": value.get("scope", "deployed"),
        "source": CITATION,
        "note": "",
        "reviewed": {"by": "lee", "on": "2026-10-23"},
    }


POINT = _entry(1, {"point": 50_000, "scope": "deployed"})
RANGE = _entry(1, {"low": 40_000, "high": 60_000, "scope": "engaged"})


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


def _run(
    conn: Any, raw_root: Path, processed: Path, overrides: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", processed.parent / "llm_requests")
    extract.run(_spec(raw_root, processed, overrides), StageContext(db_conn=conn))


def _side_id(conn: Any, label: str = SIDE) -> int:
    return int(
        conn.execute(
            text(
                "SELECT s.side_id FROM battle_sides s JOIN battles b USING (battle_id) "
                "WHERE b.source_slug = 'contested' AND s.side_label = :label"
            ),
            {"label": label},
        ).scalar_one()
    )


def _manual_rows(conn: Any) -> list[tuple[Any, ...]]:
    rows = conn.execute(
        text(
            "SELECT s.side_label, t.reported_value, t.scope, t.is_lower_bound, t.is_upper_bound, "
            "t.is_estimate, t.branch::text, src.source_type::text, src.title, "
            "t.extracted_context "
            "FROM troop_reports t JOIN battle_sides s USING (side_id) "
            "JOIN sources src ON src.source_id = t.source_id "
            "WHERE t.extraction_method = 'manual' ORDER BY s.side_label, t.reported_value"
        )
    ).fetchall()
    return [tuple(r) for r in rows]


def _gate(spec: dict[str, Any], conn: Any) -> Any:
    (check,) = [c for c in spec["quality_checks"] if c["name"] == "overrides_applied"]
    return QualityRunner(conn, spec["params"]).run_one(check)


# ─── Extract ─────────────────────────────────────────────────────────────────


def test_a_point_entry_writes_one_cited_modern_row_and_a_rerun_adds_no_duplicate(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", POINT)
    processed = tmp_path / "processed"

    _run(conn, raw_root, processed, overrides, monkeypatch)

    (row,) = _manual_rows(conn)
    assert row[:9] == (
        SIDE,
        50_000,
        "engaged",
        False,
        False,
        True,
        "total",
        "manual_entry",
        CITATION,
    )
    assert "modern estimates" in row[9]
    gap = conn.execute(
        text(
            "SELECT count(*) FROM missing_data_log WHERE side_id = :side "
            "AND field_name = 'troop_total'"
        ),
        {"side": _side_id(conn)},
    ).scalar_one()
    assert gap == 0
    assert _gate(_spec(raw_root, processed, overrides), conn).passed is True
    report = json.loads((processed / "overrides_extract.json").read_text(encoding="utf-8"))
    assert report["applied"]["troop_strength"] == 1

    _run(conn, raw_root, processed, overrides, monkeypatch)

    assert _manual_rows(conn) == [row]
    assert (
        conn.execute(
            text("SELECT count(*) FROM sources WHERE source_type = 'manual_entry'")
        ).scalar_one()
        == 1
    )


def test_a_range_entry_writes_a_lower_and_upper_row(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", RANGE)

    _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)
    _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)

    low, high = _manual_rows(conn)
    assert (low[1], low[3], low[4]) == (40_000, True, False)
    assert (high[1], high[3], high[4]) == (60_000, False, True)
    assert low[9] == high[9] and low[8] == high[8] == CITATION


def test_removing_the_entry_withdraws_its_rows_and_a_rotted_entry_fails_the_gate(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    _run(conn, raw_root, processed, _write_overrides(tmp_path / "a.yaml", POINT), monkeypatch)
    assert len(_manual_rows(conn)) == 1

    empty = _write_overrides(tmp_path / "b.yaml")
    _run(conn, raw_root, processed, empty, monkeypatch)
    assert _manual_rows(conn) == []

    rotted = _write_overrides(
        tmp_path / "c.yaml", _entry(7, {"point": 1_000, "scope": "engaged"}, "Atlantis")
    )
    _run(conn, raw_root, processed, rotted, monkeypatch)
    result = _gate(_spec(raw_root, processed, rotted), conn)
    assert result.passed is False
    assert "ov-0007" in result.message and "not found" in result.message
    assert _manual_rows(conn) == []


# ─── Reconcile load ──────────────────────────────────────────────────────────


def _add_infobox_total(conn: Any, side_id: int, value: int, context: str = "") -> None:
    source = conn.execute(
        text(
            "INSERT INTO sources (source_type, url) VALUES ('wikipedia_infobox', :url) "
            "RETURNING source_id"
        ),
        {"url": f"https://example.org/{side_id}/{value}"},
    ).scalar_one()
    conn.execute(
        text(
            "INSERT INTO troop_reports (side_id, source_id, branch, reported_value, scope, "
            "is_estimate, is_upper_bound, is_lower_bound, extraction_method, extracted_context) "
            "VALUES (:side, :source, 'total', :value, 'engaged', FALSE, FALSE, FALSE, "
            "'infobox_parser', :context)"
        ),
        {"side": side_id, "source": source, "value": value, "context": context or None},
    )


def _load(conn: Any) -> tuple[list[Any], ReconcileCounts]:
    counts = ReconcileCounts()
    reports = load_reports(conn, ancient_cutoff_year=500, counts=counts)
    return [r for r in reports if r.quantity == "troops"], counts


def test_the_loader_supersedes_the_sides_other_totals_with_the_override(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", POINT)
    _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)
    side = _side_id(conn)
    _add_infobox_total(conn, side, 250_000)
    _add_infobox_total(conn, side, 80_000, "(Herodotus)")

    troops, counts = _load(conn)

    assert [(r.side_id, r.reported_value, r.claim_regime) for r in troops] == [
        (side, 50_000.0, "modern_scholarly")
    ]
    assert counts.excluded_superseded_by_modern == 2
    assert counts.troop_span_warnings == counts.troop_span_errors == 0


def test_a_range_override_supersedes_the_other_totals_too(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", RANGE)
    _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)
    side = _side_id(conn)
    _add_infobox_total(conn, side, 250_000)

    troops, counts = _load(conn)

    assert sorted((r.reported_value, r.is_lower_bound, r.is_upper_bound) for r in troops) == [
        (40_000.0, True, False),
        (60_000.0, False, True),
    ]
    assert counts.excluded_superseded_by_modern == 1


# ─── Span guard ──────────────────────────────────────────────────────────────


def _battle_with_side(conn: Any, slug: str, label: str) -> int:
    battle = conn.execute(
        text(
            "INSERT INTO battles (name, source_slug, date_start, date_precision) "
            "VALUES (:slug, :slug, DATE '1815-06-18', 'day') RETURNING battle_id"
        ),
        {"slug": slug},
    ).scalar_one()
    return int(
        conn.execute(
            text(
                "INSERT INTO battle_sides (battle_id, side_label) VALUES (:b, :label) "
                "RETURNING side_id"
            ),
            {"b": battle, "label": label},
        ).scalar_one()
    )


def test_the_guard_warns_and_errors_by_span_and_lists_the_offending_sides(conn: Any) -> None:
    close = _battle_with_side(conn, "close", "Red")
    warn = _battle_with_side(conn, "wide", "Blue")
    error = _battle_with_side(conn, "huge", "Green")
    for side, values in (
        (close, (10_000, 20_000)),
        (warn, (10_000, 30_000)),
        (error, (10_000, 100_000)),
    ):
        for value in values:
            _add_infobox_total(conn, side, value)

    _, counts = _load(conn)

    assert (counts.troop_span_warnings, counts.troop_span_errors) == (1, 1)
    found = {
        (f["battle"], f["side"], f["severity"], f["min"], f["max"]) for f in counts.span_findings
    }
    assert found == {
        ("wide", "Blue", "warning", 10_000.0, 30_000.0),
        ("huge", "Green", "error", 10_000.0, 100_000.0),
    }


def test_a_troop_strength_override_exempts_its_side_from_the_guard(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", POINT)
    _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)
    side = _side_id(conn)
    # Both are non-modern, so supersede removes them; the guard must still not
    # see a span on a side an override settled.
    _add_infobox_total(conn, side, 10_000)
    _add_infobox_total(conn, side, 900_000)
    other = _side_id(conn, "Duchy of Selm")
    _add_infobox_total(conn, other, 10_000)
    _add_infobox_total(conn, other, 900_000)

    _, counts = _load(conn)

    assert [(f["side"], f["severity"]) for f in counts.span_findings] == [
        ("Duchy of Selm", "error")
    ]


# ─── Quality check ───────────────────────────────────────────────────────────


def _quality_check(name: str) -> dict[str, Any]:
    raw = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))
    return dict(next(c for c in raw["quality_checks"] if c["name"] == name))


def _stamp_run(conn: Any, guard: dict[str, Any] | None) -> None:
    run_id = reconcile_store.start_model_run(
        conn, model_type="source_disagreement", run_name="span", config={}
    )
    reconcile_store.complete_model_run(
        conn, run_id, {} if guard is None else {"troop_span_guard": guard}
    )


@pytest.mark.parametrize(
    ("guard", "error_passes", "warning_passes"),
    [
        # No troop_span_guard key: the guard did not run, which must fail, not pass.
        (None, False, True),
        ({"n_warnings": 0, "n_errors": 0, "sides": []}, True, True),
        ({"n_warnings": 2, "n_errors": 0, "sides": []}, True, False),
        ({"n_warnings": 0, "n_errors": 1, "sides": []}, False, True),
    ],
)
def test_the_span_checks_read_the_latest_runs_diagnostics(
    conn: Any, guard: dict[str, Any] | None, error_passes: bool, warning_passes: bool
) -> None:
    conn.execute(text("DELETE FROM model_runs"))
    _stamp_run(conn, guard)

    error = QualityRunner(conn).run_one(_quality_check("troop_span_guard"))
    warning = QualityRunner(conn).run_one(_quality_check("troop_span_warnings"))

    assert "SQL execution failed" not in error.message, error
    assert error.passed is error_passes, error
    assert warning.passed is warning_passes, warning
    assert error.severity.value == "error" and warning.severity.value == "warning"
