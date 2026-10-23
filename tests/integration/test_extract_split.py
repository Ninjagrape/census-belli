"""
Integration test: a ``battle_split`` override through the extract stage and the database.

Runs the real stage on the real Jena-Auerstedt wikitext with a split entry,
twice, and checks that the two child battles are written identically each time
and that no row of the parent article survives, including rows a previous run
(and the stages after it) left on a parent that existed before the split.
"""

from __future__ import annotations

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
from pipeline.stages import extract
from pipeline.stages.base import StageContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"
PARENT = "Battle_of_Jena_Auerstedt-7e1600f782"
JENA = f"{PARENT}#jena"
AUERSTEDT = f"{PARENT}#auerstedt"


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
    """Lay down the real Jena-Auerstedt wikitext."""
    articles = tmp_path / "raw" / "battles_html"
    articles.mkdir(parents=True)
    shutil.copy(FIXTURES / "jena_auerstedt.wikitext", articles / f"{PARENT}.wikitext")
    return tmp_path / "raw"


def _split_entry(n: int = 1) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": "battle_split",
        "target": {"battle": PARENT},
        "value": [
            {"key": "jena", "name": "Battle of Jena", "heading": "Jena"},
            {"key": "auerstedt", "name": "Battle of Auerstedt", "heading": "Auerstedt"},
        ],
        "source": "Wikipedia, infobox",
        "note": "",
        "reviewed": {"by": "lee", "on": "2026-10-22"},
    }


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


def _snapshot(conn: Any) -> dict[str, Any]:
    """Everything the run wrote for the article, without database ids."""
    battles = conn.execute(
        text(
            "SELECT source_slug, name, wikidata_id, date_start::text FROM battles "
            "WHERE source_slug LIKE :like ORDER BY source_slug"
        ),
        {"like": f"{PARENT}%"},
    ).fetchall()
    sides = conn.execute(
        text(
            "SELECT b.source_slug, s.side_label, s.outcome::text, s.polity FROM battle_sides s "
            "JOIN battles b USING (battle_id) WHERE b.source_slug LIKE :like "
            "ORDER BY b.source_slug, s.side_label"
        ),
        {"like": f"{PARENT}%"},
    ).fetchall()
    troops = conn.execute(
        text(
            "SELECT b.source_slug, s.side_label, t.reported_value, t.branch::text, t.scope "
            "FROM troop_reports t JOIN battle_sides s USING (side_id) "
            "JOIN battles b ON b.battle_id = s.battle_id WHERE b.source_slug LIKE :like "
            "ORDER BY b.source_slug, s.side_label, t.reported_value, t.branch::text"
        ),
        {"like": f"{PARENT}%"},
    ).fetchall()
    casualties = conn.execute(
        text(
            "SELECT b.source_slug, s.side_label, c.reported_value FROM casualty_reports c "
            "JOIN battle_sides s USING (side_id) JOIN battles b ON b.battle_id = s.battle_id "
            "WHERE b.source_slug LIKE :like ORDER BY 1, 2, 3"
        ),
        {"like": f"{PARENT}%"},
    ).fetchall()
    missing = conn.execute(
        text(
            "SELECT b.source_slug, m.field_name, m.side_id IS NOT NULL FROM missing_data_log m "
            "JOIN battles b USING (battle_id) WHERE b.source_slug LIKE :like ORDER BY 1, 2, 3"
        ),
        {"like": f"{PARENT}%"},
    ).fetchall()
    return {
        name: [tuple(r) for r in rows]
        for name, rows in {
            "battles": battles,
            "sides": sides,
            "troops": troops,
            "casualties": casualties,
            "missing": missing,
        }.items()
    }


def _seed_parent(conn: Any) -> int:
    """Leave the parent as a pre-split run and the later stages would have."""
    battle_id = int(
        conn.execute(
            text(
                "INSERT INTO battles (name, source_slug, wikidata_id) "
                "VALUES ('Battle of Jena-Auerstedt', :slug, 'Q1234') RETURNING battle_id"
            ),
            {"slug": PARENT},
        ).scalar_one()
    )
    source = int(
        conn.execute(
            text(
                "INSERT INTO sources (source_type, title) "
                "VALUES (CAST('wikipedia_infobox' AS source_type), 't') RETURNING source_id"
            )
        ).scalar_one()
    )
    general = int(
        conn.execute(
            text("INSERT INTO generals (canonical_name) VALUES ('Zombie') RETURNING general_id")
        ).scalar_one()
    )
    for label in ("French Empire", "Prussia"):
        side = int(
            conn.execute(
                text(
                    "INSERT INTO battle_sides (battle_id, side_label) "
                    "VALUES (:b, :l) RETURNING side_id"
                ),
                {"b": battle_id, "l": label},
            ).scalar_one()
        )
        conn.execute(
            text(
                "INSERT INTO troop_reports (side_id, source_id, reported_value) "
                "VALUES (:s, :src, 66000)"
            ),
            {"s": side, "src": source},
        )
        conn.execute(
            text(
                "INSERT INTO casualty_reports (side_id, source_id, reported_value) "
                "VALUES (:s, :src, 12600)"
            ),
            {"s": side, "src": source},
        )
        conn.execute(
            text(
                "INSERT INTO battle_commanders (battle_id, side_id, general_id) VALUES (:b, :s, :g)"
            ),
            {"b": battle_id, "s": side, "g": general},
        )
        conn.execute(
            text(
                "INSERT INTO missing_data_log (battle_id, side_id, field_name, was_imputed) "
                "VALUES (:b, :s, 'troop_total', TRUE)"
            ),
            {"b": battle_id, "s": side},
        )
    conn.execute(
        text(
            "INSERT INTO llm_calls (stage, provider, model, input_hash, status, response_json, "
            "battle_id) VALUES ('extract', 'x', 'm', 'h', 'ok', '{}', :b)"
        ),
        {"b": battle_id},
    )
    return battle_id


def _parent_rows(conn: Any) -> dict[str, int]:
    queries = {
        "battles": "SELECT count(*) FROM battles WHERE source_slug = :slug",
        "sides": (
            "SELECT count(*) FROM battle_sides s JOIN battles b USING (battle_id) "
            "WHERE b.source_slug = :slug"
        ),
        "commanders": (
            "SELECT count(*) FROM battle_commanders c JOIN battles b USING (battle_id) "
            "WHERE b.source_slug = :slug"
        ),
        "troops": (
            "SELECT count(*) FROM troop_reports t JOIN battle_sides s USING (side_id) "
            "JOIN battles b ON b.battle_id = s.battle_id WHERE b.source_slug = :slug"
        ),
        "casualties": (
            "SELECT count(*) FROM casualty_reports c JOIN battle_sides s USING (side_id) "
            "JOIN battles b ON b.battle_id = s.battle_id WHERE b.source_slug = :slug"
        ),
        "missing": (
            "SELECT count(*) FROM missing_data_log m JOIN battles b USING (battle_id) "
            "WHERE b.source_slug = :slug"
        ),
    }
    return {
        name: int(conn.execute(text(sql), {"slug": PARENT}).scalar_one())
        for name, sql in queries.items()
    }


def test_the_split_writes_two_battles_and_is_idempotent(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _write_overrides(tmp_path / "overrides.yaml", _split_entry())
    processed = tmp_path / "processed"

    _run(conn, raw_root, processed, overrides, monkeypatch)
    first = _snapshot(conn)
    _run(conn, raw_root, processed, overrides, monkeypatch)
    second = _snapshot(conn)

    assert second == first
    assert [b[0] for b in first["battles"]] == [AUERSTEDT, JENA]
    assert all(b[2] is None and b[3] == "1806-10-14" for b in first["battles"])
    assert [(s[0], s[1]) for s in first["sides"]] == [
        (AUERSTEDT, "French Empire"),
        (AUERSTEDT, "Prussia"),
        (JENA, "French Empire"),
        (JENA, "Prussia"),
    ]
    troops = {(t[0], t[1], t[2]) for t in first["troops"]}
    assert (JENA, "French Empire", 40000.0) in troops
    assert (JENA, "French Empire", 96000.0) in troops
    assert (JENA, "Prussia", 46000.0) in troops
    assert (AUERSTEDT, "French Empire", 26000.0) in troops
    assert (AUERSTEDT, "Prussia", 64000.0) in troops
    assert not {t[2] for t in first["troops"]} & {66000.0, 110000.0}
    assert not {c[2] for c in first["casualties"]} & {12600.0, 41000.0}
    assert _parent_rows(conn) == dict.fromkeys(
        ("battles", "sides", "commanders", "troops", "casualties", "missing"), 0
    )


def test_a_parent_left_by_an_earlier_run_is_removed_with_everything_under_it(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_parent(conn)
    assert _parent_rows(conn)["commanders"] == 2
    overrides = _write_overrides(tmp_path / "overrides.yaml", _split_entry())

    _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)

    assert _parent_rows(conn) == dict.fromkeys(
        ("battles", "sides", "commanders", "troops", "casualties", "missing"), 0
    )
    assert (
        conn.execute(
            text("SELECT count(*) FROM llm_calls WHERE battle_id IS NOT NULL")
        ).scalar_one()
        == 0
    )
    assert conn.execute(text("SELECT count(*) FROM llm_calls")).scalar_one() == 1
    assert [b[0] for b in _snapshot(conn)["battles"]] == [AUERSTEDT, JENA]


def test_the_parent_is_kept_bare_when_a_model_run_points_at_it(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    battle_id = _seed_parent(conn)
    general = int(conn.execute(text("SELECT general_id FROM generals LIMIT 1")).scalar_one())
    run = int(
        conn.execute(
            text(
                "INSERT INTO model_runs (run_name, model_type, config) "
                "VALUES ('m', 'bradley_terry_hierarchical', '{}') RETURNING run_id"
            )
        ).scalar_one()
    )
    conn.execute(
        text(
            "INSERT INTO battle_war_details (run_id, battle_id, general_id, war_contribution) "
            "VALUES (:r, :b, :g, 0.1)"
        ),
        {"r": run, "b": battle_id, "g": general},
    )
    overrides = _write_overrides(tmp_path / "overrides.yaml", _split_entry())

    _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)

    rows = _parent_rows(conn)
    assert rows["battles"] == 1
    assert rows["sides"] == rows["commanders"] == rows["troops"] == rows["missing"] == 0


def test_the_overrides_gate_passes_with_the_split_and_fails_without_the_article(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    overrides = _write_overrides(tmp_path / "overrides.yaml", _split_entry())
    _run(conn, raw_root, processed, overrides, monkeypatch)
    spec = _spec(raw_root, processed, overrides)
    (check,) = [c for c in spec["quality_checks"] if c["name"] == "overrides_applied"]

    assert QualityRunner(conn, spec["params"]).run_one(check).passed is True

    renamed = _split_entry()
    renamed["target"] = {"battle": "Some_Other_Article-0000000000"}
    rotted = _write_overrides(tmp_path / "rotted.yaml", renamed)
    _run(conn, raw_root, processed, rotted, monkeypatch)
    rotted_spec = _spec(raw_root, processed, rotted)
    (rotted_check,) = [c for c in rotted_spec["quality_checks"] if c["name"] == "overrides_applied"]
    result = QualityRunner(conn, rotted_spec["params"]).run_one(rotted_check)

    assert result.passed is False
    assert "ov-0001" in result.message and "battle not found" in result.message


def test_a_failed_split_leaves_the_parent_untouched_and_the_gate_failing(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_parent(conn)
    before = _parent_rows(conn)

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("split blew up")

    monkeypatch.setattr(extract, "extract_split_battle", _boom)
    processed = tmp_path / "processed"
    overrides = _write_overrides(tmp_path / "overrides.yaml", _split_entry())

    _run(conn, raw_root, processed, overrides, monkeypatch)

    assert _parent_rows(conn) == before
    assert (
        conn.execute(text("SELECT count(*) FROM battles WHERE source_slug LIKE '%#%'")).scalar_one()
        == 0
    )
    spec = _spec(raw_root, processed, overrides)
    (check,) = [c for c in spec["quality_checks"] if c["name"] == "overrides_applied"]
    assert QualityRunner(conn, spec["params"]).run_one(check).passed is False


def test_a_failed_retirement_fails_the_stage_instead_of_writing_children_beside_the_parent(
    conn: Any, raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_parent(conn)
    before = _parent_rows(conn)

    def _boom(*_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError("delete failed")

    monkeypatch.setattr(extract, "retire_battle", _boom)
    overrides = _write_overrides(tmp_path / "overrides.yaml", _split_entry())

    with pytest.raises(extract.SplitRetireError, match="Jena_Auerstedt"):
        _run(conn, raw_root, tmp_path / "processed", overrides, monkeypatch)

    assert _parent_rows(conn) == before
    assert (
        conn.execute(text("SELECT count(*) FROM battles WHERE source_slug LIKE '%#%'")).scalar_one()
        == 0
    )
