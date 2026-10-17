"""
The resolve stage applies manual overrides against a real database.

A full ``run`` over a tiny corpus (local candidate source, fixture entity JSON,
an overrides file in ``tmp_path``). What only a database shows: a wrong link is
replaced and not merely added to, the old ``battle_commanders`` row is pruned,
the side's attribution is reset for classify, a re-run changes nothing, and the
``overrides_applied`` gate reads the report the run left.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Connection, Engine, text

from pipeline.config import load_agent_spec
from pipeline.db import apply_schema, get_engine
from pipeline.quality import QualityRunner
from pipeline.stages.base import StageContext
from pipeline.stages.resolve import run

pytestmark = pytest.mark.integration

ZAMA = "Battle_of_Zama"
ILIPA = "Battle_of_Ilipa"


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """Provide an engine against a freshly applied schema."""
    eng = get_engine()
    name = (eng.url.database or "").lower()
    if "test" not in name and "general_war" not in name:
        eng.dispose()
        pytest.skip("Refusing to reset a database that does not look like a test database.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    """Give each test an empty corpus, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _person(qid: str, label: str, born: int, died: int) -> dict[str, Any]:
    """A Wikidata human, with historical (BC) years as the entity JSON writes them."""

    def claim(year: int) -> list[dict[str, Any]]:
        time = f"-{year:04d}-01-01T00:00:00Z"
        return [{"mainsnak": {"datavalue": {"value": {"time": time, "precision": 9}}}}]

    return {
        "entities": {
            qid: {
                "id": qid,
                "labels": {"en": {"value": label}},
                "descriptions": {"en": {"value": "Carthaginian or Roman commander"}},
                "claims": {
                    "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
                    "P569": claim(born),
                    "P570": claim(died),
                },
            }
        }
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _seed_battle(conn: Connection, slug: str, name: str, date: str, sides: list[str]) -> None:
    battle_id = conn.execute(
        text(
            "INSERT INTO battles (name, source_slug, date_start, date_precision) "
            "VALUES (:n, :s, CAST(:d AS DATE), 'day') RETURNING battle_id"
        ),
        {"n": name, "s": slug, "d": date},
    ).scalar_one()
    for side in sides:
        conn.execute(
            text("INSERT INTO battle_sides (battle_id, side_label, polity) VALUES (:b, :s, :s)"),
            {"b": battle_id, "s": side},
        )


class _Corpus:
    """The fixture corpus on disk, and the stage run over it."""

    def __init__(self, conn: Connection, root: Path) -> None:
        self.conn = conn
        self.processed = root / "processed"
        self.raw = root / "raw"
        self.overrides = root / "overrides.yaml"
        (self.raw / "wikidata").mkdir(parents=True)
        self.processed.mkdir()

        for qid, label, born, died in [
            ("Q101", "Scipio", 236, 183),
            ("Q7", "Publius Cornelius Scipio Africanus", 236, 183),
            ("Q8", "Mago Barca", 243, 203),
        ]:
            (self.raw / "wikidata" / f"{qid}.json").write_text(
                json.dumps(_person(qid, label, born, died)), encoding="utf-8"
            )

        _write_jsonl(
            self.processed / "battles.jsonl",
            [
                {
                    "slug": ZAMA,
                    "name": "Battle of Zama",
                    "date_start": "0202-10-19 BC",
                    "sides": [{"label": "Rome", "polity": "Rome"}, {"label": "Carthage"}],
                },
                {
                    "slug": ILIPA,
                    "name": "Battle of Ilipa",
                    "date_start": "0206-01-01 BC",
                    "sides": [{"label": "Rome", "polity": "Rome"}, {"label": "Carthage"}],
                },
            ],
        )
        _write_jsonl(
            self.processed / "commanders_raw.jsonl",
            [
                self._row(ZAMA, "Battle of Zama", "Rome", "Scipio"),
                self._row(ZAMA, "Battle of Zama", "Carthage", "Hannibal"),
                self._row(ILIPA, "Battle of Ilipa", "Carthage", "Mago"),
            ],
        )
        _seed_battle(conn, ZAMA, "Battle of Zama", "0202-10-19 BC", ["Rome", "Carthage"])
        _seed_battle(conn, ILIPA, "Battle of Ilipa", "0206-01-01 BC", ["Rome", "Carthage"])

    @staticmethod
    def _row(slug: str, battle: str, side: str, name: str) -> dict[str, Any]:
        return {
            "battle_slug": slug,
            "battle_name": battle,
            "side_label": side,
            "name": name,
            "apparent_role": "field_commander",
        }

    def spec(self) -> dict[str, Any]:
        return load_agent_spec(
            "resolve",
            {
                "candidate_source": "local",
                "processed_root": str(self.processed),
                "raw_root": str(self.raw),
                "overrides_path": str(self.overrides),
            },
        )

    def write_overrides(self, entries: list[dict[str, Any]]) -> None:
        self.overrides.write_text(
            yaml.safe_dump({"version": 1, "entries": entries}), encoding="utf-8"
        )

    def run(self) -> None:
        run(self.spec(), StageContext(db_conn=self.conn))

    def gate(self) -> Any:
        spec = self.spec()
        check = next(c for c in spec["quality_checks"] if c["name"] == "overrides_applied")
        return QualityRunner(self.conn, spec["params"]).run_one(check)

    def commanders(self) -> list[tuple[Any, ...]]:
        """Each link as (battle slug, side, wikidata id or name, method, weight)."""
        return [
            tuple(row)
            for row in self.conn.execute(
                text(
                    "SELECT b.source_slug, s.side_label, "
                    "COALESCE(g.wikidata_id, g.canonical_name), "
                    "bc.attribution_method, bc.attribution_weight "
                    "FROM battle_commanders bc "
                    "JOIN generals g USING (general_id) JOIN battles b USING (battle_id) "
                    "JOIN battle_sides s ON s.side_id = bc.side_id "
                    "ORDER BY b.source_slug, s.side_label, 3"
                )
            )
        ]

    def generals(self) -> set[str]:
        return {
            str(r[0])
            for r in self.conn.execute(
                text("SELECT COALESCE(wikidata_id, canonical_name) FROM generals")
            )
        }


def _entry(n: int, kind: str, target: dict[str, str], value: str) -> dict[str, Any]:
    return {
        "id": f"ov-{n:04d}",
        "kind": kind,
        "target": target,
        "value": value,
        "source": "Test source, p. 1",
        "note": "test",
        "reviewed": {"by": "lee", "on": "2026-10-17"},
    }


def _link(n: int, side: str, mention: str, value: str, slug: str = ZAMA) -> dict[str, Any]:
    return _entry(n, "commander_link", {"battle": slug, "side": side, "mention": mention}, value)


GOOD = [
    _link(1, "Rome", "Scipio", "Q7"),
    _link(2, "Carthage", "Hannibal", "none"),
    _entry(3, "general_merge", {"general": "Mago"}, "Q8"),
]


def test_overrides_replace_remove_and_merge_links(conn: Connection, tmp_path: Path) -> None:
    corpus = _Corpus(conn, tmp_path)
    corpus.write_overrides([])
    corpus.run()
    assert corpus.commanders() == [
        (ILIPA, "Carthage", "Mago", "equal", 1.0),
        (ZAMA, "Carthage", "Hannibal", "equal", 1.0),
        (ZAMA, "Rome", "Q101", "equal", 1.0),
    ]

    # What classify would have left behind on the sides the overrides change.
    conn.execute(
        text(
            "UPDATE battle_commanders SET attribution_method = 'llm', attribution_weight = 0.4, "
            "hierarchy_rank = 2"
        )
    )

    corpus.write_overrides(GOOD)
    corpus.run()

    assert corpus.commanders() == [
        (ILIPA, "Carthage", "Q8", "equal", 1.0),
        (ZAMA, "Rome", "Q7", "equal", 1.0),
    ]
    assert corpus.generals() == {"Q7", "Q8"}
    assert corpus.gate().passed

    log = [
        json.loads(line)
        for line in (corpus.processed / "resolution_log.jsonl").read_text("utf-8").splitlines()
    ]
    manual = {row["override_id"]: row for row in log if row.get("override_id")}
    assert set(manual) == {"ov-0001", "ov-0002", "ov-0003"}
    assert all(row["method"] == "manual" for row in manual.values())
    assert manual["ov-0001"]["wikidata_id"] == "Q7"
    assert manual["ov-0002"]["status"] == "unresolved"


def test_a_second_run_changes_nothing(conn: Connection, tmp_path: Path) -> None:
    corpus = _Corpus(conn, tmp_path)
    corpus.write_overrides(GOOD)
    corpus.run()
    first = corpus.commanders()
    generals = corpus.generals()

    corpus.run()

    assert corpus.commanders() == first
    assert corpus.generals() == generals
    assert corpus.gate().passed


def test_an_entry_naming_a_nonexistent_mention_fails_the_gate(
    conn: Connection, tmp_path: Path
) -> None:
    corpus = _Corpus(conn, tmp_path)
    corpus.write_overrides([*GOOD, _link(4, "Rome", "Nobody At All", "Q7")])
    corpus.run()

    result = corpus.gate()

    assert not result.passed
    assert "ov-0004" in result.message
    # The other entries still applied: one rotten entry does not undo the rest.
    assert (ZAMA, "Rome", "Q7", "equal", 1.0) in corpus.commanders()


def test_the_gate_fails_before_the_stage_has_run(conn: Connection, tmp_path: Path) -> None:
    corpus = _Corpus(conn, tmp_path)
    corpus.write_overrides(GOOD)
    assert not corpus.gate().passed
