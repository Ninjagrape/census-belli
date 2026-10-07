"""Integration tests for the inspector against a real PostgreSQL.

Seeds a small synthetic dataset through the read-write engine, then drives
every query through ``read_only_engine`` and smoke-tests each dashboard page
with Streamlit's ``AppTest``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError
from streamlit.testing.v1 import AppTest

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.inspector import queries
from pipeline.inspector.connection import read_only_engine
from pipeline.inspector.gates import GATE_COLUMNS, run_all_gates, run_stage_gates
from tests.integration.inspector_seed import BATTLES, seed_database, write_data_root

APP = Path(__file__).resolve().parents[2] / "scripts" / "inspector_app.py"
PAGE_PATHS = [
    "inspector_pages/overview.py",
    "inspector_pages/crawl.py",
    "inspector_pages/extract.py",
    "inspector_pages/resolve.py",
    "inspector_pages/reconcile.py",
    "inspector_pages/classify.py",
    "inspector_pages/impute.py",
    "inspector_pages/battle.py",
    "inspector_pages/model.py",
]


def _database_available() -> tuple[bool, str]:
    """Check whether DATABASE_URL points at a reachable database."""
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
    """Provide a read-write engine on a freshly applied schema."""
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(f"Refusing to reset database {db_name!r}.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def ids(engine: Engine) -> dict[str, int]:
    """Seed the synthetic dataset once for the module."""
    return seed_database(engine)


@pytest.fixture(scope="module")
def ro_engine(engine: Engine, ids: dict[str, int]) -> Iterator[Engine]:
    """A read-only engine on the seeded database."""
    eng = read_only_engine(engine.url.render_as_string(hide_password=False))
    yield eng
    eng.dispose()


@pytest.fixture
def conn(ro_engine: Engine) -> Iterator[Connection]:
    """A read-only connection."""
    with ro_engine.connect() as connection:
        yield connection


def test_read_only_engine_rejects_writes(conn: Connection) -> None:
    with pytest.raises(DBAPIError, match="read-only"):
        conn.execute(text("INSERT INTO wars (name) VALUES ('x')"))


def test_table_counts_and_runs(conn: Connection) -> None:
    counts = queries.table_counts(conn).set_index("table_name")["n_rows"]
    assert counts["battles"] == len(BATTLES)
    assert counts["battle_sides"] == 6
    assert counts["troop_reports"] == 6

    latest = queries.latest_model_runs(conn).set_index("model_type")
    assert set(latest.index) == {"reconcile", "imputation"}
    assert "rhat_max" in latest.loc["reconcile", "diagnostics"]
    assert len(queries.model_runs(conn)) == 2
    assert queries.skill_estimates(conn).empty


def test_crawl_queries(conn: Connection) -> None:
    status = queries.crawl_status_counts(conn)
    assert set(status["http_status"]) == {"200", "404"}
    errors = queries.crawl_errors(conn)
    assert list(errors["errors"]) == ["not found"]
    assert queries.crawl_over_time(conn)["n_fetches"].sum() == 2


def test_extract_queries_include_bc_battles(conn: Connection) -> None:
    century = queries.battles_by_century(conn)
    assert century["n"].sum() == len(BATTLES)
    assert century["century_start"].min() < 0
    ranged = queries.battles_by_century(conn, (1800, 1900))
    assert list(ranged["century_start"]) == [1800]

    nulls = queries.null_fraction_battles(conn).set_index("column_name")
    assert nulls.loc["name", "null_fraction"] == 0.0
    assert nulls.loc["wikidata_id", "null_fraction"] == 1.0
    side_nulls = queries.null_fraction_battle_sides(conn).set_index("column_name")
    assert side_nulls.loc["est_troops_total", "null_fraction"] == 0.0

    breakdown = queries.troop_report_breakdown(conn)
    assert breakdown["n"].sum() == 6
    assert set(breakdown["scope"]) == {"engaged", "unknown"}

    llm = queries.llm_summary(conn).set_index("stage")
    assert llm.loc["extract", "prompt_tokens"] == 100
    assert llm.loc["extract", "cost_usd"] == pytest.approx(0.0012)
    assert llm.loc["classify", "status"] == "api_error"


def test_resolve_queries(conn: Connection) -> None:
    generals = queries.generals_summary(conn).set_index("canonical_name")
    assert generals.loc["Miltiades", "n_aliases"] == 1
    assert generals.loc["Miltiades", "n_battles"] == 1
    missing = queries.generals_missing_wikidata(conn)
    assert list(missing["canonical_name"]) == ["Wellington"]
    dist = queries.commanders_per_side(conn).set_index("n_commanders")["n_sides"]
    assert dist.to_dict() == {0: 3, 1: 3}


def test_reconcile_queries(conn: Connection) -> None:
    sides = queries.reconcile_sides(conn)
    assert len(sides) == 6
    assert (sides["est_troops_neutral"] > 0).all()
    assert set(sides["est_troops_method"]) == {"source_disagreement"}

    reports = queries.troop_reports_detail(conn)
    assert set(reports["source_title"]) == {"Herodotus, Histories", "Wikipedia infobox"}
    assert reports["extracted_context"].notna().all()

    bias = queries.source_bias(conn).set_index("title")
    assert bias.loc["Herodotus, Histories", "troop_bias_mu"] == pytest.approx(-0.8)
    assert bias.loc["Wikipedia infobox", "bias_n_troop_reports"] == 5

    casualties = queries.casualty_estimates(conn)
    assert casualties["est_casualties"].notna().all()
    assert list(queries.casualty_reports_detail(conn)["reported_value"]) == [192.0]


def test_classify_queries(conn: Connection) -> None:
    roles = queries.role_counts(conn)
    assert roles["n"].sum() == 3
    weights = queries.attribution_weights(conn)
    assert (weights["side_weight_sum"] == 1.0).all()
    assert (queries.side_weight_sums(conn)["weight_sum"] == 1.0).all()

    review = queries.review_queue(conn)
    assert list(review["canonical_name"]) == ["Wellington"]

    matrix = queries.missing_data_matrix(conn).set_index(["field_name", "missingness_class"])
    assert matrix.loc[("troop_total", "mnar"), "n"] == 1

    defender = queries.defender_coverage(conn).set_index("is_defender")["n_sides"]
    assert defender.to_dict() == {"attacker": 3, "defender": 3}


def test_impute_rows(conn: Connection) -> None:
    imputed = queries.imputed_rows(conn)
    assert len(imputed) == 1
    assert imputed.loc[0, "imputation_method"] == "era_polity_prior"
    assert imputed.loc[0, "imputed_lo"] < imputed.loc[0, "imputed_value"]


def test_battle_search_and_drilldown(conn: Connection, ids: dict[str, int]) -> None:
    found = queries.search_battles(conn, "marathon")
    assert list(found["battle_id"]) == [ids["b0"]]
    assert queries.search_battles(conn, "100%").empty
    assert len(queries.search_battles(conn, "Battle of")) == len(BATTLES)

    bounds = queries.battle_year_bounds(conn).iloc[0]
    assert bounds["min_year"] == -489
    assert bounds["max_year"] == 1815

    battle = queries.battle_row(conn, ids["b0"]).iloc[0]
    assert battle["name"] == "Battle of Marathon"
    assert battle["year_astronomical"] == -489
    assert "BC" in battle["date_start"]

    marathon = ids["b0"]
    assert len(queries.battle_sides(conn, marathon)) == 2
    commanders = queries.battle_commanders(conn, marathon)
    assert list(commanders["general_name"]) == ["Miltiades"]
    assert "commanded" in commanders.loc[0, "role_evidence"]
    assert len(queries.battle_troop_reports(conn, marathon)) == 2
    assert len(queries.battle_casualty_reports(conn, marathon)) == 1
    assert queries.battle_missing_data(conn, ids["b2"])["field_name"].tolist() == [
        "troop_total",
        "outcome",
    ]
    assert list(queries.battle_llm_calls(conn, marathon)["status"]) == ["ok"]
    assert len(queries.battle_crawl_log(conn, marathon)) == 2


def test_gates_return_a_frame_and_survive_errors(conn: Connection) -> None:
    frame = run_stage_gates(conn, "crawl")
    assert list(frame.columns) == GATE_COLUMNS
    assert not frame.empty

    broken = run_stage_gates(conn, "no_such_stage")
    assert broken.loc[0, "passed"] is False or not broken.loc[0, "passed"]
    assert "SpecError" in broken.loc[0, "message"]

    everything = run_all_gates(conn)
    assert set(everything["stage"]) >= {"crawl", "reconcile", "impute"}
    # The read-only connection must still be usable after failing checks.
    assert conn.execute(text("SELECT 1")).scalar() == 1


def _run_app(monkeypatch: pytest.MonkeyPatch, engine: Engine, root: Path) -> AppTest:
    """Build an AppTest pointed at the seeded database and data root."""
    st.cache_data.clear()
    monkeypatch.setenv("INSPECT_DATABASE_URL", engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("INSPECT_DATA_ROOT", str(root))
    return AppTest.from_file(str(APP), default_timeout=90)


def _exceptions(at: AppTest) -> list[Any]:
    return [e.value for e in at.exception]


@pytest.fixture(scope="module")
def data_root(ids: dict[str, int], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data root whose imputed rounds match the seeded sides."""
    root = tmp_path_factory.mktemp("data")
    keys = sorted(k for k in ids if len(k) == 3 and k[0] == "s" and k[1:].isdigit())
    write_data_root(root, [ids[k] for k in keys], [ids[f"b{k[1]}"] for k in keys])
    return root


@pytest.mark.parametrize("page", PAGE_PATHS)
def test_every_page_renders_without_exceptions(
    page: str, engine: Engine, ids: dict[str, int], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _run_app(monkeypatch, engine, data_root)
    at.run()
    assert not _exceptions(at)
    at.switch_page(page).run()
    assert not _exceptions(at), _exceptions(at)


def test_overview_gate_button_and_battle_page(
    engine: Engine, ids: dict[str, int], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _run_app(monkeypatch, engine, data_root)
    at.run()
    at.button[0].click().run()
    assert not _exceptions(at)
    assert len(at.dataframe) >= 3

    at.switch_page("inspector_pages/battle.py").run()
    at.text_input[-1].set_value("waterloo").run()
    assert not _exceptions(at)


def test_app_degrades_when_database_is_unreachable(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    st.cache_data.clear()
    monkeypatch.setenv("INSPECT_DATABASE_URL", "postgresql+psycopg://nobody:pw@127.0.0.1:1/none")
    monkeypatch.setenv("INSPECT_DATA_ROOT", str(data_root))
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    for page in PAGE_PATHS:
        at.switch_page(page).run()
        assert not _exceptions(at), (page, _exceptions(at))
    assert any("unreachable" in w.value for w in at.warning)
