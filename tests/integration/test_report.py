"""Integration tests for the report stage, against a real PostgreSQL.

No model is fitted. After ``seed_database`` the module fixture inserts an
evaluable ``bradley_terry_hierarchical`` run with ``general_skill_estimates``
and ``battle_war_details`` rows whose per-battle contributions sum to each
``total_war``, and each test writes the model-diagnostics and evaluation files
the stage reads under ``tmp_path``. Nothing under ``data/`` is touched.

Cases that change database state do so inside the ``conn`` fixture's
transaction and call the stage directly, because ``run_stage`` commits the
shared connection. The CLI path (``run_stage``) is used on the committed seed.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.config import load_agent_spec
from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.orchestrator import run_quality_checks, run_stage
from pipeline.quality import QualityCheckResult, QualityRunner, Severity
from pipeline.reporters.disclosures import STATIC_DISCLOSURES
from pipeline.reporters.markdown import _esc
from pipeline.stages import report as report_stage
from pipeline.stages.base import StageContext, check_conforms
from tests.integration.inspector_seed import seed_database


def _database_available() -> tuple[bool, str]:
    """Check whether DATABASE_URL points at a reachable database."""
    try:
        url = database_url()
    except DatabaseConfigError as exc:
        return False, str(exc)
    try:
        engine = get_engine(url, pool_pre_ping=False)
        with engine.connect() as probe:
            probe.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        return False, f"cannot connect: {type(exc).__name__}: {exc}"
    return True, ""


_AVAILABLE, _REASON = _database_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _AVAILABLE, reason=f"No database available ({_REASON}). Run `make db-up` first."
    ),
]

POOLED = "bradley_terry_hierarchical"
MIN_BATTLES = 3
SPEC_CHECKS = {
    "top_rankings_have_intervals",
    "export_non_empty",
    "no_orphaned_war_details",
    "war_decomposition_reconciles",
    "reported_run_converged",
    "exclusion_concentration",
    "rank_ties_are_resolved",
    "export_matches_db",
    "evaluation_current",
}
# The explorer page closes three script elements: the data block, plotly.js and
# the page script. Anything beyond that came from data.
EXPLORER_SCRIPT_CLOSES = 3
HOSTILE_NAME = "</script><b>x</b>"
EVAL_DISCLOSURE = "Orientation: positive skill favours the side with more force & fewer <losses>."
EXCLUSION_MESSAGE = "18 top general(s) under 50% credited; default_split share of sides 100%"
COUNTS = {
    "battles_in_scope": 5,
    "battles_included": 3,
    "battles_no_creditable_side": 1,
    "battles_no_outcome": 1,
    "sides_multi_commander_all": 4,
    "sides_default_split_all": 2,
}
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# (name, wikidata id, [(battle key, war_contribution)], stored rank)
# Hannibal has 2 battles and so stays under-determined while holding stored
# rank 4, which leaves a gap in the headline ranks that the stage must keep.
SPECS: list[tuple[str, str | None, list[tuple[str, float]], int]] = [
    ("Miltiades", "Q1", [("b0", 1.0), ("b1", 0.9), ("b2", 0.8)], 1),
    ("Octavian", "Q2", [("b0", 0.8), ("b1", 0.6), ("b2", 0.5)], 2),
    ("Wellington", None, [("b0", 0.7), ("b1", 0.4), ("b2", 0.3)], 3),
    ("Hannibal", "Q4", [("b0", 0.9), ("b1", 0.1)], 4),
    ("Scipio", "Q3", [("b0", 0.5), ("b1", 0.2), ("b2", -0.1)], 5),
    ("Caesar", "Q5", [("b2", 0.2)], 6),
]
HEADLINE_RANKS = [1, 2, 3, 5]


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """Provide an engine against a freshly applied schema."""
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if "corpus" in db_name or not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.fail(f"Refusing to reset database {db_name!r}: not a test database.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


def _scalar(conn: Connection, sql: str, params: dict[str, Any]) -> int:
    return int(conn.execute(text(sql), params).scalar_one())


def _insert_general(conn: Connection, name: str, wikidata_id: str | None) -> int:
    found = conn.execute(
        text("SELECT general_id FROM generals WHERE canonical_name = :n"), {"n": name}
    ).scalar()
    if found is not None:
        return int(found)
    return _scalar(
        conn,
        "INSERT INTO generals (canonical_name, wikidata_id) VALUES (:n, :q) RETURNING general_id",
        {"n": name, "q": wikidata_id},
    )


@pytest.fixture(scope="module")
def world(engine: Engine) -> dict[str, Any]:
    """Seed the base data, then add an evaluable run with consistent WAR rows."""
    ids = seed_database(engine)
    general_ids: dict[str, int] = {}
    war_by_name: dict[str, float] = {}
    with engine.begin() as conn:
        run_id = _scalar(
            conn,
            "INSERT INTO model_runs (run_name, model_type, config, started_at, completed_at, "
            "diagnostics) VALUES ('report_test_run', :t, CAST('{}' AS JSONB), now(), now(), "
            "CAST(:d AS JSONB)) RETURNING run_id",
            {
                "t": POOLED,
                "d": json.dumps({"converged": True, "estimates_written": True, "counts": COUNTS}),
            },
        )
        for name, qid, battles, rank in SPECS:
            gid = _insert_general(conn, name, qid)
            general_ids[name] = gid
            total = math.fsum(w for _, w in battles)
            war_by_name[name] = total
            conn.execute(
                text(
                    "INSERT INTO general_skill_estimates (run_id, general_id, skill_mean, "
                    "skill_median, skill_sd, skill_ci_lo, skill_ci_hi, skill_ci90_lo, "
                    "skill_ci90_hi, total_war, war_per_battle, war_sd, war_ci_lo, war_ci_hi, "
                    "p_above_next_rank, n_battles, rank) "
                    "VALUES (:r, :g, :m, :m, 0.5, :lo, :hi, :lo9, :hi9, :w, :wp, :sd, "
                    ":wlo, :whi, :p, :n, :k)"
                ),
                {
                    "r": run_id,
                    "g": gid,
                    "m": total / 2,
                    "lo": total / 2 - 1.0,
                    "hi": total / 2 + 1.0,
                    "lo9": total / 2 - 0.8,
                    "hi9": total / 2 + 0.8,
                    "w": total,
                    "wp": total / len(battles),
                    "sd": 0.3,
                    "wlo": total - 0.9,
                    "whi": total + 1.1,
                    "p": None if rank == len(SPECS) else 0.5 + 0.05 * rank,
                    "n": len(battles),
                    "k": rank,
                },
            )
            for key, war in battles:
                conn.execute(
                    text(
                        "INSERT INTO battle_war_details (run_id, battle_id, general_id, "
                        "war_contribution, win_prob_model, win_prob_baseline, actual_outcome, "
                        "attribution_weight) VALUES (:r, :b, :g, :w, 0.7, 0.5, 1.0, 1.0)"
                    ),
                    {"r": run_id, "b": ids[key], "g": gid, "w": war},
                )
    return {"run_id": run_id, "ids": ids, "general_ids": general_ids, "war": war_by_name}


@pytest.fixture
def conn(engine: Engine, world: dict[str, Any]) -> Iterator[Connection]:
    """Give each test a transaction, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


# ─── Input files ─────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_inputs(
    tmp: Path, world: dict[str, Any], *, evaluated_run: int | None = None, exclusion_ok: bool = True
) -> dict[str, Any]:
    """Write the model-diagnostics and evaluation files; return the spec overrides."""
    run_id: int = world["run_id"]
    diagnostics, evaluation = tmp / "diagnostics", tmp / "evaluation"
    gates = [
        {
            "name": "fmi_top_generals",
            "passed": True,
            "severity": "warning",
            "threshold": "",
            "actual": "modest",
            "message": "imputation uncertainty is modest",
        },
        {
            "name": "exclusion_concentration",
            "passed": exclusion_ok,
            "severity": "warning",
            "threshold": "",
            "actual": "ok" if exclusion_ok else EXCLUSION_MESSAGE,
            "message": "" if exclusion_ok else EXCLUSION_MESSAGE,
        },
    ]
    _write_json(
        diagnostics / f"run{run_id}" / "gate_report.json",
        {"estimates_written": True, "gates": gates, "run_id": run_id},
    )
    _write_csv(
        diagnostics / f"run{run_id}" / "rubin_per_general.csv",
        [
            "general_id",
            "name",
            "component_id",
            "n_battles",
            "effective_battles",
            "n_uncredited_battles",
            "first_year",
            "last_year",
        ],
        [
            [world["general_ids"][name], name, 1, len(battles), float(len(battles)), 1, -490, 1815]
            for name, _, battles, _ in SPECS
        ],
    )
    figure = evaluation / f"run{run_id}" / "figures" / "calibration.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(PNG_1X1)
    ranks = [[world["general_ids"][name], rank] for name, _, _, rank in SPECS]
    for variant in ("baseline", "observed_only"):
        _write_csv(
            evaluation / f"run{run_id}" / "sensitivity" / f"{variant}.csv",
            ["general_id", "rank"],
            ranks,
        )
    _write_json(
        evaluation / f"run{run_id}" / "evaluation_report.json",
        {
            "model_run_id": evaluated_run if evaluated_run is not None else run_id,
            "disclosures": [EVAL_DISCLOSURE],
            "cv": {"imputed": {"binary": {"accuracy": 0.7, "brier": 0.22, "n": 53}}},
            "gates": [{"name": "held_out_accuracy", "passed": True, "message": "ok"}],
            "sensitivity": {
                "variants": [
                    {
                        "name": "observed_only",
                        "status": "ok",
                        "message": None,
                        "vs_baseline": {"score": {"spearman_rho": 0.97, "top_overlap": 3}},
                    }
                ]
            },
            "figures": [f"data\\processed\\evaluation\\run{run_id}\\figures\\calibration.png"],
        },
    )
    return {
        "output_root": str(tmp / "report"),
        "diagnostics_root": str(diagnostics),
        "evaluation_root": str(evaluation),
        "report_path": str(tmp / "report" / "manifest.json"),
    }


def _execute(
    conn: Connection, tmp: Path, world: dict[str, Any], **inputs: Any
) -> tuple[dict[str, QualityCheckResult], Path]:
    """Write inputs, run the stage on ``conn`` and then the spec's checks."""
    overrides = _write_inputs(tmp, world, **inputs)
    spec = load_agent_spec("report", overrides)
    report_stage.run(spec, StageContext(db_conn=conn))
    results = {r.name: r for r in run_quality_checks(spec, db_conn=conn)}
    return results, Path(overrides["output_root"])


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_stage_satisfies_the_contract() -> None:
    check_conforms(report_stage)


def test_clean_run_passes_every_gate_and_writes_consistent_outputs(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    results, out = _execute(conn, tmp_path, world)

    assert set(results) == SPEC_CHECKS
    failed = {n: r.message for n, r in results.items() if not r.passed}
    assert not failed, failed
    assert [r for r in results.values() if r.severity is Severity.ERROR]

    db_headline = conn.execute(
        text("SELECT COUNT(*) FROM general_skill_estimates WHERE run_id = :r AND n_battles >= 3"),
        {"r": world["run_id"]},
    ).scalar_one()
    rows = _csv_rows(out / "rankings.csv")
    headline = [r for r in rows if r["headline"] == "true"]
    assert len(headline) == db_headline == len(HEADLINE_RANKS)
    assert [int(r["rank"]) for r in headline] == HEADLINE_RANKS
    assert all(r["rank"] == "" for r in rows if r["headline"] == "false")
    assert len(rows) == len(SPECS)

    payload = _load(out / "rankings.json")
    assert len(payload["generals"]) == len(SPECS)
    for general in payload["generals"]:
        contributions = math.fsum(b["expected_war"] for b in general["battles"])
        assert abs(contributions - general["total_war"]) < 1e-6, general["name"]
        assert len(general["battles"]) == general["n_battles"]

    manifest = _load(out / "manifest.json")
    assert manifest["provisional"] is False
    assert manifest["reasons"] == []
    assert manifest["model_run_id"] == manifest["evaluation_run_id"] == world["run_id"]
    assert manifest["counts"] == {
        "headline": len(HEADLINE_RANKS),
        "under_determined": len(SPECS) - len(HEADLINE_RANKS),
        "csv_rows": len(SPECS),
        "json_generals": len(SPECS),
    }
    assert {g["name"] for g in manifest["gates"]} == {"export_matches_db", "evaluation_current"}
    assert all(g["passed"] for g in manifest["gates"])


def test_stored_war_intervals_replace_the_normal_approximation(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    _, out = _execute(conn, tmp_path, world)
    payload = _load(out / "rankings.json")
    markdown = (out / "report.md").read_text(encoding="utf-8")
    page = (out / "explorer.html").read_text(encoding="utf-8")

    for general in payload["generals"]:
        if general["headline"]:
            assert general["war_interval_method"] == "posterior"
            assert general["war_ci_lo"] == pytest.approx(general["total_war"] - 0.9)
            assert general["war_ci_hi"] == pytest.approx(general["total_war"] + 1.1)
    first = next(g for g in payload["generals"] if g["rank"] == 1)
    assert first["p_above_next_rank"] == pytest.approx(0.55)
    assert "war_normal_approx_lo" not in first
    assert "normal approximation" not in payload["overlap_method"]
    assert "normal approximation" not in markdown
    assert "war_normal_approx" not in page
    assert "p_above_next_rank" in _csv_rows(out / "rankings.csv")[0]


def test_a_run_without_stored_war_intervals_falls_back_with_a_disclosure(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    conn.execute(
        text(
            "UPDATE general_skill_estimates SET war_ci_lo = NULL, war_ci_hi = NULL, "
            "p_above_next_rank = NULL WHERE run_id = :r"
        ),
        {"r": world["run_id"]},
    )

    results, out = _execute(conn, tmp_path, world)

    payload = _load(out / "rankings.json")
    headline = [g for g in payload["generals"] if g["headline"]]
    assert {g["war_interval_method"] for g in headline} == {"normal_approx"}
    assert all(g["p_above_next_rank"] is None for g in headline)
    assert any("normal approximation" in d for d in payload["disclosures"])
    assert "(approx.)" in (out / "report.md").read_text(encoding="utf-8")
    assert all(r.passed for r in results.values() if r.severity is Severity.ERROR)


def test_every_disclosure_reaches_report_markdown_and_explorer(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    _, out = _execute(conn, tmp_path, world)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    page = (out / "explorer.html").read_text(encoding="utf-8")
    payload = _load(out / "rankings.json")

    expected = [*STATIC_DISCLOSURES, EVAL_DISCLOSURE]
    assert all(d in payload["disclosures"] for d in expected)
    assert any(
        d.startswith("The model credits 3 of 5 in-scope battles") for d in payload["disclosures"]
    )
    assert any(d.startswith("2 of 4 multi-commander sides") for d in payload["disclosures"])
    for disclosure in payload["disclosures"]:
        assert _esc(disclosure) in markdown, disclosure
        assert html.escape(disclosure, quote=False) in page, disclosure
    assert "PROVISIONAL" not in markdown
    assert "PROVISIONAL" not in page
    assert "figures/calibration.png" in markdown
    assert "\\" not in markdown.split("## Figures")[1].split("## Disclosures")[0]
    assert (out / "figures" / "calibration.png").read_bytes() == PNG_1X1
    assert (out / "figures" / "forest_top.png").is_file()

    ranks = payload["sensitivity"]["ranks"]
    assert set(ranks) == {"baseline", "observed_only"}
    assert ranks["observed_only"][str(world["general_ids"]["Miltiades"])] == 1


def test_evaluation_report_for_another_run_makes_outputs_provisional(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    results, out = _execute(conn, tmp_path, world, evaluated_run=world["run_id"] + 1)

    assert not results["evaluation_current"].passed
    assert "stale evaluation report" in results["evaluation_current"].message
    manifest = _load(out / "manifest.json")
    assert manifest["provisional"] is True
    assert manifest["evaluation_run_id"] is None
    assert any("stale evaluation report" in r for r in manifest["reasons"])
    evaluation_gate = next(g for g in manifest["gates"] if g["name"] == "evaluation_current")
    assert evaluation_gate["passed"] is False
    for name in ("rankings.csv", "rankings.json", "explorer.html", "report.md"):
        assert (out / name).is_file(), name
    assert "PROVISIONAL" in (out / "report.md").read_text(encoding="utf-8")
    assert _load(out / "rankings.json")["sensitivity"] is None
    assert results["export_matches_db"].passed, (
        "only the evaluation gate should fail on a stale evaluation"
    )


def test_failed_exclusion_concentration_makes_outputs_provisional(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    results, out = _execute(conn, tmp_path, world, exclusion_ok=False)

    gate = results["exclusion_concentration"]
    assert not gate.passed
    assert gate.severity is Severity.ERROR
    assert "18 top general(s)" in gate.message
    manifest = _load(out / "manifest.json")
    assert manifest["provisional"] is True
    assert any(
        r.startswith("model gate exclusion_concentration failed") and "18 top general(s)" in r
        for r in manifest["reasons"]
    )
    assert len(manifest["reasons"]) == 1
    assert (out / "rankings.csv").is_file()
    assert (out / "explorer.html").is_file()
    assert "PROVISIONAL" in (out / "explorer.html").read_text(encoding="utf-8")
    assert {n for n, r in results.items() if not r.passed} == {"exclusion_concentration"}


def test_no_evaluable_run_writes_nothing_and_fails_the_converged_gate(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    conn.execute(
        text(
            "UPDATE model_runs SET diagnostics = jsonb_set(diagnostics, '{converged}', "
            "'false'::jsonb) WHERE run_id = :r"
        ),
        {"r": world["run_id"]},
    )

    results, out = _execute(conn, tmp_path, world)

    assert not results["reported_run_converged"].passed
    assert results["reported_run_converged"].severity is Severity.ERROR
    assert not out.exists() or not any(out.iterdir())
    assert not Path(str(out / "manifest.json")).exists()


def test_hostile_general_name_cannot_close_the_explorer_script(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    conn.execute(
        text("UPDATE generals SET canonical_name = :n WHERE canonical_name = 'Caesar'"),
        {"n": HOSTILE_NAME},
    )

    _, out = _execute(conn, tmp_path, world)
    page = (out / "explorer.html").read_text(encoding="utf-8")
    markdown = (out / "report.md").read_text(encoding="utf-8")

    assert page.count("</script>") == EXPLORER_SCRIPT_CLOSES
    assert HOSTILE_NAME not in page
    assert "\\u003c/script\\u003e\\u003cb\\u003ex\\u003c/b\\u003e" in page
    assert HOSTILE_NAME not in markdown
    assert "&lt;/script&gt;&lt;b&gt;x&lt;/b&gt;" in markdown
    names = [g["name"] for g in _load(out / "rankings.json")["generals"]]
    assert HOSTILE_NAME in names
    assert HOSTILE_NAME in {r["name"] for r in _csv_rows(out / "rankings.csv")}


def test_war_decomposition_gate_fails_when_a_total_is_corrupted(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    spec = load_agent_spec("report", _write_inputs(tmp_path, world))
    check = [c for c in spec["quality_checks"] if c["name"] == "war_decomposition_reconciles"]
    runner = QualityRunner(conn, spec["params"])

    assert runner.run_all(check)[0].passed

    conn.execute(
        text(
            "UPDATE general_skill_estimates SET total_war = total_war + 0.5 "
            "WHERE run_id = :r AND general_id = :g"
        ),
        {"r": world["run_id"], "g": world["general_ids"]["Octavian"]},
    )
    corrupted = runner.run_all(check)[0]

    assert not corrupted.passed
    assert corrupted.severity is Severity.ERROR
    assert corrupted.actual_value == 1


def test_cli_path_runs_the_spec_gates_over_the_overridden_directories(
    engine: Engine, world: dict[str, Any], tmp_path: Path
) -> None:
    overrides = _write_inputs(tmp_path, world)

    with engine.connect() as shared:
        result = run_stage("report", overrides, db_conn=shared)

    assert result.error is None
    assert {c.name for c in result.quality_checks} == SPEC_CHECKS
    by_name = {c.name: c for c in result.quality_checks}
    assert {n for n, c in by_name.items() if not c.passed} == set()
    manifest = _load(Path(overrides["output_root"]) / "manifest.json")
    assert manifest["model_run_id"] == world["run_id"]
    assert manifest["provisional"] is False
    assert by_name["export_matches_db"].passed
    assert by_name["evaluation_current"].passed


def test_cli_path_reads_the_diagnostics_root_it_was_given(
    engine: Engine, world: dict[str, Any], tmp_path: Path
) -> None:
    """The stage reads ``diagnostics_root``; the spec's model_gate must read the same one."""
    overrides = _write_inputs(tmp_path, world)

    with engine.connect() as shared:
        result = run_stage("report", overrides, db_conn=shared)

    by_name = {c.name: c for c in result.quality_checks}
    assert by_name["exclusion_concentration"].passed, by_name["exclusion_concentration"].message
    assert result.success
