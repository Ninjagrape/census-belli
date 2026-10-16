"""
Integration tests for the classify stage's database writes.

These need a real PostgreSQL instance, for the same reason
``tests/integration/test_resolve.py`` does: an enum cast, an ``ON CONFLICT``
clause that reads the row it is replacing, and a BC date ``datetime.date``
cannot hold. Without a reachable database every test skips rather than fails.

What is worth the cost of a live database, because each fails silently in a
way a fixture-only unit test would not show:

- **The offline LLM path is real.** A pre-seeded ``llm_calls`` row is looked
  up by the exact hash the stage computes from a live side's commanders and a
  real article excerpt -- not a hash asserted from memory.
- **Idempotency and the resolve/classify boundary.** Re-running classify, and
  re-running resolve's own upsert after classify, both have to leave the
  right rows alone -- docs/handover-log.md §12.3's rule, now also owed to classify's
  own writes.
- **The quality gates execute against the real schema.** §4.1 and §11.4 of
  docs/handover-log.md are both a gate whose SQL names something that turned out not
  to exist, which can only ever be caught by running it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import structlog.testing
import yaml
from sqlalchemy import Connection, Engine, text

from pipeline.crawlers.wikipedia import article_filename
from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.extractors import BattleExtraction, BattleFacts, MissingField, SideExtraction
from pipeline.extractors import write_battle as extract_write_battle
from pipeline.llm.base import request_hash
from pipeline.llm.factory import llm_params
from pipeline.quality import QualityRunner
from pipeline.resolvers import (
    Identity,
    Mention,
    MentionGroup,
    ResolveCounts,
    load_battle_index,
    load_side_index,
    write_identity,
)
from pipeline.resolvers.records import BattleContext
from pipeline.stages import classify as classify_stage
from pipeline.stages.base import StageContext, check_conforms

AGENTS = Path(__file__).resolve().parents[2] / "agents"

ACTIUM_NAME = "Battle of Actium"
ACTIUM_URL = "https://en.wikipedia.org/wiki/Battle_of_Actium"
ACTIUM_SIDE = "Octavian"
ACTIUM_OTHER_SIDE = "Antony and Cleopatra"
# 31 BC. Astronomical -30: there is a year zero, so the historical year is one
# further from the epoch than the astronomical one (docs/handover-log.md §4.5).
ACTIUM_DATE = "0031-09-02 BC"
ACTIUM_YEAR = -30

NO_ARTICLE_NAME = "Battle of Noarticle"
NO_ARTICLE_DATE = "1800-06-15"

_NAVAL_ARTICLE = """== Background ==
Octavian and Agrippa led the fleet against Antony and Cleopatra near Actium.

== Battle ==
The ships engaged in a naval battle off the coast of Actium. Agrippa's
squadron broke the enemy line while Octavian remained offshore.

[[Category:Naval battles of the Roman Republic]]
[[Category:31 BC]]
"""


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
def conn(engine: Engine) -> Iterator[Connection]:
    """Give each test an empty corpus, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _load_spec(raw_root: Path) -> dict[str, Any]:
    """Load ``agents/classify.yaml`` with params pointed at a fixture root.

    Args:
        raw_root: A ``tmp_path``-backed directory standing in for
            ``data/raw``; the real directory is never touched.

    Returns:
        The parsed spec, with ``params.raw_root`` and ``params.llm_mode``
        overridden and everything else -- prompts, gates -- untouched.
    """
    spec = yaml.safe_load((AGENTS / "classify.yaml").read_text(encoding="utf-8"))
    spec["params"]["raw_root"] = str(raw_root)
    spec["params"]["llm_mode"] = "offline"
    return spec


# ─── Fixture corpus ──────────────────────────────────────────────────────────


def _seed_battle(
    conn: Connection, name: str, date: str, *, wikipedia_url: str | None = None
) -> int:
    """Insert one battle, as the extract stage would."""
    return int(
        conn.execute(
            text(
                "INSERT INTO battles (name, wikipedia_url, date_start, date_precision) "
                "VALUES (:name, :url, CAST(:date AS DATE), 'day') RETURNING battle_id"
            ),
            {"name": name, "url": wikipedia_url, "date": date},
        ).scalar_one()
    )


def _seed_side(conn: Connection, battle_id: int, label: str) -> int:
    """Insert one battle side."""
    return int(
        conn.execute(
            text(
                "INSERT INTO battle_sides (battle_id, side_label, polity) "
                "VALUES (:battle_id, :label, 'Roman Republic') RETURNING side_id"
            ),
            {"battle_id": battle_id, "label": label},
        ).scalar_one()
    )


def _seed_general(conn: Connection, name: str, *, wikidata_id: str | None = None) -> int:
    """Insert one canonical general, as resolve would."""
    return int(
        conn.execute(
            text(
                "INSERT INTO generals (canonical_name, wikidata_id) "
                "VALUES (:name, :wikidata_id) RETURNING general_id"
            ),
            {"name": name, "wikidata_id": wikidata_id},
        ).scalar_one()
    )


def _seed_commander(conn: Connection, battle_id: int, side_id: int, general_id: int) -> int:
    """Insert one ``battle_commanders`` row with ``command_role = 'unknown'``."""
    return int(
        conn.execute(
            text(
                "INSERT INTO battle_commanders (battle_id, side_id, general_id, command_role) "
                "VALUES (:battle_id, :side_id, :general_id, 'unknown') RETURNING bc_id"
            ),
            {"battle_id": battle_id, "side_id": side_id, "general_id": general_id},
        ).scalar_one()
    )


def _seed_missing(
    conn: Connection,
    battle_id: int,
    field_name: str,
    *,
    side_id: int | None = None,
    missingness_class: str = "unclassified",
) -> int:
    """Insert one ``missing_data_log`` row."""
    return int(
        conn.execute(
            text(
                "INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class) "
                "VALUES (:battle_id, :side_id, :field_name, "
                "CAST(:missingness_class AS missingness_class)) RETURNING log_id"
            ),
            {
                "battle_id": battle_id,
                "side_id": side_id,
                "field_name": field_name,
                "missingness_class": missingness_class,
            },
        ).scalar_one()
    )


def _seed_corpus(conn: Connection, tmp_path: Path) -> dict[str, Any]:
    """Build the fixture corpus every test in this module reads.

    Args:
        conn: An open, transactional connection.
        tmp_path: pytest's per-test temp directory, standing in for
            ``data/raw``. The naval article fixture is written under
            ``tmp_path/battles_html/``, at the exact filename
            ``pipeline.classifiers.load.find_article_path`` will look for.

    Returns:
        Every id a test might need, plus ``raw_root``.
    """
    actium_id = _seed_battle(conn, ACTIUM_NAME, ACTIUM_DATE, wikipedia_url=ACTIUM_URL)
    octavian_side = _seed_side(conn, actium_id, ACTIUM_SIDE)
    antony_side = _seed_side(conn, actium_id, ACTIUM_OTHER_SIDE)

    octavian_general = _seed_general(conn, "Octavian")
    agrippa_general = _seed_general(conn, "Agrippa", wikidata_id="Q48174")
    antony_general = _seed_general(conn, "Mark Antony")

    octavian_bc = _seed_commander(conn, actium_id, octavian_side, octavian_general)
    agrippa_bc = _seed_commander(conn, actium_id, octavian_side, agrippa_general)
    antony_bc = _seed_commander(conn, actium_id, antony_side, antony_general)

    battle_type_log = _seed_missing(conn, actium_id, "battle_type")
    mnar_log = _seed_missing(
        conn,
        actium_id,
        "commander_general_id",
        side_id=octavian_side,
        missingness_class="mnar",
    )

    no_article_id = _seed_battle(conn, NO_ARTICLE_NAME, NO_ARTICLE_DATE)
    no_article_side = _seed_side(conn, no_article_id, "Side A")
    uncached_one = _seed_general(conn, "Uncached One")
    uncached_two = _seed_general(conn, "Uncached Two")
    uncached_bc_one = _seed_commander(conn, no_article_id, no_article_side, uncached_one)
    uncached_bc_two = _seed_commander(conn, no_article_id, no_article_side, uncached_two)
    no_article_battle_type_log = _seed_missing(conn, no_article_id, "battle_type")

    raw_root = tmp_path / "raw"
    article_dir = raw_root / "battles_html"
    article_dir.mkdir(parents=True)
    stem = Path(article_filename(ACTIUM_URL)).stem
    (article_dir / f"{stem}.wikitext").write_text(_NAVAL_ARTICLE, encoding="utf-8")

    return {
        "raw_root": raw_root,
        "actium_id": actium_id,
        "octavian_side": octavian_side,
        "antony_side": antony_side,
        "octavian_bc": octavian_bc,
        "agrippa_bc": agrippa_bc,
        "antony_bc": antony_bc,
        "battle_type_log": battle_type_log,
        "mnar_log": mnar_log,
        "no_article_id": no_article_id,
        "no_article_side": no_article_side,
        "uncached_bc_one": uncached_bc_one,
        "uncached_bc_two": uncached_bc_two,
        "no_article_battle_type_log": no_article_battle_type_log,
    }


def _seed_actium_llm_call(
    conn: Connection,
    spec: dict[str, Any],
    raw_root: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Pre-seed the cached answer for Actium's two-commander side.

    Builds the exact request the stage will build for the Octavian side --
    same connection, same spec, same raw root -- so the ``llm_calls`` row is
    keyed by the hash the stage really computes rather than one asserted from
    memory.

    Args:
        conn: An open, transactional connection, already carrying the seeded
            corpus.
        spec: The loaded, path-overridden spec.
        raw_root: The fixture's raw root.
        extra: Further top-level keys for the cached answer, such as ``is_defender``.
    """
    pairs = classify_stage.build_pending_role_requests(conn, raw_root, spec)
    octavian_side_requests = [
        (side, request) for side, request in pairs if side.side_label == ACTIUM_SIDE
    ]
    assert len(octavian_side_requests) == 1, "expected exactly one pending Octavian-side request"
    _side, request = octavian_side_requests[0]

    config = llm_params(spec)
    digest = request_hash(request, config["provider"], config["model"])

    response = {
        "classifications": [
            {
                "name": "Agrippa",
                "command_role": "field_commander",
                "hierarchy_rank": 0,
                "reports_to": None,
                "attribution_weight": 0.85,
                "confidence": 0.9,
                "reasoning": "Agrippa's squadron broke the enemy line tactically",
            },
            {
                "name": "Octavian",
                "command_role": "sovereign",
                "hierarchy_rank": 1,
                "reports_to": "Agrippa",
                "attribution_weight": 0.15,
                "confidence": 0.9,
                "reasoning": "Octavian remained offshore and did not direct the fleet",
            },
        ],
        "needs_review": False,
        "review_reason": None,
        **(extra or {}),
    }
    _insert_llm_call(conn, config, digest, response)


def _insert_llm_call(
    conn: Connection, config: dict[str, Any], digest: str, response: dict[str, Any]
) -> None:
    """Insert one completed ``llm_calls`` row keyed by ``digest``."""
    conn.execute(
        text(
            """
            INSERT INTO llm_calls (
                stage, provider, model, input_hash, status,
                prompt_tokens, completion_tokens, cost_usd, response_json
            ) VALUES (
                'classify', :provider, :model, :input_hash, 'ok',
                100, 50, 0.001, CAST(:response_json AS JSONB)
            )
            """
        ),
        {
            "provider": config["provider"],
            "model": config["model"],
            "input_hash": digest,
            "response_json": json.dumps(response),
        },
    )


def _row(conn: Connection, sql: str, **params: Any) -> Any:
    """Run a one-row query and return it, or None."""
    return conn.execute(text(sql), params).fetchone()


def _scalar(conn: Connection, sql: str, **params: Any) -> Any:
    """Run a one-value query."""
    return conn.execute(text(sql), params).scalar()


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_check_conforms_accepts_the_classify_stage_module() -> None:
    check_conforms(classify_stage)


def test_a_cached_llm_answer_is_applied_to_both_commanders(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    agrippa = _row(
        conn,
        "SELECT command_role, attribution_weight, attribution_method, reports_to_bc_id "
        "FROM battle_commanders WHERE bc_id = :bc_id",
        bc_id=seed["agrippa_bc"],
    )
    octavian = _row(
        conn,
        "SELECT command_role, attribution_weight, attribution_method, reports_to_bc_id "
        "FROM battle_commanders WHERE bc_id = :bc_id",
        bc_id=seed["octavian_bc"],
    )

    assert agrippa.command_role == "field_commander"
    assert octavian.command_role == "sovereign"
    assert agrippa.attribution_method == "llm"
    assert octavian.attribution_method == "llm"
    assert agrippa.attribution_weight == pytest.approx(0.85, abs=1e-9)
    assert octavian.attribution_weight == pytest.approx(0.15, abs=1e-9)
    assert agrippa.attribution_weight + octavian.attribution_weight == pytest.approx(1.0)

    assert agrippa.reports_to_bc_id is None
    assert octavian.reports_to_bc_id == seed["agrippa_bc"]
    same_side_bc_ids = {seed["agrippa_bc"], seed["octavian_bc"]}
    assert octavian.reports_to_bc_id in same_side_bc_ids


def test_an_uncached_side_stays_default_split_and_is_counted_awaiting_llm(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])  # unrelated to Side A

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    one = _row(
        conn,
        "SELECT command_role, attribution_method FROM battle_commanders WHERE bc_id = :bc_id",
        bc_id=seed["uncached_bc_one"],
    )
    two = _row(
        conn,
        "SELECT command_role, attribution_method FROM battle_commanders WHERE bc_id = :bc_id",
        bc_id=seed["uncached_bc_two"],
    )

    assert one.command_role == "unknown"
    assert two.command_role == "unknown"
    assert one.attribution_method == "default_split"
    assert two.attribution_method == "default_split"


def test_an_offline_run_writes_the_cache_misses_to_a_batch_file(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    config = llm_params(spec)
    expected = {
        request_hash(request, config["provider"], config["model"])
        for _side, request in classify_stage.build_pending_role_requests(
            conn, seed["raw_root"], spec
        )
    }

    outcome = classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert outcome is not None and outcome.batch_file is not None
    lines = [
        json.loads(raw)
        for raw in outcome.batch_file.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    hashes = {line["request_hash"] for line in lines}
    # The stage logs a troop_total gap while it runs, so its missingness request
    # does not exist before the first run; afterwards the export path must build
    # the identical one, or the answer imported between runs would never be found.
    assert expected < hashes
    extra = hashes - expected
    assert len(extra) == 1
    after = {
        request_hash(request, config["provider"], config["model"])
        for _context, request in classify_stage.build_pending_missingness_requests(conn, spec)
    }
    assert extra == after
    assert {line["agent_model"] for line in lines} == {"sonnet"}
    assert outcome.deferred_requests == len(hashes)


def test_the_offline_script_builds_the_same_request_hash_as_the_stage(
    conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])

    stage_pairs = classify_stage.build_pending_role_requests(conn, seed["raw_root"], spec)
    config = llm_params(spec)
    stage_hashes = {
        side.side_id: request_hash(request, config["provider"], config["model"])
        for side, request in stage_pairs
    }

    @contextmanager
    def _reuse_conn() -> Iterator[Connection]:
        yield conn

    # scripts.llm_offline._classify_requests opens its own connection via
    # pipeline.db.get_connection; patched to reuse this test's transactional
    # connection so it sees the uncommitted fixture rows, without needing a
    # second real connection or an early commit.
    monkeypatch.setattr("pipeline.db.get_connection", _reuse_conn)

    from scripts.llm_offline import _classify_requests

    script_requests = _classify_requests(spec, Path("data/processed"))
    script_hashes = {
        request_hash(request, config["provider"], config["model"]) for request in script_requests
    }

    assert set(stage_hashes.values()) == script_hashes
    assert script_hashes, "expected at least one pending request to export"


def test_a_single_commander_side_is_field_commander_with_full_weight(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    antony = _row(
        conn,
        "SELECT command_role, attribution_weight, attribution_method "
        "FROM battle_commanders WHERE bc_id = :bc_id",
        bc_id=seed["antony_bc"],
    )
    assert antony.command_role == "field_commander"
    assert antony.attribution_weight == pytest.approx(1.0)
    assert antony.attribution_method == "rule_single"


def test_resolves_upsert_does_not_undo_classifys_role(conn: Connection, tmp_path: Path) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    before = _scalar(
        conn,
        "SELECT command_role FROM battle_commanders WHERE bc_id = :bc_id",
        bc_id=seed["agrippa_bc"],
    )
    assert before == "field_commander"

    mention = Mention(
        battle_slug="Battle_of_Actium",
        battle_name=ACTIUM_NAME,
        side_label=ACTIUM_SIDE,
        name="Agrippa",
        apparent_role="field_commander",
        role_evidence="",
        context=BattleContext(
            slug="Battle_of_Actium", name=ACTIUM_NAME, year=ACTIUM_YEAR, date_text=ACTIUM_DATE
        ),
    )
    group = MentionGroup(
        key="agrippa",
        keys=("agrippa",),
        display_name="Agrippa",
        surface_forms=["Agrippa"],
        mentions=[mention],
        confidence=0.95,
        method="exact_wikidata",
    )
    identity = Identity(
        key="Q48174",
        canonical_name="Agrippa",
        qid="Q48174",
        confidence=0.95,
        method="exact_wikidata",
        groups=[group],
    )

    battles = load_battle_index(conn)
    sides = load_side_index(conn)
    write_identity(conn, identity, battles, sides, ResolveCounts(), {})

    after = _scalar(
        conn,
        "SELECT command_role FROM battle_commanders WHERE bc_id = :bc_id",
        bc_id=seed["agrippa_bc"],
    )
    assert after == "field_commander"


def test_a_naval_article_sets_battle_type_and_observes_the_missing_row(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    battle_type = _scalar(
        conn, "SELECT battle_type FROM battles WHERE battle_id = :id", id=seed["actium_id"]
    )
    assert battle_type == "naval"

    log_row = _row(
        conn,
        "SELECT missingness_class, notes FROM missing_data_log WHERE log_id = :id",
        id=seed["battle_type_log"],
    )
    assert log_row.missingness_class == "observed"
    assert log_row.notes is not None and log_row.notes.startswith("classify: ")


def test_a_battle_with_no_article_stays_unknown_and_its_row_becomes_mar(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    battle_type = _scalar(
        conn, "SELECT battle_type FROM battles WHERE battle_id = :id", id=seed["no_article_id"]
    )
    assert battle_type == "unknown"

    log_row = _row(
        conn,
        "SELECT missingness_class, notes FROM missing_data_log WHERE log_id = :id",
        id=seed["no_article_battle_type_log"],
    )
    assert log_row.missingness_class == "mar"
    assert log_row.notes is not None and log_row.notes.startswith("classify: ")


def test_missingness_never_overwrites_resolves_mnar(conn: Connection, tmp_path: Path) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    still_mnar = _scalar(
        conn,
        "SELECT missingness_class FROM missing_data_log WHERE log_id = :id",
        id=seed["mnar_log"],
    )
    assert still_mnar == "mnar"


def test_running_classify_twice_writes_identical_rows(conn: Connection, tmp_path: Path) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    def _snapshot() -> tuple[Any, ...]:
        commanders = conn.execute(
            text(
                "SELECT bc_id, command_role, hierarchy_rank, reports_to_bc_id, "
                "attribution_weight, attribution_method, confidence "
                "FROM battle_commanders ORDER BY bc_id"
            )
        ).fetchall()
        battles = conn.execute(
            text("SELECT battle_id, battle_type, fortified FROM battles ORDER BY battle_id")
        ).fetchall()
        missing = conn.execute(
            text("SELECT log_id, missingness_class, notes FROM missing_data_log ORDER BY log_id")
        ).fetchall()
        return tuple(commanders), tuple(battles), tuple(missing)

    # Identical from the very first run. A side the LLM classified on run 1
    # looks internally consistent on run 2; without the llm-preservation rule
    # in pipeline.stages.classify the fixed weight table would overwrite the
    # model's weights (0.85/0.15 became 0.867/0.133) and relabel them
    # rule_roles. That was the behaviour this test first recorded.
    classify_stage.run(spec, context=StageContext(db_conn=conn))
    first = _snapshot()

    classify_stage.run(spec, context=StageContext(db_conn=conn))
    second = _snapshot()

    assert first == second
    llm_rows = conn.execute(
        text("SELECT COUNT(*) FROM battle_commanders WHERE attribution_method = 'llm'")
    ).scalar_one()
    assert llm_rows >= 2, "the cached Actium answer should survive a re-run as 'llm'"


def test_all_three_classify_gates_execute_with_no_sql_error(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    results = {r.name: r for r in QualityRunner(conn).run_all(spec["quality_checks"])}

    assert set(results) == {
        "all_commanders_classified",
        "attribution_weights_sum",
        "battle_type_unknown_share",
        "missing_data_all_logged",
    }
    for name, result in results.items():
        assert "does not exist" not in (result.message or ""), f"{name}: {result.message}"
        print(f"{name}: passed={result.passed} value={result.actual_value}")


# ─── Troop gaps, reconciled presence and siege defenders ─────────────────────


def _seed_troop_gap_battle(conn: Connection) -> dict[str, int]:
    """Seed a 1750 battle: a winner with a reconciled estimate, and a loser
    whose only report reconcile found unusable, so it has no estimate."""
    battle_id = _seed_battle(conn, "Battle of Gapville", "1750-05-01")
    winner = _seed_side(conn, battle_id, "Winners")
    loser = _seed_side(conn, battle_id, "Losers")
    conn.execute(
        text(
            "UPDATE battle_sides SET est_troops_total = 20000, est_troops_method = 'x', "
            "outcome = 'victory' WHERE side_id = :s"
        ),
        {"s": winner},
    )
    conn.execute(
        text("UPDATE battle_sides SET outcome = 'defeat' WHERE side_id = :s"), {"s": loser}
    )
    source_id = conn.execute(
        text(
            "INSERT INTO sources (source_type, title) VALUES ('wikipedia_infobox', 't') "
            "RETURNING source_id"
        )
    ).scalar_one()
    conn.execute(
        text(
            "INSERT INTO troop_reports (side_id, source_id, reported_value, branch, "
            "extracted_context) VALUES (:s, :src, 3, 'total', '3 batteries')"
        ),
        {"s": loser, "src": source_id},
    )
    log_id = _seed_missing(conn, battle_id, "troop_total", side_id=loser)
    return {"battle_id": battle_id, "winner": winner, "loser": loser, "log_id": log_id}


def test_a_side_reconcile_left_without_an_estimate_is_treated_as_having_no_troops(
    conn: Connection, tmp_path: Path
) -> None:
    ids = _seed_troop_gap_battle(conn)
    spec = _load_spec(tmp_path / "raw")

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    # Raw reports exist for the loser, but reconcile gave no estimate; the
    # other side has one, so the losing-side-only-winner-known rule fires.
    assert (
        _scalar(
            conn,
            "SELECT missingness_class FROM missing_data_log WHERE log_id = :id",
            id=ids["log_id"],
        )
        == "mnar"
    )


def test_a_side_with_no_estimate_and_no_log_row_gets_one(conn: Connection, tmp_path: Path) -> None:
    battle_id = _seed_battle(conn, "Battle of Silence", "1750-05-01")
    side_id = _seed_side(conn, battle_id, "Nobody")
    spec = _load_spec(tmp_path / "raw")

    classify_stage.run(spec, context=StageContext(db_conn=conn))
    classify_stage.run(spec, context=StageContext(db_conn=conn))

    rows = conn.execute(
        text(
            "SELECT notes FROM missing_data_log WHERE side_id = :s AND field_name = 'troop_total'"
        ),
        {"s": side_id},
    ).fetchall()
    assert len(rows) == 1
    assert rows[0].notes.startswith("classify: ")
    results = {r.name: r for r in QualityRunner(conn).run_all(spec["quality_checks"])}
    assert results["missing_data_all_logged"].passed


def test_the_unlogged_troop_total_note_is_the_shared_constant(
    conn: Connection, tmp_path: Path
) -> None:
    from pipeline.classifiers.records import NO_ESTIMATE_NOTE_PREFIX, UNLOGGED_TROOP_TOTAL_NOTE

    battle_id = _seed_battle(conn, "Battle of Shared Note", "1750-05-01")
    side_id = _seed_side(conn, battle_id, "Nobody")
    classify_stage.run(_load_spec(tmp_path / "raw"), context=StageContext(db_conn=conn))

    note = conn.execute(
        text(
            "SELECT notes FROM missing_data_log WHERE side_id = :s AND field_name = 'troop_total'"
        ),
        {"s": side_id},
    ).scalar_one()
    assert note.startswith(UNLOGGED_TROOP_TOTAL_NOTE)
    assert UNLOGGED_TROOP_TOTAL_NOTE.startswith(NO_ESTIMATE_NOTE_PREFIX)


def test_the_besieged_side_of_a_named_siege_is_marked_defender(
    conn: Connection, tmp_path: Path
) -> None:
    battle_id = _seed_battle(conn, "Siege of Massilia", "0049-06-01 BC")
    besiegers = _seed_side(conn, battle_id, "Supporters of Julius Caesar")
    city = _seed_side(conn, battle_id, "City of Massilia")
    other_id = _seed_battle(conn, "Siege of Acre (1799)", "1799-03-20")
    acre_a = _seed_side(conn, other_id, "Ottoman Empire")
    acre_b = _seed_side(conn, other_id, "French Republic")
    spec = _load_spec(tmp_path / "raw")

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    def flag(side_id: int) -> Any:
        return _scalar(conn, "SELECT is_defender FROM battle_sides WHERE side_id = :s", s=side_id)

    assert flag(city) is True
    assert flag(besiegers) is False
    assert flag(acre_a) is None
    assert flag(acre_b) is None


# ─── Infobox order, defender answers, LLM missingness, terrain, campaigns ───


def _write_article(raw_root: Path, url: str, body: str) -> None:
    """Write a wikitext article where ``find_article_path`` will look for it."""
    directory = raw_root / "battles_html"
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(article_filename(url)).stem
    (directory / f"{stem}.wikitext").write_text(body, encoding="utf-8")


def _flag(conn: Connection, side_id: int) -> Any:
    return _scalar(conn, "SELECT is_defender FROM battle_sides WHERE side_id = :s", s=side_id)


def test_the_role_request_numbers_commanders_by_infobox_order_and_counts_unresolved(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    # Insert order is Octavian then Agrippa; the infobox lists Agrippa first.
    conn.execute(
        text("UPDATE battle_commanders SET infobox_order = :o WHERE bc_id = :b"),
        [
            {"o": 1, "b": seed["octavian_bc"]},
            {"o": 0, "b": seed["agrippa_bc"]},
        ],
    )

    pairs = classify_stage.build_pending_role_requests(conn, seed["raw_root"], spec)
    side, request = next((s, r) for s, r in pairs if s.side_label == ACTIUM_SIDE)

    assert [c.name for c in side.commanders] == ["Agrippa", "Octavian"]
    assert side.unresolved_count == 1
    assert request.user.index("#1 Agrippa") < request.user.index("#2 Octavian")
    assert "1 further commander named in the infobox" in request.user


def test_null_infobox_order_falls_back_to_bc_id_order_without_numbers(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])

    pairs = classify_stage.build_pending_role_requests(conn, seed["raw_root"], spec)
    side, request = next((s, r) for s, r in pairs if s.side_label == ACTIUM_SIDE)

    assert [c.name for c in side.commanders] == ["Octavian", "Agrippa"]
    assert "#1" not in request.user


def test_an_llm_defender_answer_fills_a_null_flag_only(conn: Connection, tmp_path: Path) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"], extra={"is_defender": True})

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _flag(conn, seed["octavian_side"]) is True
    assert _flag(conn, seed["antony_side"]) is None


def test_a_null_llm_defender_answer_writes_nothing(conn: Connection, tmp_path: Path) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"], extra={"is_defender": None})

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _flag(conn, seed["octavian_side"]) is None


def test_an_existing_defender_flag_is_not_overwritten_by_the_llm(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    conn.execute(
        text("UPDATE battle_sides SET is_defender = FALSE WHERE side_id = :s"),
        {"s": seed["octavian_side"]},
    )
    _seed_actium_llm_call(conn, spec, seed["raw_root"], extra={"is_defender": True})

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _flag(conn, seed["octavian_side"]) is False


def test_an_answer_that_misses_a_commander_is_rejected_whole_including_its_flag(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(
        conn,
        spec,
        seed["raw_root"],
        extra={
            "is_defender": True,
            "classifications": [
                {
                    "name": "Agrippa",
                    "command_role": "field_commander",
                    "hierarchy_rank": 0,
                    "reports_to": None,
                    "attribution_weight": 1.0,
                    "confidence": 0.9,
                    "reasoning": "only one of the two commanders was answered",
                }
            ],
        },
    )

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _flag(conn, seed["octavian_side"]) is None
    llm_rows = _scalar(
        conn,
        "SELECT COUNT(*) FROM battle_commanders WHERE side_id = :s AND attribution_method = 'llm'",
        s=seed["octavian_side"],
    )
    assert llm_rows == 0


def test_the_siege_title_rule_keeps_priority_over_the_llm_flag(
    conn: Connection, tmp_path: Path
) -> None:
    battle_id = _seed_battle(conn, "Siege of Massilia", "0049-06-01 BC")
    besiegers = _seed_side(conn, battle_id, "Supporters of Julius Caesar")
    city = _seed_side(conn, battle_id, "City of Massilia")
    names = ("Gaius Trebonius", "Decimus Brutus")
    for name in names:
        _seed_commander(conn, battle_id, besiegers, _seed_general(conn, name))
    spec = _load_spec(tmp_path / "raw")
    config = llm_params(spec)
    pairs = classify_stage.build_pending_role_requests(conn, tmp_path / "raw", spec)
    _side, request = next((s, r) for s, r in pairs if s.side_id == besiegers)
    # The model contradicts the title; the title rule is written first and wins.
    answer = {
        "classifications": [
            {
                "name": name,
                "command_role": "field_commander",
                "hierarchy_rank": 0,
                "reports_to": None,
                "attribution_weight": 0.5,
                "confidence": 0.8,
                "reasoning": "co-commanders of the siege",
            }
            for name in names
        ],
        "is_defender": True,
        "needs_review": False,
        "review_reason": None,
    }
    _insert_llm_call(
        conn, config, request_hash(request, config["provider"], config["model"]), answer
    )

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _flag(conn, city) is True
    assert _flag(conn, besiegers) is False


def test_two_defenders_in_one_battle_write_neither(conn: Connection, tmp_path: Path) -> None:
    from pipeline.classifiers import SideGroup

    sides = [
        (
            SideGroup(
                battle_id=1, side_id=10, side_label="A", battle_name="B", year_astronomical=1
            ),
            None,
        ),
        (
            SideGroup(
                battle_id=1, side_id=11, side_label="B", battle_name="B", year_astronomical=1
            ),
            None,
        ),
        (
            SideGroup(
                battle_id=2, side_id=20, side_label="C", battle_name="C", year_astronomical=1
            ),
            None,
        ),
    ]

    usable = classify_stage._usable_llm_defender_flags(
        {10: True, 11: True, 20: True},
        sides,  # type: ignore[arg-type]
    )

    assert usable == {20: True}


def test_a_cached_missingness_answer_is_applied_on_the_rerun(
    conn: Connection, tmp_path: Path
) -> None:
    seed = _seed_corpus(conn, tmp_path)
    spec = _load_spec(seed["raw_root"])
    _seed_actium_llm_call(conn, spec, seed["raw_root"])

    first = classify_stage.run(spec, context=StageContext(db_conn=conn))
    assert first is not None and first.batch_file is not None
    pending = classify_stage.build_pending_missingness_requests(conn, spec)
    assert len(pending) == 1
    context, request = pending[0]
    config = llm_params(spec)
    _insert_llm_call(
        conn,
        config,
        request_hash(request, config["provider"], config["model"]),
        {"missingness_type": "mar", "reason": "documentation thins out for small fleets"},
    )

    second = classify_stage.run(spec, context=StageContext(db_conn=conn))

    row = _row(
        conn,
        "SELECT missingness_class, notes FROM missing_data_log WHERE log_id = :id",
        id=context.log_id,
    )
    assert row.missingness_class == "mar"
    assert "LLM classification" in row.notes
    # Only the uncached no-article side is still waiting.
    assert second is not None and second.deferred_requests == 1


def test_terrain_is_written_only_where_empty_and_marks_its_log_row(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    url_a = "https://en.wikipedia.org/wiki/Battle_of_Terrain_A"
    url_b = "https://en.wikipedia.org/wiki/Battle_of_Terrain_B"
    _write_article(raw_root, url_a, "== Battle ==\nThe armies met on a plain beside a river.")
    _write_article(raw_root, url_b, "== Battle ==\nThe armies met on a plain beside a river.")
    empty = _seed_battle(conn, "Battle of Terrain A", "1700-01-01", wikipedia_url=url_a)
    filled = _seed_battle(conn, "Battle of Terrain B", "1700-01-02", wikipedia_url=url_b)
    conn.execute(
        text("UPDATE battles SET terrain = ARRAY['desert'] WHERE battle_id = :b"), {"b": filled}
    )
    log_id = _seed_missing(conn, empty, "terrain")
    spec = _load_spec(raw_root)

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    def terrain(battle_id: int) -> list[str]:
        return list(
            _scalar(conn, "SELECT terrain FROM battles WHERE battle_id = :b", b=battle_id) or []
        )

    assert terrain(empty) == ["river", "plain"]
    assert terrain(filled) == ["desert"]
    log = _row(conn, "SELECT missingness_class FROM missing_data_log WHERE log_id = :i", i=log_id)
    assert log.missingness_class == "observed"

    classify_stage.run(spec, context=StageContext(db_conn=conn))
    assert terrain(empty) == ["river", "plain"]


def _set_rule_type(conn: Connection, battle_id: int, battle_type: str) -> None:
    """Make a battle look like an earlier classify run typed it by rule."""
    conn.execute(
        text(
            "UPDATE battles SET battle_type = CAST(:t AS battle_type), "
            "battle_type_source = 'classify' WHERE battle_id = :b"
        ),
        {"t": battle_type, "b": battle_id},
    )
    log_id = _seed_missing(conn, battle_id, "battle_type", missingness_class="observed")
    conn.execute(
        text(
            "UPDATE missing_data_log SET notes = 'classify: inferred field (t)' WHERE log_id = :i"
        ),
        {"i": log_id},
    )


def test_campaign_replaces_a_rule_derived_type_but_never_an_extract_type(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    war = _seed_battle(conn, "Six-Day War", "1967-06-05")
    long_op = _seed_battle(conn, "Expedition Longrun", "1944-01-01")
    extract_typed = _seed_battle(conn, "Operation Extracted", "1944-01-01")
    for battle_id in (long_op, extract_typed):
        conn.execute(
            text("UPDATE battles SET date_end = date_start + 200 WHERE battle_id = :b"),
            {"b": battle_id},
        )
    _set_rule_type(conn, war, "field")
    _set_rule_type(conn, long_op, "field")
    conn.execute(
        text(
            "UPDATE battles SET battle_type = 'field', battle_type_source = 'extract' "
            "WHERE battle_id = :b"
        ),
        {"b": extract_typed},
    )
    spec = _load_spec(raw_root)

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    def kind(battle_id: int) -> str:
        return _scalar(conn, "SELECT battle_type FROM battles WHERE battle_id = :b", b=battle_id)

    assert kind(war) == "campaign"
    assert kind(long_op) == "campaign"
    assert kind(extract_typed) == "field"

    classify_stage.run(spec, context=StageContext(db_conn=conn))
    assert kind(war) == "campaign"


def test_r3_campaign_typing_rewrites_classify_typed_corpus_rows(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    # (name, start, end, precision, typed by, expected type)
    cases = [
        ("Battle of Moscow", "1941-01-01", "1942-04-20", "decade", "classify", "campaign"),
        ("Operation Barbarossa", "1941-06-22", "1941-12-05", "day", "classify", "campaign"),
        ("Siege of Leningrad", "1941-09-08", "1944-01-27", "day", "classify", "campaign"),
        ("Suez Crisis", "1957-03-01", "1956-11-07", "month", "classify", "campaign"),
        ("Siege of Tobruk", "1941-04-10", "1941-11-27", "day", "classify", "siege_offensive"),
        ("Battle of Verdun", "1916-02-21", "1916-12-18", "day", "classify", "field"),
        ("Battle of Moscow", "1941-01-01", "1942-04-20", "decade", "extract", "field"),
        ("Operation Torch", "1942-11-16", "1942-11-16", "day", "extract", "field"),
    ]
    ids = []
    for name, start, end, precision, source, _ in cases:
        battle_id = _seed_battle(conn, name, start)
        kind = "siege_offensive" if name.startswith("Siege") else "field"
        conn.execute(
            text(
                "UPDATE battles SET date_end = CAST(:e AS DATE), date_precision = :p, "
                "battle_type = CAST(:t AS battle_type), battle_type_source = :s "
                "WHERE battle_id = :b"
            ),
            {"e": end, "p": precision, "t": kind, "s": source, "b": battle_id},
        )
        if source == "classify":
            _set_rule_type(conn, battle_id, kind)
        ids.append(battle_id)

    classify_stage.run(_load_spec(raw_root), context=StageContext(db_conn=conn))

    for battle_id, case in zip(ids, cases, strict=True):
        assert (
            _scalar(conn, "SELECT battle_type FROM battles WHERE battle_id = :b", b=battle_id)
            == case[5]
        ), case[:2]


def test_an_unknown_war_titled_battle_becomes_a_campaign_without_an_article(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    war = _seed_battle(conn, "Yom Kippur War", "1973-10-06")
    spec = _load_spec(raw_root)

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    kind = _scalar(conn, "SELECT battle_type FROM battles WHERE battle_id = :b", b=war)
    assert kind == "campaign"


def test_the_unknown_share_gate_is_a_proportion_of_battles_with_articles(
    conn: Connection, tmp_path: Path
) -> None:
    spec = _load_spec(tmp_path / "raw")
    gate = next(c for c in spec["quality_checks"] if c["name"] == "battle_type_unknown_share")
    for i in range(10):
        battle_id = _seed_battle(conn, f"Battle of Gate {i}", "1700-01-01")
        conn.execute(
            text("UPDATE battles SET source_slug = :s WHERE battle_id = :b"),
            {"s": f"gate_{i}", "b": battle_id},
        )
    # Battles without an article never count, however many there are.
    for i in range(30):
        _seed_battle(conn, f"Battle of Noslug {i}", "1700-01-01")

    def result() -> Any:
        return next(r for r in QualityRunner(conn).run_all([gate]))

    assert result().passed is False
    assert result().actual_value == pytest.approx(1.0)
    conn.execute(text("UPDATE battles SET battle_type = 'field' WHERE source_slug IS NOT NULL"))
    assert result().passed is True


# ─── Provenance of battle_type survives extract re-runs ─────────────────────

_FLEET_URL = "https://en.wikipedia.org/wiki/Battle_of_Longfleet"


def _extract_record(
    *,
    battle_type: str | None = None,
    date_end: str | None = None,
    name: str = "Battle of Longfleet",
) -> BattleExtraction:
    """A merged battle as extract writes it, logging every covariate it lacks."""
    facts = BattleFacts(
        name=name,
        wikipedia_url=_FLEET_URL,
        date_start="1800-01-01",
        date_end=date_end,
        date_precision="day",
        battle_type=battle_type,
    )
    absent = [
        name
        for name, value in (("battle_type", battle_type), ("terrain", None), ("fortified", None))
        if value is None
    ]
    return BattleExtraction(
        slug="longfleet",
        name=name,
        facts=facts,
        sides=[SideExtraction(label="A"), SideExtraction(label="B")],
        missing=[MissingField(field_name=name, notes="no source reported it") for name in absent],
        n_sources=1,
    )


def _battle_state(conn: Connection, battle_id: int) -> Any:
    return _row(
        conn,
        "SELECT CAST(battle_type AS TEXT) AS battle_type, battle_type_source "
        "FROM battles WHERE battle_id = :b",
        b=battle_id,
    )


def test_a_rule_typed_battle_is_still_rewritten_to_campaign_after_extract_reruns(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    _write_article(raw_root, _FLEET_URL, _NAVAL_ARTICLE)
    battle_id = extract_write_battle(conn, _extract_record())
    spec = _load_spec(raw_root)

    classify_stage.run(spec, context=StageContext(db_conn=conn))
    state = _battle_state(conn, battle_id)
    assert (state.battle_type, state.battle_type_source) == ("naval", "classify")

    # Extract owns missing_data_log and rewrites it on every run, which used to
    # erase the only record that classify had chosen this type.
    extract_write_battle(conn, _extract_record())
    state = _battle_state(conn, battle_id)
    assert (state.battle_type, state.battle_type_source) == ("naval", "classify")

    # A "Battle of" title is exempt from the span rule, so rename it to a neutral title.
    conn.execute(
        text(
            "UPDATE battles SET name = 'Expedition Longfleet', date_end = date_start + 200 "
            "WHERE battle_id = :b"
        ),
        {"b": battle_id},
    )
    classify_stage.run(spec, context=StageContext(db_conn=conn))

    state = _battle_state(conn, battle_id)
    assert state.battle_type == "campaign"
    assert state.battle_type_source == "classify"
    notes = _scalar(
        conn,
        "SELECT notes FROM missing_data_log WHERE battle_id = :b AND field_name = 'battle_type'",
        b=battle_id,
    )
    assert "reclassified" in notes


def test_a_type_extract_supplied_is_recorded_as_extract_and_never_made_a_campaign(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    # An "Operation" with a 200-day day-precision span meets the campaign rule,
    # so only the battle_type_source guard keeps this extract-typed row a field battle.
    record = _extract_record(battle_type="field", date_end="1800-07-20", name="Operation Longfleet")
    battle_id = extract_write_battle(conn, record)
    assert tuple(_battle_state(conn, battle_id)) == ("field", "extract")

    # A re-run in which no source names a type leaves the provenance alone.
    extract_write_battle(conn, replace(record, facts=replace(record.facts, battle_type=None)))
    assert tuple(_battle_state(conn, battle_id)) == ("field", "extract")

    classify_stage.run(_load_spec(raw_root), context=StageContext(db_conn=conn))

    assert tuple(_battle_state(conn, battle_id)) == ("field", "extract")


def _populated_covariate_battle(
    conn: Connection, tmp_path: Path
) -> tuple[dict[str, Any], Path, dict[str, int]]:
    """A classified battle whose covariate columns are set but whose rows extract re-logged."""
    raw_root = tmp_path / "raw"
    _write_article(raw_root, _FLEET_URL, _NAVAL_ARTICLE)
    battle_id = extract_write_battle(conn, _extract_record())
    spec = _load_spec(raw_root)
    classify_stage.run(spec, context=StageContext(db_conn=conn))
    conn.execute(
        text(
            "UPDATE battles SET terrain = ARRAY['coastal'], fortified = FALSE WHERE battle_id = :b"
        ),
        {"b": battle_id},
    )
    extract_write_battle(conn, _extract_record())

    log_ids = {
        r.field_name: r.log_id
        for r in conn.execute(
            text(
                "SELECT log_id, field_name FROM missing_data_log WHERE battle_id = :b "
                "AND field_name IN ('battle_type', 'terrain', 'fortified') "
                "AND missingness_class = 'unclassified'"
            ),
            {"b": battle_id},
        )
    }
    assert set(log_ids) == {"battle_type", "terrain", "fortified"}
    return spec, raw_root, log_ids


def _spy_on_missingness_requests(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    calls: list[list[int]] = []
    original = classify_stage._build_missingness_pairs

    def spy(conn_: Any, ids: list[int], spec_: dict[str, Any]) -> Any:
        calls.append(list(ids))
        return original(conn_, ids, spec_)

    monkeypatch.setattr(classify_stage, "_build_missingness_pairs", spy)
    return calls


def test_covariate_rows_extract_reinserts_for_populated_columns_end_observed(
    conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _raw_root, log_ids = _populated_covariate_battle(conn, tmp_path)
    # An open row nothing excludes, so the spy sees a genuinely non-empty input.
    other = _seed_battle(conn, "Battle of Elsewhere", "1801-01-01")
    open_id = _seed_missing(conn, other, "troop_total")
    calls = _spy_on_missingness_requests(monkeypatch)

    classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert calls, "the missingness request builder was never reached"
    requested = [log_id for ids in calls for log_id in ids]
    assert open_id in requested
    assert not set(requested) & set(log_ids.values())
    for field_name, log_id in log_ids.items():
        row = _row(
            conn,
            "SELECT missingness_class, notes FROM missing_data_log WHERE log_id = :i",
            i=log_id,
        )
        assert row.missingness_class == "observed", field_name
        assert f"classify: observed {field_name} already set" in row.notes
    assert "source classify" in _scalar(
        conn, "SELECT notes FROM missing_data_log WHERE log_id = :i", i=log_ids["battle_type"]
    )


def test_the_offline_export_omits_populated_covariate_rows(
    conn: Connection, tmp_path: Path
) -> None:
    spec, raw_root, log_ids = _populated_covariate_battle(conn, tmp_path)
    other = _seed_battle(conn, "Battle of Elsewhere", "1801-01-01")
    open_id = _seed_missing(conn, other, "troop_total")

    exported = {
        context.log_id
        for context, _request in classify_stage.build_pending_missingness_requests(
            conn, spec, raw_root
        )
    }

    assert open_id in exported
    assert not exported & set(log_ids.values())


def test_a_fortified_row_settled_by_the_siege_rule_this_run_ends_observed(
    conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    battle_id = extract_write_battle(conn, _extract_record(name="Siege of Longfleet"))
    assert _scalar(conn, "SELECT fortified FROM battles WHERE battle_id = :b", b=battle_id) is None
    fortified_log = _scalar(
        conn,
        "SELECT log_id FROM missing_data_log WHERE battle_id = :b AND field_name = 'fortified'",
        b=battle_id,
    )
    calls = _spy_on_missingness_requests(monkeypatch)

    classify_stage.run(_load_spec(raw_root), context=StageContext(db_conn=conn))

    assert _scalar(conn, "SELECT fortified FROM battles WHERE battle_id = :b", b=battle_id) is True
    assert fortified_log not in [log_id for ids in calls for log_id in ids]
    row = _row(
        conn, "SELECT missingness_class FROM missing_data_log WHERE log_id = :i", i=fortified_log
    )
    assert row.missingness_class == "observed"


def _set_campaign(conn: Connection, battle_id: int, source: str, precision: str) -> None:
    conn.execute(
        text(
            "UPDATE battles SET battle_type = 'campaign', battle_type_source = :s, "
            "date_precision = :p WHERE battle_id = :b"
        ),
        {"s": source, "p": precision, "b": battle_id},
    )


def _kind(conn: Connection, battle_id: int) -> str:
    return _scalar(conn, "SELECT battle_type FROM battles WHERE battle_id = :b", b=battle_id)


def test_a_classify_typed_campaign_reverts_when_its_date_precision_is_refined(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    siege = _seed_battle(conn, "Siege of Tobruk", "1941-04-10")
    conn.execute(
        text("UPDATE battles SET date_end = '1941-11-27' WHERE battle_id = :b"), {"b": siege}
    )
    _set_campaign(conn, siege, "classify", "decade")
    _seed_missing(conn, siege, "battle_type", missingness_class="observed")
    spec = _load_spec(raw_root)

    classify_stage.run(spec, context=StageContext(db_conn=conn))
    assert _kind(conn, siege) == "campaign"

    conn.execute(
        text("UPDATE battles SET date_precision = 'day' WHERE battle_id = :b"), {"b": siege}
    )
    with structlog.testing.capture_logs() as logs:
        classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _kind(conn, siege) == "siege_offensive"
    reverted = [e for e in logs if e["event"] == "classify_campaign_reverted"]
    assert len(reverted) == 1
    assert reverted[0]["battle_id"] == siege
    assert reverted[0]["old_type"] == "campaign"
    assert reverted[0]["new_type"] == "siege_offensive"
    notes = _scalar(
        conn,
        "SELECT notes FROM missing_data_log WHERE battle_id = :b AND field_name = 'battle_type'",
        b=siege,
    )
    assert "reverted 'campaign' to 'siege_offensive'" in notes

    with structlog.testing.capture_logs() as logs:
        classify_stage.run(spec, context=StageContext(db_conn=conn))
    assert _kind(conn, siege) == "siege_offensive"
    assert not [e for e in logs if e["event"].startswith("classify_campaign")]


def test_a_campaign_reverting_to_unknown_ends_in_a_never_typed_battles_row_state(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    reverted = _seed_battle(conn, "Battle of Nowhere", "1941-04-10")
    untouched = _seed_battle(conn, "Battle of Elsewhere", "1941-04-10")
    _set_campaign(conn, reverted, "classify", "day")
    reverted_log = _seed_missing(conn, reverted, "battle_type", missingness_class="observed")
    untouched_log = _seed_missing(conn, untouched, "battle_type")
    spec = _load_spec(raw_root)

    # The missingness pass reads before the type pass writes, so the reopened
    # row is classified on the following run, as any unclassified row would be.
    classify_stage.run(spec, context=StageContext(db_conn=conn))
    classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _kind(conn, reverted) == "unknown"
    assert _kind(conn, untouched) == "unknown"
    sql = "SELECT missingness_class FROM missing_data_log WHERE log_id = :i"
    assert _scalar(conn, sql, i=reverted_log) == _scalar(conn, sql, i=untouched_log)
    assert _scalar(conn, sql, i=reverted_log) != "observed"


def test_a_classify_typed_campaign_whose_rule_still_fires_is_untouched_and_silent(
    conn: Connection, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    war = _seed_battle(conn, "Yom Kippur War", "1973-10-06")
    _set_campaign(conn, war, "classify", "day")
    log_id = _seed_missing(conn, war, "battle_type", missingness_class="observed")
    conn.execute(
        text("UPDATE missing_data_log SET notes = 'classify: inferred' WHERE log_id = :i"),
        {"i": log_id},
    )
    spec = _load_spec(raw_root)

    with structlog.testing.capture_logs() as logs:
        classify_stage.run(spec, context=StageContext(db_conn=conn))
        classify_stage.run(spec, context=StageContext(db_conn=conn))

    assert _kind(conn, war) == "campaign"
    assert not [e for e in logs if e["event"].startswith("classify_campaign")]
    notes = _scalar(conn, "SELECT notes FROM missing_data_log WHERE log_id = :i", i=log_id)
    assert notes == "classify: inferred"


def test_an_extract_typed_campaign_is_never_retyped(conn: Connection, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    siege = _seed_battle(conn, "Siege of Tobruk", "1941-04-10")
    unsourced = _seed_battle(conn, "Siege of Malta", "1941-04-10")
    _set_campaign(conn, siege, "extract", "day")
    _set_campaign(conn, unsourced, "extract", "day")
    conn.execute(
        text("UPDATE battles SET battle_type_source = NULL WHERE battle_id = :b"),
        {"b": unsourced},
    )

    classify_stage.run(_load_spec(raw_root), context=StageContext(db_conn=conn))

    assert _kind(conn, siege) == "campaign"
    assert _kind(conn, unsourced) == "campaign"
