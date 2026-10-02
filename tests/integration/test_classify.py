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
  right rows alone -- handover.md §12.3's rule, now also owed to classify's
  own writes.
- **The quality gates execute against the real schema.** §4.1 and §11.4 of
  handover.md are both a gate whose SQL names something that turned out not
  to exist, which can only ever be caught by running it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Connection, Engine, text

from pipeline.crawlers.wikipedia import article_filename
from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
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
# further from the epoch than the astronomical one (handover.md §4.5).
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


def _seed_commander(
    conn: Connection, battle_id: int, side_id: int, general_id: int
) -> int:
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


def _seed_actium_llm_call(conn: Connection, spec: dict[str, Any], raw_root: Path) -> None:
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
    }

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
            text(
                "SELECT battle_id, battle_type, fortified FROM battles ORDER BY battle_id"
            )
        ).fetchall()
        missing = conn.execute(
            text(
                "SELECT log_id, missingness_class, notes FROM missing_data_log ORDER BY log_id"
            )
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
        "missing_data_all_logged",
    }
    for name, result in results.items():
        assert "does not exist" not in (result.message or ""), f"{name}: {result.message}"
        print(f"{name}: passed={result.passed} value={result.actual_value}")
