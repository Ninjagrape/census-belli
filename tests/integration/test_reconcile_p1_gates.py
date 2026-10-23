"""The P1 recovery gates, judged by the real SQL on fabricated diagnostics.

Each gate reads one number from the latest ``model_runs.diagnostics`` payload and
compares it with a threshold from ``agents/reconcile.yaml``. These tests put the
number on either side of the threshold and check the verdict and the severity
the redesign calls for (``docs/handover-log/32-troop-redesign-review-2026-10-23.md``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.db import apply_schema, get_engine
from pipeline.quality import QualityRunner
from pipeline.reconcilers import store
from tests.integration.test_reconcile import (
    _AVAILABLE,
    _REASON,
    _fabricate_run,
    _quality_check,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _AVAILABLE,
        reason=f"No database available ({_REASON}). Run `make db-up` first.",
    ),
]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(f"Refusing to reset database {db_name!r}")
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


def _payload(published: dict[str, Any] | None = None, neutral: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {
            "troops": {"recovery": published or {}},
            "troops_neutral": {"recovery": neutral or {}},
        }
    )


def _passes(conn: Connection, gate: str, payload: str) -> bool:
    _fabricate_run(conn, payload)
    return QualityRunner(conn).run_one(_quality_check(gate)).passed


# gate name, recovery key, passing value at the boundary, failing value just past it
_ERROR_GATES = [
    ("singleton_slope_recovers", "gate_singleton_slope", 0.85, 0.849),
    ("estimates_not_collapsed", "gate_spread_ratio", 0.85, 0.849),
    ("within_battle_ratio_recovers", "gate_within_battle_slope", 0.5, 0.499),
    ("side_deviation_not_collapsed", "gate_s_d_median", 0.25, 0.249),
    ("between_battle_variance_is_plausible", "gate_ppc_tail_p", 0.05, 0.049),
    ("noise_not_leaking", "gate_noise_ratio", 1.5, 1.501),
]


@pytest.mark.parametrize(("gate", "key", "ok", "bad"), _ERROR_GATES)
def test_the_error_gates_flip_at_their_threshold(
    conn: Connection, gate: str, key: str, ok: float, bad: float
) -> None:
    assert _quality_check(gate)["severity"] == "error"

    assert _passes(conn, gate, _payload({key: ok}, {key: ok}))
    assert not _passes(conn, gate, _payload({key: bad}, {key: ok})), "published fit counts"
    assert not _passes(conn, gate, _payload({key: ok}, {key: bad})), "neutral fit counts"


@pytest.mark.parametrize(("gate", "key"), [g[:2] for g in _ERROR_GATES])
def test_a_skipped_recovery_gate_passes(conn: Connection, gate: str, key: str) -> None:
    assert _passes(conn, gate, _payload({key: None}, {key: None}))
    assert _passes(conn, gate, _payload())


def test_the_old_within_battle_floor_of_0_8_is_gone(conn: Connection) -> None:
    assert _passes(
        conn,
        "within_battle_ratio_recovers",
        _payload({"gate_within_battle_slope": 0.6}, {"gate_within_battle_slope": 0.6}),
    )


def test_wb_and_reliability_must_agree_within_a_tenth_as_a_warning(conn: Connection) -> None:
    gate = "wb_matches_reliability"
    assert _quality_check(gate)["severity"] == "warning"
    key = "gate_wb_reliability_gap"

    assert _passes(conn, gate, _payload({key: 0.1}, {key: 0.0}))
    assert not _passes(conn, gate, _payload({key: 0.101}, {key: 0.0}))
    assert not _passes(conn, gate, _payload({key: 0.0}, {key: 0.101}))
    assert _passes(conn, gate, _payload({key: None}, {key: None}))


@pytest.mark.parametrize(
    ("gate", "key"),
    [
        ("neutral_post1800_level_relative", "gate_relative_level_violation"),
        ("neutral_all_sides_level_in_band", "gate_all_ratio_violation"),
    ],
)
def test_the_neutral_level_bands_are_warnings_on_the_neutral_fit(
    conn: Connection, gate: str, key: str
) -> None:
    assert _quality_check(gate)["severity"] == "warning"

    assert _passes(conn, gate, _payload(neutral={key: 0}))
    assert not _passes(conn, gate, _payload(neutral={key: 1}))
    assert _passes(conn, gate, _payload(neutral={key: None}))


def test_the_post_1800_band_params_match_the_stated_bands() -> None:
    import yaml

    from tests.integration.test_reconcile import AGENTS

    params = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))["params"]

    assert params["likelihood_family"] == "normal"
    assert (params["post1800_ratio_band_lo"], params["post1800_ratio_band_hi"]) == (0.85, 1.15)
    assert (params["all_ratio_band_lo"], params["all_ratio_band_hi"]) == (0.9, 1.1)
    assert params["min_within_battle_slope"] == 0.5
    assert params["min_s_d_median"] == 0.25
    assert params["max_noise_ratio"] == 1.5
    assert params["min_ppc_tail_p"] == 0.05
    assert params["max_wb_reliability_gap"] == 0.1


def test_impute_reads_the_reconcile_runs_within_battle_slope(conn: Connection) -> None:
    diagnostics = json.dumps(
        {
            "troops": {"recovery": {"within_battle_slope": 0.62}},
            "troops_neutral": {"recovery": {"within_battle_slope": 0.58}},
        }
    )
    run_id = conn.execute(
        text(
            "INSERT INTO model_runs (run_name, model_type, config, started_at, completed_at, "
            "diagnostics) VALUES ('p1_slope', 'source_disagreement', CAST('{}' AS JSONB), now(), "
            "now(), CAST(:d AS JSONB)) RETURNING run_id"
        ),
        {"d": diagnostics},
    ).scalar_one()

    assert store.reconcile_within_battle_slope(conn, run_id, neutral=True) == pytest.approx(0.58)
    assert store.reconcile_within_battle_slope(conn, run_id, neutral=False) == pytest.approx(0.62)
    assert store.reconcile_within_battle_slope(conn, run_id + 999, neutral=True) is None
    assert store.reconcile_within_battle_slope(conn, None, neutral=True) is None
