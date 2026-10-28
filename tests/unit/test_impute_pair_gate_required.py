"""The kept-pair gate fails, not falls back, when reconcile recorded no slope."""

from __future__ import annotations

from pipeline.imputers.gates import GateResult, require_pair_reference


def _gates() -> list[GateResult]:
    return [
        GateResult("kept_pair_ratio_slope", True, "error", "skipped: 0 pairs", ">= 0.8"),
        GateResult("other", True, "error", "ok", "x"),
    ]


def test_a_missing_reference_slope_fails_the_pair_gate_with_a_clear_message() -> None:
    out = require_pair_reference(_gates(), None, reconcile_run_id=7)

    pair = next(g for g in out if g.name == "kept_pair_ratio_slope")
    assert not pair.passed
    assert pair.severity == "error"
    assert "reconcile run 7 has no within_battle_slope; re-run reconcile" in pair.message
    assert next(g for g in out if g.name == "other").passed


def test_a_present_reference_slope_leaves_the_gates_alone() -> None:
    assert require_pair_reference(_gates(), 0.6, reconcile_run_id=7) == _gates()
