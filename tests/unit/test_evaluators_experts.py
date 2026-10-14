"""Unit tests for pipeline.evaluators.experts."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import yaml

from pipeline.evaluators.experts import (
    ExpertConfig,
    PairwiseCheck,
    RankCheck,
    ReferenceList,
    UncreditedCheck,
    compare_reference,
    load_expert_config,
    named_checks_gate,
    run_named_checks,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "expert_rankings.yaml"
QIDS = {"a": "Q1", "b": "Q2", "c": "Q3", "d": "Q4", "e": "Q5"}


def _config(
    reference_rankings: tuple[ReferenceList, ...] = (),
    expert_lists: tuple[ReferenceList, ...] = (),
) -> ExpertConfig:
    return ExpertConfig(
        generals=QIDS,
        rank_checks=(RankCheck("a", (1, 3)),),
        pairwise_checks=(PairwiseCheck("b", "c"),),
        reference_rankings=reference_rankings,
        expert_lists=expert_lists,
    )


def test_real_config_is_well_formed() -> None:
    cfg = load_expert_config(CONFIG_PATH)
    assert all(re.match(r"^Q[1-9][0-9]*$", q) for q in cfg.generals.values())
    assert len(set(cfg.generals.values())) == len(cfg.generals)
    arsht = next(r for r in cfg.reference_rankings if r.name == "arsht_2019_subset")
    assert len(arsht.members) == 22
    assert "not Arsht's published ranks" in arsht.caveat
    assert cfg.generals["napoleon"] == "Q517"
    assert cfg.rank_checks and cfg.pairwise_checks and cfg.expert_lists


def test_loader_rejects_bad_input(tmp_path: Path) -> None:
    def write(text: str) -> Path:
        p = tmp_path / "c.yaml"
        p.write_text(text, encoding="utf-8")
        return p

    with pytest.raises(ValueError, match="malformed qid"):
        load_expert_config(write("generals: {a: {qid: Q0x}}"))
    with pytest.raises(ValueError, match="duplicate qid"):
        load_expert_config(write("generals: {a: {qid: Q1}, b: {qid: Q1}}"))
    with pytest.raises(ValueError, match="unknown general key"):
        load_expert_config(
            write("generals: {a: {qid: Q1}}\nrank_checks: [{general: z, rank_range: [1, 2]}]")
        )


def test_named_checks_pass_fail_not_evaluable() -> None:
    cfg = _config()
    res = run_named_checks(cfg, {"Q1": 2, "Q2": 1, "Q3": 5}, 10)
    assert [r["status"] for r in res] == ["pass", "pass"]
    res = run_named_checks(cfg, {"Q1": 9, "Q2": 6, "Q3": 5}, 10)
    assert [r["status"] for r in res] == ["fail", "fail"]
    res = run_named_checks(cfg, {"Q1": 1, "Q2": 2}, 10)
    assert res[0]["status"] == "pass"
    assert res[1]["status"] == "not_evaluable"
    assert "c" in res[1]["detail"]


def test_gate_requires_every_check_to_pass() -> None:
    cfg = _config()
    ok = named_checks_gate(run_named_checks(cfg, {"Q1": 1, "Q2": 1, "Q3": 2}, 5))
    assert ok["passed"] is True
    missing = named_checks_gate(run_named_checks(cfg, {"Q1": 1}, 5))
    assert missing["passed"] is False
    assert missing["actual"]["not_evaluable"] == 1
    assert named_checks_gate([])["passed"] is False


def test_ordered_reference_spearman() -> None:
    ref = ReferenceList("r", "ordered", ("a", "b", "c", "d"))
    cfg = _config(reference_rankings=(ref,))
    same = compare_reference(cfg, {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}, 10)[0]
    assert same["spearman"] == pytest.approx(1.0)
    rev = compare_reference(cfg, {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}, 10)[0]
    assert rev["spearman"] == pytest.approx(-1.0)
    few = compare_reference(cfg, {"Q1": 1, "Q2": 2}, 10)[0]
    assert math.isnan(few["spearman"])
    assert few["n_matched"] == 2
    assert few["unmatched"] == ["c", "d"]


def test_unordered_reference_metrics() -> None:
    ref = ReferenceList("u", "unordered", ("a", "b", "c", "d"))
    cfg = _config(expert_lists=(ref,))
    row = compare_reference(cfg, {"Q1": 1, "Q2": 10, "Q3": 100}, 100, top_n=50)[0]
    assert row["n_matched"] == 3
    assert row["unmatched"] == ["d"]
    assert row["share_in_top_n"] == pytest.approx(2 / 4)
    assert row["mean_percentile_rank"] == pytest.approx((111 / 3) / 100)
    assert row["our_ranks"] == {"a": 1, "b": 10, "c": 100}


def test_real_config_check_set() -> None:
    cfg = load_expert_config(CONFIG_PATH)
    assert "genghis_khan" not in cfg.generals
    assert [u.general for u in cfg.uncredited_checks] == ["justinian"]
    assert cfg.rank_checks
    assert all(not rc.gated for rc in cfg.rank_checks)
    assert all(pc.higher != "belisarius" for pc in cfg.pairwise_checks)
    pair = next(pc for pc in cfg.pairwise_checks if pc.higher == "agrippa")
    assert pair.lower == "augustus"
    assert "tests attribution" in pair.note


def test_napoleon_quote_keeps_accents() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    entry = next(e for e in raw["expert_lists"] if e["name"] == "napoleon_great_captains")
    assert entry["source_quote"] == (
        "Faites la guerre offensive comme Alexandre, Annibal, C\u00e9sar, Gustave Adolphe, "
        "Turenne, le Prince Eug\u00e8ne et Fr\u00e9d\u00e9ric; lisez, relisez l'histoire de leurs "
        "quatre vingt trois campagnes; modelez vous sur eux."
    )


def test_rank_checks_report_but_do_not_gate() -> None:
    cfg = ExpertConfig(
        generals=QIDS,
        rank_checks=(RankCheck("a", (1, 3), gated=False),),
        pairwise_checks=(PairwiseCheck("b", "c"),),
        reference_rankings=(),
        expert_lists=(),
    )
    res = run_named_checks(cfg, {"Q1": 9, "Q2": 1, "Q3": 2}, 10)
    assert res[0]["status"] == "fail"
    assert res[0]["gated"] is False
    assert res[1]["gated"] is True
    gate = named_checks_gate(res)
    assert gate["passed"] is True
    assert gate["actual"]["total"] == 1
    unranked = named_checks_gate(run_named_checks(cfg, {"Q2": 1, "Q3": 2}, 10))
    assert unranked["passed"] is True


def test_uncredited_check_passes_when_absent_and_fails_when_ranked() -> None:
    cfg = ExpertConfig(
        generals=QIDS,
        rank_checks=(),
        pairwise_checks=(),
        reference_rankings=(),
        expert_lists=(),
        uncredited_checks=(UncreditedCheck("d"),),
    )
    absent = run_named_checks(cfg, {"Q1": 1}, 10)
    assert absent[0]["status"] == "pass"
    assert absent[0]["kind"] == "uncredited"
    assert absent[0]["gated"] is True
    assert named_checks_gate(absent)["passed"] is True
    ranked = run_named_checks(cfg, {"Q1": 1, "Q4": 7}, 10)
    assert ranked[0]["status"] == "fail"
    assert named_checks_gate(ranked)["passed"] is False


def test_ordered_reference_has_no_expected_rank_fields() -> None:
    ref = ReferenceList("r", "ordered", ("a", "b", "c"), caveat="subset")
    row = compare_reference(_config(reference_rankings=(ref,)), {"Q1": 1, "Q2": 2, "Q3": 3}, 5)[0]
    assert not [k for k in row if "expected" in k]
    assert row["caveat"] == "subset"
