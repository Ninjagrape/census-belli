"""Expert-ranking validation targets for the evaluate stage.

Loads ``config/expert_rankings.yaml`` and scores a model ranking against the
named sanity checks and the external reference lists. The targets are
sanity checks, not ground truth.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger()

_QID_RE = re.compile(r"^Q[1-9][0-9]*$")
_LIST_KINDS = frozenset({"ordered", "unordered"})
MIN_SPEARMAN_MEMBERS = 3


@dataclass(frozen=True)
class RankCheck:
    """A general who must land inside an inclusive rank range."""

    general: str
    rank_range: tuple[int, int]
    note: str = ""
    gated: bool = True


@dataclass(frozen=True)
class PairwiseCheck:
    """A pair where ``higher`` must rank better than ``lower``."""

    higher: str
    lower: str
    note: str = ""


@dataclass(frozen=True)
class UncreditedCheck:
    """A general who must be absent from the ranking (no credited battles)."""

    general: str
    note: str = ""


@dataclass(frozen=True)
class ReferenceList:
    """An external list of generals, ordered or unordered."""

    name: str
    kind: str
    members: tuple[str, ...]
    source: str = ""
    caveat: str = ""


@dataclass(frozen=True)
class ExpertConfig:
    """Parsed validation targets.

    Attributes:
        generals: Short key to Wikidata Q-id.
        rank_checks: Rank-range sanity checks.
        pairwise_checks: Pairwise ordering checks.
        uncredited_checks: Generals who must not appear in the ranking.
        reference_rankings: Ordered external rankings.
        expert_lists: Unordered external lists.
    """

    generals: Mapping[str, str]
    rank_checks: tuple[RankCheck, ...]
    pairwise_checks: tuple[PairwiseCheck, ...]
    reference_rankings: tuple[ReferenceList, ...]
    expert_lists: tuple[ReferenceList, ...]
    uncredited_checks: tuple[UncreditedCheck, ...] = ()


def _require_known(keys: Sequence[str], generals: Mapping[str, str], where: str) -> None:
    for key in keys:
        if key not in generals:
            raise ValueError(f"unknown general key {key!r} in {where}")


def _parse_generals(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("'generals' must be a non-empty mapping")
    generals: dict[str, str] = {}
    seen_qids: dict[str, str] = {}
    for key, entry in raw.items():
        qid = entry.get("qid") if isinstance(entry, dict) else None
        if not isinstance(qid, str) or not _QID_RE.match(qid):
            raise ValueError(f"malformed qid for general {key!r}: {qid!r}")
        if qid in seen_qids:
            raise ValueError(f"duplicate qid {qid} for {key!r} and {seen_qids[qid]!r}")
        seen_qids[qid] = key
        generals[str(key)] = qid
    return generals


def _parse_lists(
    raw: Any, generals: Mapping[str, str], section: str, expected_kind: str
) -> tuple[ReferenceList, ...]:
    lists: list[ReferenceList] = []
    names: set[str] = set()
    for entry in raw or []:
        name = str(entry["name"])
        kind = str(entry.get("kind", expected_kind))
        if kind not in _LIST_KINDS or kind != expected_kind:
            raise ValueError(f"{section}[{name}] has kind {kind!r}, expected {expected_kind!r}")
        if name in names:
            raise ValueError(f"duplicate list name {name!r} in {section}")
        names.add(name)
        members = tuple(str(m) for m in entry["members"])
        if len(set(members)) != len(members):
            raise ValueError(f"duplicate member in {section}[{name}]")
        _require_known(members, generals, f"{section}[{name}]")
        source = str(entry.get("source") or entry.get("source_url") or "")
        lists.append(
            ReferenceList(
                name=name,
                kind=kind,
                members=members,
                source=source,
                caveat=str(entry.get("caveat") or ""),
            )
        )
    return tuple(lists)


def load_expert_config(path: Path) -> ExpertConfig:
    """Load and validate the expert rankings file.

    Args:
        path: Path to ``expert_rankings.yaml``.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: On an unknown general key, a malformed or duplicate Q-id,
            a malformed rank range, or a duplicate list name.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    generals = _parse_generals(raw.get("generals"))

    rank_checks: list[RankCheck] = []
    for entry in raw.get("rank_checks") or []:
        lo, hi = (int(v) for v in entry["rank_range"])
        if not 1 <= lo <= hi:
            raise ValueError(f"bad rank_range {entry['rank_range']!r} for {entry['general']!r}")
        _require_known([entry["general"]], generals, "rank_checks")
        rank_checks.append(
            RankCheck(
                str(entry["general"]),
                (lo, hi),
                str(entry.get("note", "")),
                bool(entry.get("gated", True)),
            )
        )

    pairwise: list[PairwiseCheck] = []
    for entry in raw.get("pairwise_checks") or []:
        _require_known([entry["higher"], entry["lower"]], generals, "pairwise_checks")
        if entry["higher"] == entry["lower"]:
            raise ValueError(f"pairwise check compares {entry['higher']!r} with itself")
        pairwise.append(
            PairwiseCheck(str(entry["higher"]), str(entry["lower"]), str(entry.get("note", "")))
        )

    uncredited: list[UncreditedCheck] = []
    for entry in raw.get("uncredited_checks") or []:
        _require_known([entry["general"]], generals, "uncredited_checks")
        uncredited.append(UncreditedCheck(str(entry["general"]), str(entry.get("note", ""))))

    return ExpertConfig(
        generals=generals,
        rank_checks=tuple(rank_checks),
        pairwise_checks=tuple(pairwise),
        reference_rankings=_parse_lists(
            raw.get("reference_rankings"), generals, "reference_rankings", "ordered"
        ),
        expert_lists=_parse_lists(raw.get("expert_lists"), generals, "expert_lists", "unordered"),
        uncredited_checks=tuple(uncredited),
    )


def _result(
    name: str,
    kind: str,
    status: str,
    detail: str,
    ranks: Mapping[str, int | None],
    *,
    gated: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "status": status,
        "detail": detail,
        "ranks": dict(ranks),
        "gated": gated,
    }


def run_named_checks(
    config: ExpertConfig, rank_by_qid: Mapping[str, int], n_ranked: int
) -> list[dict[str, Any]]:
    """Evaluate rank-range, pairwise and uncredited checks against a ranking.

    Rank 1 is best. A rank-range or pairwise check whose general is absent from
    ``rank_by_qid`` is ``not_evaluable``, never a pass. An uncredited check
    passes when its general is absent and fails when ranked.

    Args:
        config: Parsed targets.
        rank_by_qid: Wikidata Q-id to rank (1 = best).
        n_ranked: Number of ranked generals, used in the detail text.

    Returns:
        One result dict per check with keys name, kind, status, detail, ranks, gated.
    """
    results: list[dict[str, Any]] = []
    for rc in config.rank_checks:
        rank = rank_by_qid.get(config.generals[rc.general])
        name = f"{rc.general}_rank_{rc.rank_range[0]}_{rc.rank_range[1]}"
        ranks = {rc.general: rank}
        if rank is None:
            results.append(
                _result(
                    name,
                    "rank_range",
                    "not_evaluable",
                    f"not ranked: {rc.general}",
                    ranks,
                    gated=rc.gated,
                )
            )
            continue
        ok = rc.rank_range[0] <= rank <= rc.rank_range[1]
        detail = f"rank {rank} of {n_ranked}, expected {rc.rank_range[0]}-{rc.rank_range[1]}"
        status = "pass" if ok else "fail"
        results.append(_result(name, "rank_range", status, detail, ranks, gated=rc.gated))

    for pc in config.pairwise_checks:
        hi = rank_by_qid.get(config.generals[pc.higher])
        lo = rank_by_qid.get(config.generals[pc.lower])
        name = f"{pc.higher}_above_{pc.lower}"
        pair_ranks = {pc.higher: hi, pc.lower: lo}
        if hi is None or lo is None:
            missing = ", ".join(k for k, v in pair_ranks.items() if v is None)
            results.append(
                _result(name, "pairwise", "not_evaluable", f"not ranked: {missing}", pair_ranks)
            )
            continue
        detail = f"{pc.higher} rank {hi}, {pc.lower} rank {lo}"
        results.append(_result(name, "pairwise", "pass" if hi < lo else "fail", detail, pair_ranks))

    for uc in config.uncredited_checks:
        rank = rank_by_qid.get(config.generals[uc.general])
        name = f"{uc.general}_uncredited"
        if rank is None:
            detail = f"{uc.general} not ranked (no credited battles)"
        else:
            detail = f"{uc.general} ranked {rank} of {n_ranked}, expected unranked"
        status = "pass" if rank is None else "fail"
        results.append(_result(name, "uncredited", status, detail, {uc.general: rank}))
    return results


def named_checks_gate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise named-check results as a quality gate.

    Args:
        results: Output of :func:`run_named_checks`.

    Returns:
        Gate dict; ``passed`` only if every gated check passed. Checks with
        ``gated`` false are reported by :func:`run_named_checks` but ignored here.
    """
    gated = [r for r in results if r.get("gated", True)]
    counts = {"pass": 0, "fail": 0, "not_evaluable": 0}
    for res in gated:
        counts[str(res["status"])] += 1
    passed = bool(gated) and counts["pass"] == len(gated)
    failing = [str(r["name"]) for r in gated if r["status"] != "pass"]
    message = (
        f"all {counts['pass']} named checks pass"
        if passed
        else f"{counts['fail']} failed, {counts['not_evaluable']} not evaluable: "
        f"{', '.join(failing) or 'no checks defined'}"
    )
    return {
        "name": "named_checks",
        "passed": passed,
        "actual": {**counts, "total": len(gated)},
        "threshold": "all gated checks pass and none not_evaluable",
        "message": message,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman correlation with average ranks; NaN if undefined."""
    if len(xs) < MIN_SPEARMAN_MEMBERS:
        return math.nan
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return math.nan
    return cov / math.sqrt(vx * vy)


def compare_reference(
    config: ExpertConfig,
    rank_by_qid: Mapping[str, int],
    n_ranked: int,
    *,
    top_n: int = 50,
) -> list[dict[str, Any]]:
    """Compare a ranking with each external list.

    Ordered lists get a Spearman correlation between list position and our
    rank over members we ranked (NaN below three matches); list positions are
    used for correlation only and are not claimed to be published ranks. Unordered lists get
    the share of all members that fall inside our top ``top_n`` and the mean
    percentile rank (rank / n_ranked, lower is better) of matched members.

    Args:
        config: Parsed targets.
        rank_by_qid: Wikidata Q-id to rank (1 = best).
        n_ranked: Number of ranked generals.
        top_n: Cutoff for the top-N share.

    Returns:
        One dict per list: name, kind, n_members, n_matched, our_ranks,
        unmatched, caveat, plus kind-specific metrics.
    """
    out: list[dict[str, Any]] = []
    for ref in (*config.reference_rankings, *config.expert_lists):
        our: dict[str, int] = {}
        unmatched: list[str] = []
        for key in ref.members:
            rank = rank_by_qid.get(config.generals[key])
            if rank is None:
                unmatched.append(key)
            else:
                our[key] = rank
        row: dict[str, Any] = {
            "name": ref.name,
            "kind": ref.kind,
            "source": ref.source,
            "caveat": ref.caveat,
            "n_members": len(ref.members),
            "n_matched": len(our),
            "our_ranks": our,
            "unmatched": unmatched,
        }
        if ref.kind == "ordered":
            positions = [float(ref.members.index(k) + 1) for k in our]
            row["spearman"] = _spearman(positions, [float(r) for r in our.values()])
        else:
            ranks = list(our.values())
            row["top_n"] = top_n
            row["share_in_top_n"] = (
                sum(r <= top_n for r in ranks) / len(ref.members) if ref.members else math.nan
            )
            row["mean_percentile_rank"] = (
                sum(ranks) / len(ranks) / n_ranked if ranks and n_ranked > 0 else math.nan
            )
        out.append(row)
    logger.info("expert_reference_compared", lists=len(out))
    return out
