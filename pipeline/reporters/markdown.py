"""The static report, ``report.md``, rendered with ``string.Template``."""

from __future__ import annotations

import re
from collections.abc import Sequence
from string import Template
from typing import Final

from pipeline.reporters.records import EvaluationInfo
from pipeline.reporters.tables import RankedGeneral
from pipeline.reporters.view import ReportView

__all__ = ["render_markdown"]

UNDER_LIST_LIMIT: Final[int] = 200
NA: Final[str] = "n/a"
CREDITED_NOTE: Final[str] = (
    "Credited battles counts the battles in which the general has any positive share of "
    "credit. Effective battles (in rankings.csv) is the sum of those shares, so it can be lower."
)
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|~])")

_PAGE = Template(
    """# General WAR rankings, model run ${run_id}
${banner}
Generals are ranked by expected total WAR (wins above replacement), in the order the model \
stage stored. This report reads those numbers and recomputes none of them; only the rank \
overlap flag is derived here, from the stored WAR intervals. Skill and WAR intervals are the \
stored ${level} credible intervals; the WAR sd is Rubin-pooled.

## Headline ranking

${headline_intro}

${headline_table}

${headline_note}

## Generals without a rank

${under_intro}

${under_table}

## Evaluation

${evaluation}

## Figures

${figures}

## Disclosures

${disclosures}
"""
)


def _esc(value: str) -> str:
    """Escape a name for a Markdown table cell and for inline HTML."""
    plain = " ".join(value.split())
    plain = plain.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_SPECIAL.sub(r"\\\1", plain)


def _num(value: float | None, digits: int = 2) -> str:
    return NA if value is None else f"{value:.{digits}f}"


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}]"


def _war_interval(g: RankedGeneral) -> str:
    if g.war_lo is None or g.war_hi is None:
        return NA
    text = _ci(g.war_lo, g.war_hi)
    return f"{text} (approx.)" if g.war_interval_approx else text


def _banner(view: ReportView) -> str:
    if not view.provisional:
        return ""
    items = "\n".join(f"- {_esc(r)}" for r in view.reasons)
    return (
        "\n> **PROVISIONAL. Do not publish this ranking.** It was produced, but these checks "
        f"did not pass:\n>\n{_quote(items)}\n"
    )


def _quote(block: str) -> str:
    return "\n".join(f"> {line}" for line in block.splitlines())


def _overlap_cell(value: bool | None) -> str:
    return NA if value is None else ("yes" if value else "no")


def _headline_table(top: Sequence[RankedGeneral], show_overlap: bool) -> str:
    head = (
        "| Rank | General | Credited battles | Expected WAR | WAR sd | WAR (95% interval) "
        "| P(above next ranked) | Skill (95% CI) |"
    )
    rule = "|---:|---|---:|---:|---:|---|---:|---|"
    if show_overlap:
        head, rule = head + " Overlaps next listed general |", rule + "---|"
    lines = [head, rule]
    for g in top:
        e = g.estimate
        cells = [
            NA if g.rank is None else str(g.rank),
            _esc(e.name),
            str(e.n_battles),
            _num(e.total_war),
            _num(e.war_sd),
            _war_interval(g),
            _num(g.p_above_next, 3),
            f"{e.skill_mean:.2f} {_ci(e.skill_ci_lo, e.skill_ci_hi)}",
        ]
        if show_overlap:
            cells.append(_overlap_cell(g.overlaps_next))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _under_table(under: Sequence[RankedGeneral]) -> str:
    lines = ["| General | Credited battles | Skill (95% CI) |", "|---|---:|---|"]
    for g in under[:UNDER_LIST_LIMIT]:
        e = g.estimate
        lines.append(
            f"| {_esc(e.name)} | {e.n_battles} | "
            f"{e.skill_mean:.2f} {_ci(e.skill_ci_lo, e.skill_ci_hi)} |"
        )
    if len(under) > UNDER_LIST_LIMIT:
        lines.append(
            f"\nThe first {UNDER_LIST_LIMIT} of {len(under)} are listed; see rankings.csv."
        )
    return "\n".join(lines)


def _evaluation_section(view: ReportView) -> str:
    ev = view.evaluation
    if ev is None:
        return f"Not available: {_esc(view.evaluation_reason or 'no evaluation report was read')}."
    lines: list[str] = []
    if ev.cv_accuracy is not None:
        n = f" over {ev.cv_n} held-out battles" if ev.cv_n else ""
        lines.append(
            f"Held-out accuracy {_num(ev.cv_accuracy, 3)}, Brier score {_num(ev.cv_brier, 3)}{n}."
        )
    lines.append(_gate_table(ev))
    lines.append(_variant_table(ev))
    return "\n\n".join(x for x in lines if x)


def _gate_table(ev: EvaluationInfo) -> str:
    if not ev.gates:
        return ""
    rows = [
        f"| {_esc(str(g.get('name', '')))} | {'pass' if g.get('passed') else 'FAIL'} | "
        f"{_esc(str(g.get('message', '')))} |"
        for g in ev.gates
    ]
    return "\n".join(["| Evaluation gate | Result | Detail |", "|---|---|---|", *rows])


def _variant_table(ev: EvaluationInfo) -> str:
    if not ev.variants:
        return ""
    rows = [
        f"| {_esc(v.name)} | {_esc(v.status)} | {_num(v.spearman_rho, 3)} | "
        f"{NA if v.top_overlap is None else v.top_overlap} |"
        for v in ev.variants
    ]
    head = ["| Sensitivity variant | Status | Spearman rho vs baseline | Top-set overlap |"]
    return "\n".join([*head, "|---|---|---:|---:|", *rows])


def _figure_lines(figure_names: Sequence[str]) -> str:
    if not figure_names:
        return "No figures are available."
    return "\n\n".join(f"![{name.removesuffix('.png')}](figures/{name})" for name in figure_names)


def render_markdown(view: ReportView, figure_names: Sequence[str]) -> str:
    """Render ``report.md``.

    Args:
        view: The report view.
        figure_names: File names under ``figures/`` to embed, forward-slash
            relative, own figures first.

    Returns:
        The Markdown text.
    """
    p = view.params
    top = view.top
    intro = (
        f"Top {len(top)} of {len(view.headline)} generals with at least {p.min_battles} "
        "credited battles."
    )
    if p.show_rank_overlap:
        intro += (
            " The last column flags whether a general's WAR interval overlaps the next listed "
            f"general's by more than {p.overlap_threshold:.0%} of the narrower interval. n/a "
            "means there is no next listed general or an interval is missing. "
            "P(above next ranked) compares each general with the next listed row; the "
            "disclosures say how it was computed."
        )
        if view.uses_normal_approx:
            intro += (
                " This run stored no WAR credible interval, so intervals marked approx. are "
                f"total WAR plus or minus {view.z:.2f} times the stored WAR sd, a normal "
                "approximation and not a credible interval."
            )
    under_intro = (
        f"{len(view.under_determined)} generals have fewer than {p.min_battles} credited "
        "battles. Their skill is too weakly determined to rank, so they are listed without one."
    )
    return _PAGE.substitute(
        run_id=view.model_run_id,
        banner=_banner(view),
        level=f"{p.ci_level:.0%}",
        headline_intro=intro,
        headline_table=_headline_table(top, p.show_rank_overlap),
        headline_note=CREDITED_NOTE,
        under_intro=under_intro,
        under_table=_under_table(view.under_determined),
        evaluation=_evaluation_section(view),
        figures=_figure_lines(figure_names),
        # Evaluate's disclosures are file input, so they are escaped like names.
        disclosures="\n".join(f"- {_esc(d)}" for d in view.disclosures),
    )
