"""
Merge what several sources said about one battle into one record.

Two principles govern this module.

The first is that structured sources outrank LLM extraction. Wikidata's
date is curated; the infobox's date was typed by an editor into a labelled
field; the body's date was read out of prose by a model. When they differ,
the more structured one is written -- but the disagreement is recorded, not
discarded, because a battle whose sources contradict each other is exactly
the battle a reviewer should see.

The second is that troop numbers are never reconciled here. Every
conflicting figure becomes its own ``troop_reports`` row. The reconcile
stage fits a source-disagreement model that estimates per-source bias --
ancient sources inflate, and the size of that inflation is something the
model learns from seeing every report side by side. Picking a winner here
would delete the evidence that stage exists to use.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Final

import structlog

from pipeline.extractors.records import (
    BattleExtraction,
    BattleFacts,
    CasualtyReport,
    CommanderMention,
    Disagreement,
    ExtractionFailure,
    MissingField,
    SideExtraction,
    SourceExtraction,
    TroopReport,
)
from pipeline.extractors.textnorm import normalise_label

__all__ = [
    "SOURCE_PRIORITY",
    "merge_battle",
    "missing_fields",
]

logger = structlog.get_logger()

# Higher wins. Structured extraction beats LLM extraction; among the LLM
# sources, the article body beats a fetched citation, because the citation
# pass reads pages that are often about the war rather than the battle.
SOURCE_PRIORITY: Final[dict[str, int]] = {
    "wikidata": 40,
    "dbpedia": 35,
    "wikipedia_infobox": 30,
    "wikipedia_body": 20,
    "web_secondary": 10,
    "primary_source": 5,
}

# Scalar fields merged by priority. Lists and side-level data are merged by
# union instead, further down.
_SCALAR_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "wikidata_id",
    "wikipedia_url",
    "date_start",
    "date_end",
    "date_precision",
    "latitude",
    "longitude",
    "location_name",
    "battle_type",
    "fortified",
    "weather",
    "victor",
    "outcome_level",
    "outcome_evidence",
)

# Disagreement on these is worth a human's time; disagreement on the others
# is usually two sources spelling a place differently.
_REVIEWABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {"date_start", "outcome_level", "victor", "wikidata_id"}
)

# Fields whose absence blocks the model rather than merely weakening it.
_REQUIRED_BATTLE_FIELDS: Final[tuple[str, ...]] = ("date_start", "location_name", "outcome_level")

# Battle-level covariates in the model's linear predictor (agents/model.yaml:
# beta_type, terrain, defensive advantage). Their absence weakens the fit
# rather than blocking it, but impute has to know they were absent, so they are
# logged with a note that distinguishes them from the required set above.
#
# Deliberately excludes weather, which is extracted and stored but read by no
# stage. Logging every field that happens to be NULL would bury the missingness
# that matters under a row per ancient battle for a field nothing consumes.
#
# battle_type is expected to be missing on most battles until classify infers
# it: the generic infobox stopped asserting a type once it emerged that
# Wikipedia uses that template for naval battles too. Those rows are the
# standing record of that gap rather than a surprise.
_COVARIATE_BATTLE_FIELDS: Final[tuple[str, ...]] = ("battle_type", "terrain", "fortified")

# Common suffixes dropped when matching a side across sources, so the
# infobox's "French Empire" and the body's "France" land on one side.
_SIDE_SUFFIXES: Final[tuple[str, ...]] = (
    "empire",
    "kingdom",
    "republic",
    "forces",
    "army",
    "navy",
    "fleet",
    "coalition",
    "alliance",
    "confederacy",
    "state",
    "states",
)

_VICTORY_MIRROR: Final[dict[str, str]] = {
    "decisive_victory": "decisive_defeat",
    "victory": "defeat",
    "pyrrhic_victory": "defeat",
    "indecisive": "indecisive",
}


def _priority(source_type: str) -> int:
    """Rank a source type.

    Args:
        source_type: A ``source_type`` enum value.

    Returns:
        Its priority; unknown types rank below every known one.
    """
    return SOURCE_PRIORITY.get(source_type, 0)


def _side_key(label: str) -> str:
    """Build a comparison key for a side label.

    Args:
        label: A raw side label.

    Returns:
        A normalised key with common polity suffixes removed.
    """
    key = normalise_label(label)
    tokens = [t for t in key.split() if t not in _SIDE_SUFFIXES]
    return " ".join(tokens) or key


def _match_side(label: str, existing: Iterable[str]) -> str | None:
    """Find which already-seen side a label refers to.

    Exact match on the stripped key first, then a containment test, which
    catches "France" against "french" only when one key is a token subset of
    the other. Anything looser merges genuinely different belligerents, which
    is worse than carrying a duplicate side into the resolve stage.

    Args:
        label: The label to place.
        existing: Keys already in the merged record.

    Returns:
        The matching key, or None when the label is a new side.
    """
    key = _side_key(label)
    if not key:
        return None

    keys = list(existing)
    if key in keys:
        return key

    tokens = set(key.split())
    for candidate in keys:
        candidate_tokens = set(candidate.split())
        if not candidate_tokens or not tokens:
            continue
        if tokens <= candidate_tokens or candidate_tokens <= tokens:
            return candidate

    return None


def _merge_scalars(
    sources: Sequence[SourceExtraction],
) -> tuple[BattleFacts, list[Disagreement]]:
    """Choose one value per scalar field and record every dissent.

    Args:
        sources: Every source's view of the battle.

    Returns:
        The merged facts and the disagreements found.
    """
    merged = BattleFacts()
    disagreements: list[Disagreement] = []

    ordered = sorted(sources, key=lambda s: _priority(s.provenance.source_type), reverse=True)

    for field_name in _SCALAR_FIELDS:
        offered: list[tuple[str, Any]] = []
        for source in ordered:
            value = getattr(source.facts, field_name)
            if value is not None and value != "":
                offered.append((source.provenance.source_type, value))

        if not offered:
            continue

        chosen_source, chosen = offered[0]
        setattr(merged, field_name, chosen)

        rejected = {
            source_type: str(value)
            for source_type, value in offered[1:]
            if str(value).strip().lower() != str(chosen).strip().lower()
        }
        if rejected and field_name in _REVIEWABLE_FIELDS:
            disagreements.append(
                Disagreement(
                    field_name=field_name,
                    chosen=str(chosen),
                    chosen_source=chosen_source,
                    rejected=rejected,
                )
            )

    # List-valued fields are a union: two sources naming different terrain
    # features are both right far more often than either is wrong.
    terrain: list[str] = []
    part_of: list[str] = []
    for source in ordered:
        for item in source.facts.terrain:
            if item.lower() not in {t.lower() for t in terrain}:
                terrain.append(item)
        for item in source.facts.part_of:
            if item not in part_of:
                part_of.append(item)
    merged.terrain = terrain
    merged.part_of = part_of

    return merged, disagreements


def _merge_commanders(mentions: Sequence[CommanderMention]) -> list[CommanderMention]:
    """Collapse repeated mentions of the same person on one side.

    Args:
        mentions: Every commander mention attributed to one side.

    Returns:
        One mention per person, keeping the most specific role and the
        evidence that supported it, with a count of how often they appeared.
    """
    by_name: dict[str, CommanderMention] = {}

    for mention in mentions:
        key = normalise_label(mention.name)
        if not key:
            continue
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = mention
            continue

        existing.mentions += 1
        # An infobox lists presence; the body pass says what someone did. So
        # a stated role replaces "unclear" regardless of source priority.
        if existing.apparent_role == "unclear" and mention.apparent_role != "unclear":
            existing.apparent_role = mention.apparent_role
            existing.role_evidence = mention.role_evidence
        elif not existing.role_evidence and mention.role_evidence:
            existing.role_evidence = mention.role_evidence
        if not existing.note and mention.note:
            existing.note = mention.note

    return list(by_name.values())


def _dedupe_troop_reports(reports: Sequence[TroopReport]) -> list[TroopReport]:
    """Drop reports that repeat the same claim from the same source.

    Reports from *different* sources are never deduplicated, even when the
    numbers agree: the reconcile stage estimates per-source bias and needs
    to know that two sources independently said 30,000. Only an identical
    claim from the same document is dropped, which happens when an article
    repeats a figure in two passages.

    Args:
        reports: Every troop report attributed to one side.

    Returns:
        The reports with same-source duplicates removed, in input order.
    """
    seen: set[tuple[str, str, float, str, bool, bool]] = set()
    kept: list[TroopReport] = []

    for report in reports:
        key = (
            report.provenance.source_type,
            report.provenance.source_ref,
            report.reported_value,
            report.branch,
            report.is_lower_bound,
            report.is_upper_bound,
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(report)

    return kept


def _dedupe_casualty_reports(reports: Sequence[CasualtyReport]) -> list[CasualtyReport]:
    """Drop casualty reports that repeat the same claim from the same source.

    Args:
        reports: Every casualty report attributed to one side.

    Returns:
        The reports with same-source duplicates removed.
    """
    seen: set[tuple[str, str, float, str]] = set()
    kept: list[CasualtyReport] = []

    for report in reports:
        key = (
            report.provenance.source_type,
            report.provenance.source_ref,
            report.reported_value,
            report.casualty_type,
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(report)

    return kept


def _merge_sides(sources: Sequence[SourceExtraction]) -> list[SideExtraction]:
    """Union every source's sides, matching them across sources by label.

    Args:
        sources: Every source's view of the battle.

    Returns:
        The merged sides, ordered as the highest-priority source listed them.
    """
    ordered = sorted(sources, key=lambda s: _priority(s.provenance.source_type), reverse=True)

    merged: dict[str, SideExtraction] = {}

    for source in ordered:
        for side in source.sides:
            key = _match_side(side.label, merged)
            if key is None:
                key = _side_key(side.label) or side.label.lower()
                merged[key] = SideExtraction(
                    label=side.label,
                    aliases=list(side.aliases),
                    polity=side.polity,
                    outcome=side.outcome,
                )
            target = merged[key]
            if side.label != target.label and side.label not in target.aliases:
                target.aliases.append(side.label)
            for alias in side.aliases:
                if alias != target.label and alias not in target.aliases:
                    target.aliases.append(alias)
            target.commanders.extend(side.commanders)
            target.troop_reports.extend(side.troop_reports)
            target.casualty_reports.extend(side.casualty_reports)

    for side in merged.values():
        side.commanders = _merge_commanders(side.commanders)
        side.troop_reports = _dedupe_troop_reports(side.troop_reports)
        side.casualty_reports = _dedupe_casualty_reports(side.casualty_reports)

    return list(merged.values())


def _assign_outcomes(sides: Sequence[SideExtraction], facts: BattleFacts) -> None:
    """Write a per-side outcome from the battle-level victor, where it is clear.

    Args:
        sides: The merged sides, modified in place.
        facts: The merged battle facts.
    """
    level = facts.outcome_level
    if level is None:
        return

    if level == "indecisive":
        for side in sides:
            side.outcome = side.outcome or "indecisive"
        return

    victor = facts.victor
    if not victor:
        return

    victor_key = _side_key(victor)
    victor_tokens = set(victor_key.split())
    if not victor_tokens:
        return

    winners = [
        side
        for side in sides
        if victor_tokens & set(_side_key(side.label).split())
        or any(victor_tokens & set(_side_key(a).split()) for a in side.aliases)
    ]

    # Exactly one side must match, or the label is ambiguous ("Allied
    # victory" in a battle with three allied contingents) and guessing which
    # side won is not something to do silently.
    if len(winners) != 1:
        logger.info(
            "victor_label_not_matched_to_one_side",
            victor=victor,
            sides=[s.label for s in sides],
            matched=len(winners),
        )
        return

    for side in sides:
        side.outcome = level if side is winners[0] else _VICTORY_MIRROR.get(level)


def missing_fields(battle: BattleExtraction) -> list[MissingField]:
    """List everything no source supplied for a battle.

    Missing data is logged rather than imputed here; the impute stage needs
    to know what was absent and the classify stage needs to label why.

    Args:
        battle: A merged battle.

    Returns:
        One entry per missing field, battle-level entries first.
    """
    missing: list[MissingField] = []

    for field_name in _REQUIRED_BATTLE_FIELDS:
        if getattr(battle.facts, field_name) is None:
            missing.append(MissingField(field_name=field_name, notes="no source reported it"))

    for field_name in _COVARIATE_BATTLE_FIELDS:
        value = getattr(battle.facts, field_name)
        # An empty terrain list is an absence; fortified=False is an answer,
        # so absence is tested explicitly rather than by truthiness.
        if value is None or (isinstance(value, list) and not value):
            missing.append(
                MissingField(
                    field_name=field_name,
                    notes="model covariate; no source reported it",
                )
            )

    if len(battle.sides) < 2:
        missing.append(
            MissingField(
                field_name="battle_sides",
                notes=f"only {len(battle.sides)} side(s) extracted; a battle needs at least 2",
            )
        )

    for side in battle.sides:
        if not side.troop_reports:
            missing.append(MissingField(field_name="troop_total", side_label=side.label))
        elif all(r.scope == "unknown" for r in side.troop_reports):
            missing.append(
                MissingField(
                    field_name="troop_scope",
                    side_label=side.label,
                    notes="every troop report has unknown scope; not usable as a force ratio",
                )
            )
        if not side.casualty_reports:
            missing.append(MissingField(field_name="casualties", side_label=side.label))
        if not side.commanders:
            missing.append(MissingField(field_name="commanders", side_label=side.label))
        if side.outcome is None:
            missing.append(MissingField(field_name="outcome", side_label=side.label))

    return missing


def merge_battle(
    slug: str,
    sources: Sequence[SourceExtraction],
    *,
    failures: Sequence[ExtractionFailure] = (),
    fallback_name: str = "",
) -> BattleExtraction:
    """Merge every source's view of one battle into a single record.

    Args:
        slug: The battle's stable identifier in ``data/raw``.
        sources: What each extractor produced for this battle.
        failures: Extraction attempts that produced nothing, carried through
            so the record can be flagged for review.
        fallback_name: Name to use when no source supplied one.

    Returns:
        The merged battle, with disagreements and missing fields recorded.
    """
    facts, disagreements = _merge_scalars(sources)
    sides = _merge_sides(sources)
    _assign_outcomes(sides, facts)

    notes: list[str] = []
    for source in sources:
        notes.extend(source.notes)

    battle = BattleExtraction(
        slug=slug,
        name=facts.name or fallback_name or slug,
        facts=facts,
        sides=sides,
        disagreements=disagreements,
        failures=list(failures),
        notes=notes,
        # Counted by (type, document): the infobox and the body of one
        # article are separate sources, since an editor typed one and a model
        # read the other, and they can disagree.
        n_sources=len({(s.provenance.source_type, s.provenance.source_ref) for s in sources}),
    )
    battle.missing = missing_fields(battle)

    logger.debug(
        "battle_merged",
        slug=slug,
        sources=battle.n_sources,
        sides=len(sides),
        troop_reports=sum(len(s.troop_reports) for s in sides),
        disagreements=len(disagreements),
        missing=len(battle.missing),
    )

    return battle
