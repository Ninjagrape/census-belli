"""
Applying the resolve stage's manual overrides.

``config/overrides.yaml`` carries two kinds this stage owns (see
:mod:`pipeline.overrides` for the file format and the loader):

``commander_link``
    One commander mention, named by battle, side and the name extract wrote,
    is linked to a Q-id or, with the value ``none``, to nobody.
``general_merge``
    Every group that resolved to one general (a Q-id or a corpus-local
    canonical name) is relinked to a final Q-id.

Both are applied *during* resolution rather than after the database write, so
the lifespan guard, the gap split and the stale-link prune see the final link
set, and a re-run reproduces the correction instead of undoing it.

**Mechanism.** :meth:`ResolveOverrides.take_links` pulls the matching mentions
out of their name groups into groups of their own, one per entry, each with a
fixed :class:`~pipeline.resolvers.records.Decision` that the stage seeds into
its resolver. ``_GroupResolver._decide`` skips a group that already has a
decision, so these groups never reach the lifespan guard, the fuzzy matcher,
the corpus matcher or the LLM, and no disambiguation request is generated for
them. The stage also keeps them out of the corpus-anchor pass: a correction
for one mention must not pull other mentions in by name.
:meth:`ResolveOverrides.apply_merges` then relinks decisions after matching.

A manual link bypasses the lifespan guard on purpose. The event
``resolve_manual_link`` records that, and whether the guard would have
rejected the link, so the bypass is visible rather than silent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Final

import structlog

from pipeline.overrides import OverrideEntry, OverrideReport, target_key
from pipeline.resolvers.candidates import QidLookupSource, is_unlabelled
from pipeline.resolvers.matcher import partition_by_lifespan
from pipeline.resolvers.names import clean_surface, name_keys
from pipeline.resolvers.records import Candidate, Decision, Mention, MentionGroup

__all__ = ["MANUAL_CONFIDENCE", "NONE_VALUE", "ResolveOverrides"]

logger = structlog.get_logger()

# A curated, cited correction is the strongest evidence the stage ever has.
MANUAL_CONFIDENCE: Final[float] = 1.0

# The ``commander_link`` value that unlinks a mention.
NONE_VALUE: Final[str] = "none"

_MANUAL_KEY_PREFIX: Final[str] = "manual:"


def _norm(value: str) -> str:
    """Compare form of a slug, side label or mention name: strip + casefold."""
    return value.strip().casefold()


def _name_forms(mentions: Sequence[Mention]) -> tuple[list[str], tuple[str, ...]]:
    """The surface forms and match keys a set of mentions carries."""
    forms = sorted({clean_surface(m.name) or m.name.strip() for m in mentions})
    keys: list[str] = []
    for mention in mentions:
        for key in name_keys(mention.name):
            if key not in keys:
                keys.append(key)
    return forms, tuple(keys)


def _with_mentions(group: MentionGroup, mentions: list[Mention]) -> MentionGroup:
    """Copy a group so it carries only the given mentions' names.

    Args:
        group: The group mentions were taken out of.
        mentions: The mentions it keeps.

    Returns:
        A copy whose ``surface_forms`` and ``keys`` come from ``mentions``
        alone, so a general linked to this part is not given the removed
        mention's spelling as an alias. ``display_name`` is left alone: the
        LLM prompt is rendered from it and must stay byte-stable.
    """
    forms, keys = _name_forms(mentions)
    return replace(group, mentions=mentions, surface_forms=forms, keys=keys or group.keys)


def _manual_group(entry: OverrideEntry, mentions: list[Mention]) -> MentionGroup:
    """Build the group an entry's mentions are resolved as.

    Args:
        entry: A ``commander_link`` entry.
        mentions: The mentions it matched.

    Returns:
        A group keyed on the entry id, so it can never merge with a name group.
    """
    forms, keys = _name_forms(mentions)
    key = f"{_MANUAL_KEY_PREFIX}{entry.id}"
    return MentionGroup(
        key=key,
        keys=keys or (key,),
        display_name=forms[0],
        surface_forms=forms,
        mentions=mentions,
        confidence=MANUAL_CONFIDENCE,
        method="manual",
    )


def _linked_decision(entry: OverrideEntry, candidate: Candidate, reasoning: str) -> Decision:
    """A final link to a fetched entity, carrying the entity data a normal link gets."""
    return Decision(
        status="linked",
        canonical_name=candidate.label,
        method="manual",
        qid=candidate.qid,
        confidence=MANUAL_CONFIDENCE,
        reasoning=reasoning,
        candidate=candidate,
        override_id=entry.id,
    )


def _candidate_for(source: object, qid: str) -> Candidate | None:
    """Fetch an entity by Q-id from a source, if the source can."""
    if not isinstance(source, QidLookupSource):
        return None
    return source.candidate_by_qid(qid)


def _named(candidate: Candidate, name: str) -> Candidate:
    """Give an unlabelled entity the editor's name for it.

    Args:
        candidate: A fetched entity.
        name: The name the override supplies.

    Returns:
        The candidate with ``name`` as its label when it had no English label,
        otherwise the candidate unchanged.
    """
    return replace(candidate, label=name) if is_unlabelled(candidate) else candidate


class ResolveOverrides:
    """The resolve-stage overrides of one run, and what became of each.

    Args:
        entries: The ``commander_link`` and ``general_merge`` entries.
        report: Receives exactly one ``applied`` or ``unmatched`` per entry.
    """

    def __init__(self, entries: Iterable[OverrideEntry], report: OverrideReport) -> None:
        entry_list = list(entries)
        self._links = [e for e in entry_list if e.kind == "commander_link"]
        self._merges = [e for e in entry_list if e.kind == "general_merge"]
        self._report = report
        # Groups an entry governs. A merge never touches them.
        self._manual_keys: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self._links or self._merges)

    @property
    def manual_keys(self) -> frozenset[str]:
        """Keys of the groups a ``commander_link`` entry decided."""
        return frozenset(self._manual_keys)

    def required_qids(self, groups: Sequence[MentionGroup]) -> list[str]:
        """List the entities the run must fetch by Q-id.

        Args:
            groups: The grouped mentions, before anything is taken out.

        Returns:
            The Q-ids of link entries that match a mention, and of every merge
            target, deduplicated in file order. An entry matching nothing needs
            no fetch: it will be reported unmatched.
        """
        wanted: list[str] = []
        for entry in self._links:
            qid = str(entry.value)
            if qid != NONE_VALUE and self._matching(groups, entry):
                wanted.append(qid)
        wanted.extend(str(e.value) for e in self._merges)
        return list(dict.fromkeys(wanted))

    def take_links(
        self, groups: list[MentionGroup], source: object
    ) -> tuple[list[MentionGroup], dict[str, Decision]]:
        """Pull each ``commander_link`` entry's mentions out into a fixed group.

        Args:
            groups: The name groups. Edited in place: matched mentions are
                removed, and groups left empty are dropped.
            source: The candidate source, asked for each link target's entity.

        Returns:
            The new groups, and their decisions keyed by group key. A link
            whose entity cannot be fetched is reported unmatched and its
            mentions are left in their name groups, so a failed fetch never
            deletes an existing link.
        """
        manual: list[MentionGroup] = []
        decisions: dict[str, Decision] = {}

        for entry in self._links:
            mentions = self._matching(groups, entry)
            if not mentions:
                self._report.unmatched(
                    entry,
                    f"no mention {entry.target['mention']!r} on side {entry.target['side']!r} "
                    f"of battle {entry.target['battle']!r} in the extracted commanders",
                )
                continue

            qid = str(entry.value)
            group = _manual_group(entry, mentions)
            if qid == NONE_VALUE:
                decision = Decision(
                    status="unresolved",
                    canonical_name=group.display_name,
                    method="manual",
                    reasoning=f"commander_link {entry.id}: unlinked by override",
                    override_id=entry.id,
                )
            else:
                candidate = _candidate_for(source, qid)
                if candidate is None:
                    self._report.unmatched(
                        entry,
                        f"{qid} could not be fetched as a human with a label; "
                        "the mention keeps its automatic resolution",
                    )
                    continue
                if is_unlabelled(candidate):
                    # The editor named the person; the label service had no English
                    # label to give, and a Q-id must never become a canonical name.
                    candidate = _named(candidate, group.display_name)
                    logger.info(
                        "resolve_manual_link_unlabelled",
                        id=entry.id,
                        qid=qid,
                        name=candidate.label,
                    )
                decision = _linked_decision(
                    entry, candidate, f"commander_link {entry.id}: linked to {qid} by override"
                )
                self._log_link(entry, qid, mentions, candidate)

            self._remove(groups, mentions)
            manual.append(group)
            decisions[group.key] = decision
            self._manual_keys.add(group.key)
            self._report.applied(entry)

        return manual, decisions

    def apply_merges(
        self,
        groups: Sequence[MentionGroup],
        decisions: dict[str, Decision],
        source: object,
    ) -> None:
        """Relink every group that resolved to a merge source onto its target.

        Args:
            groups: Every group the run resolved.
            decisions: Their final decisions; relinked entries are replaced.
            source: The candidate source, asked for each merge target's entity.
        """
        for entry in self._merges:
            source_key = target_key(entry)[1]
            matched = [
                g
                for g in groups
                if g.key not in self._manual_keys
                and g.key in decisions
                and self._resolves_to(decisions[g.key], source_key)
            ]
            if not matched:
                self._report.unmatched(
                    entry, f"no resolved group links to or is named {entry.target['general']!r}"
                )
                continue
            target_qid = str(entry.value)
            candidate = _candidate_for(source, target_qid)
            if candidate is None:
                self._report.unmatched(
                    entry,
                    f"merge target {target_qid} could not be fetched as a human with a label",
                )
                continue
            for group in matched:
                previous = decisions[group.key]
                named = _named(candidate, previous.canonical_name)
                if named is not candidate:
                    logger.info(
                        "resolve_manual_link_unlabelled",
                        id=entry.id,
                        qid=target_qid,
                        name=named.label,
                    )
                decisions[group.key] = _linked_decision(
                    entry,
                    named,
                    f"general_merge {entry.id}: {previous.canonical_name!r} "
                    f"({previous.qid or 'corpus-local'}) merged into {target_qid}",
                )
                logger.info(
                    "resolve_manual_merge",
                    id=entry.id,
                    source=entry.target["general"],
                    qid=target_qid,
                    group=group.display_name,
                    mentions=len(group.mentions),
                )
            self._report.applied(entry)

    @staticmethod
    def _resolves_to(decision: Decision, source_key: str) -> bool:
        """Whether a decision names the merge source.

        Args:
            decision: A group's final decision.
            source_key: The entry's normalised ``general`` key.

        Returns:
            True for a link to that Q-id, or a corpus-local identity whose
            canonical name folds to the key.
        """
        if decision.status == "linked" and decision.qid:
            return decision.qid.upper() == source_key
        if decision.status == "new" and not decision.qid:
            return _norm(decision.canonical_name) == source_key
        return False

    @staticmethod
    def _matching(groups: Sequence[MentionGroup], entry: OverrideEntry) -> list[Mention]:
        """Find the mentions a ``commander_link`` entry targets."""
        battle = _norm(entry.target["battle"])
        side = _norm(entry.target["side"])
        name = _norm(entry.target["mention"])
        return [
            m
            for group in groups
            for m in group.mentions
            if _norm(m.battle_slug) == battle
            and _norm(m.side_label) == side
            and _norm(m.name) == name
        ]

    @staticmethod
    def _remove(groups: list[MentionGroup], mentions: list[Mention]) -> None:
        """Take mentions out of their groups, dropping any group left empty."""
        taken = {id(m) for m in mentions}
        rebuilt: list[MentionGroup] = []
        for group in groups:
            kept = [m for m in group.mentions if id(m) not in taken]
            if len(kept) == len(group.mentions):
                rebuilt.append(group)
            elif kept:
                rebuilt.append(_with_mentions(group, kept))
        groups[:] = rebuilt

    @staticmethod
    def _log_link(
        entry: OverrideEntry, qid: str, mentions: list[Mention], candidate: Candidate
    ) -> None:
        """Log a manual link, and whether the lifespan guard would have rejected it."""
        _, detached = partition_by_lifespan(mentions, candidate)
        logger.info(
            "resolve_manual_link",
            id=entry.id,
            qid=qid,
            battle=entry.target["battle"],
            mention=entry.target["mention"],
            mentions=len(mentions),
            guard_would_reject=bool(detached),
            guard_bypassed=True,
        )
