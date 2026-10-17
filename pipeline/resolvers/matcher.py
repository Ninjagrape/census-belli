"""
Deciding which Wikidata entity a commander mention refers to.

This is steps 1 and 2 of ``agents/resolve.yaml``, and it decides most of the
corpus. The LLM step exists only for what is left.

Two failure modes matter, and they are not symmetric:

**A false merge** links two different commanders to one entity. Their records
pool, and the model fits one skill parameter to two people's battles. Nothing
downstream can detect it: the merged record looks like an ordinary prolific
commander.

**A false split** gives one commander two entities. Their record halves, both
halves are noisier, and both shrink harder towards the replacement level. That
is a loss of power, and it is visible -- two near-identical names in the
ranking.

So the matcher is built to prefer splits. A match must clear the fuzzy
threshold *and* be compatible with the battle's date *and* beat its runner-up
by a margin; anything else is handed to the LLM or made a new entity.

The date gate is the one that does the real work. Name similarity alone links
"Hannibal" in an 1811 battle to Hannibal Barca, and "Scipio" to whichever of
the five the query service returned first. A candidate who was not alive
cannot have commanded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import structlog
from rapidfuzz import fuzz

from pipeline.resolvers.candidates import candidate_keys
from pipeline.resolvers.names import fold
from pipeline.resolvers.records import Candidate, Decision, Identity, Mention, MentionGroup

__all__ = [
    "AMBIGUITY_MARGIN",
    "LIFESPAN_SLACK_YEARS",
    "LLM_FLOOR_BELOW_THRESHOLD",
    "MAX_PLAUSIBLE_AGE_YEARS",
    "MIN_COMMAND_AGE_YEARS",
    "best_score",
    "career_clusters",
    "command_window",
    "lifespan_verdict",
    "partition_by_lifespan",
    "match_against_identities",
    "match_group",
    "rank_for_prompt",
    "viable_candidates",
]

logger = structlog.get_logger()

# Qids whose contradictory lifespan has been logged, so the warning appears once
# per candidate rather than once per mention the window is asked about.
_CONTRADICTORY_LOGGED: set[str] = set()


def reset_contradictory_log() -> None:
    """Forget which contradictory lifespans were logged (a test hook)."""
    _CONTRADICTORY_LOGGED.clear()


# Dates in this corpus are imprecise, and a commander who died in the battle
# is the normal case rather than the exception, so the window is opened at
# both ends by a couple of years before anything is rejected on it.
LIFESPAN_SLACK_YEARS: Final[int] = 2

# How long after a known birth, or before a known death, a candidate may still
# have commanded when the other end of their life is unrecorded. Wikidata has a
# great many half-dated ancients, and treating the missing end as open made
# them eligible for every later battle in history. Generous on purpose: this
# bounds a lifespan to a century, it does not judge whether an age is likely.
MAX_PLAUSIBLE_AGE_YEARS: Final[int] = 100

# Nobody commands before this age. The window of a candidate with a known birth
# opens at birth plus this, so a namesake born ten years before a battle is not
# credited with it. Low on purpose: boy kings and nominal commanders exist, and
# this gate separates generations, it does not judge competence.
MIN_COMMAND_AGE_YEARS: Final[int] = 15

# How far the best candidate must beat the runner-up to be accepted without
# asking the LLM. Two entities scoring within this of each other are usually
# a father and a son, or a man and the ship named after him.
AMBIGUITY_MARGIN: Final[float] = 6.0

# Below this, a candidate is not worth an LLM call: the deterministic steps
# found nothing that even resembles the mention, and asking a model to choose
# between a name and three unrelated names invites a confident wrong answer.
LLM_FLOOR_BELOW_THRESHOLD: Final[float] = 15.0

# Tiebreaks, added after the threshold test so they can order candidates but
# never promote one over the bar on their own.
_MILITARY_BONUS: Final[float] = 3.0
_POLITY_BONUS: Final[float] = 2.0

_EXACT_CONFIDENCE: Final[float] = 0.95
_NEW_ENTITY_CONFIDENCE: Final[float] = 0.6
_FUZZY_CONFIDENCE_FLOOR: Final[float] = 0.5
_FUZZY_CONFIDENCE_CEILING: Final[float] = 0.92
# A corpus-internal match rests on this corpus only, with no external
# authority behind it, so it is never as trusted as a Wikidata link.
_CORPUS_CONFIDENCE_FACTOR: Final[float] = 0.9

# The longest plausible span between two battles of one commander's career.
# Longer than any real career, deliberately: this gate exists to separate
# "Scipio" in 202 BC from "Scipio" in 1500 AD, not to police career length.
_CAREER_SPAN_YEARS: Final[int] = 60


def best_score(keys: Sequence[str], other_keys: Sequence[str]) -> float:
    """Score the best pairing between two sets of folded name keys.

    Args:
        keys: The mention group's keys.
        other_keys: A candidate's or identity's keys.

    Returns:
        The highest ``rapidfuzz`` WRatio over all pairs, 0 when either side
        is empty. WRatio rather than a plain ratio because the surface forms
        differ in length and word order far more than in spelling.
    """
    if not keys or not other_keys:
        return 0.0
    return max(float(fuzz.WRatio(key, other)) for key in keys for other in other_keys)


def command_window(candidate: Candidate) -> tuple[int, int] | None:
    """The astronomical years in which a candidate could have commanded.

    Args:
        candidate: The entity.

    Returns:
        ``(low, high)`` inclusive, or None when the candidate has no dated
        life. A contradictory life (earliest possible birth after the latest
        possible death) gives an empty window, ``(1, 0)``, which no year falls
        in: the candidate is rejected, not skipped. A stub with impossible
        dates is evidence of nothing, and linking it on the name alone risks a
        false merge. A real person rejected this way stays unlinked and
        visible, and the overrides file can relink him. A known birth opens the window
        at birth plus :data:`MIN_COMMAND_AGE_YEARS`. A half-dated life is
        bounded at the missing end rather than left open. Wikidata carries
        many ancients with a birth and no death -- often an explicit "no
        value" snak -- and an unbounded upper end made them immortal:
        Q1576150, a Carthaginian commander born about 300 BC with no recorded
        death, was judged alive at the Battle of Lissa in 1811, *positively*,
        so the exact-match rule preferred it over dateless namesakes and
        linked it at 0.95 confidence. That is a false merge, the one error
        nothing downstream can detect. The cap is far beyond any real career,
        like :data:`_CAREER_SPAN_YEARS`: this gate separates centuries, it
        does not police age.
    """
    birth = candidate.birth_year
    death = candidate.death_year
    if birth is None and death is None:
        return None

    # Wikidata stubs exist whose birth is after their death (born 700, died
    # 645). Abstaining would let the name alone link such a stub for a battle
    # in any year, a false merge nothing downstream detects, so the window is
    # empty and the candidate is rejected for every dated battle.
    if (
        birth is not None
        and death is not None
        and birth - candidate.birth_slack > death + candidate.death_slack
    ):
        if candidate.qid not in _CONTRADICTORY_LOGGED:
            _CONTRADICTORY_LOGGED.add(candidate.qid)
            logger.warning(
                "resolve_contradictory_lifespan",
                qid=candidate.qid,
                born=birth,
                died=death,
                action="candidate rejected for every dated battle",
            )
        return 1, 0

    # A decade- or century-precision date is read at its widest: the earliest
    # possible birth opens the window and the latest possible death closes it,
    # so a coarse date can never make a candidate look too young or already dead.
    if birth is not None:
        low = birth - candidate.birth_slack + MIN_COMMAND_AGE_YEARS
        if death is not None:
            high = death + candidate.death_slack
        else:
            high = birth + candidate.birth_slack + MAX_PLAUSIBLE_AGE_YEARS
    else:
        assert death is not None
        low = death - candidate.death_slack - MAX_PLAUSIBLE_AGE_YEARS
        high = death + candidate.death_slack
    return low - LIFESPAN_SLACK_YEARS, high + LIFESPAN_SLACK_YEARS


def lifespan_verdict(candidate: Candidate, years: Sequence[int]) -> bool | None:
    """Judge whether a candidate could have commanded in these battles.

    Args:
        candidate: The entity.
        years: Astronomical years of the group's battles.

    Returns:
        True when at least one battle falls inside the candidate's command
        window, False when every battle falls outside it, and None when the
        question cannot be asked -- no dated battle, or no dated life. None is
        not a pass: it means the gate abstains and the name has to carry the
        match alone.
    """
    if not years:
        return None
    window = command_window(candidate)
    if window is None:
        return None
    low, high = window
    return any(low <= year <= high for year in years)


def partition_by_lifespan(
    mentions: Sequence[Mention], candidate: Candidate
) -> tuple[list[Mention], list[Mention]]:
    """Separate the mentions a candidate could have commanded from the rest.

    A group is linked as a whole on the strength of one in-lifespan battle
    (:func:`lifespan_verdict`), which carried every namesake mention with it.
    This is the per-mention check that follows the link.

    Args:
        mentions: A linked group's mentions.
        candidate: The entity it was linked to.

    Returns:
        ``(kept, detached)``. Undated mentions are kept, because the gate
        abstains on them; so is everything when the candidate has no dated
        life.
    """
    window = command_window(candidate)
    if window is None:
        return list(mentions), []
    low, high = window
    kept: list[Mention] = []
    detached: list[Mention] = []
    for mention in mentions:
        year = mention.context.year if mention.context else None
        outside = year is not None and not low <= year <= high
        (detached if outside else kept).append(mention)
    return kept, detached


def career_clusters(mentions: Sequence[Mention]) -> list[list[Mention]]:
    """Split one name's mentions where the dated years leave a lifetime.

    Two battles further apart than :data:`_CAREER_SPAN_YEARS` are not one
    career (the same test :func:`_careers_overlap` applies between
    identities), so "Caesar" in 45 BC and in 1948 are two people. Splitting
    runs before candidate matching, because one in-lifespan battle otherwise
    keeps a modern namesake viable for the whole group.

    Args:
        mentions: One group's mentions.

    Returns:
        Clusters in chronological order, the mentions of each in their
        original order. Undated mentions join the largest cluster (the
        earliest on a tie). A single cluster means no split.
    """
    dated = sorted({m.context.year for m in mentions if m.context and m.context.year is not None})
    if len(dated) < 2:
        return [list(mentions)]

    bounds: list[tuple[int, int]] = []
    first = previous = dated[0]
    for year in dated[1:]:
        if year - previous > _CAREER_SPAN_YEARS:
            bounds.append((first, previous))
            first = year
        previous = year
    bounds.append((first, previous))
    if len(bounds) == 1:
        return [list(mentions)]

    clusters: list[list[Mention]] = [[] for _ in bounds]
    undated: list[Mention] = []
    for mention in mentions:
        when = mention.context.year if mention.context else None
        if when is None:
            undated.append(mention)
            continue
        index = next(i for i, (low, high) in enumerate(bounds) if low <= when <= high)
        clusters[index].append(mention)
    largest = max(range(len(clusters)), key=lambda i: (len(clusters[i]), -i))
    clusters[largest].extend(undated)
    return clusters


def viable_candidates(
    candidates: Sequence[Candidate],
    years: Sequence[int],
) -> list[Candidate]:
    """Drop candidates the battle dates rule out.

    Args:
        candidates: Everything the candidate source returned.
        years: Astronomical years of the group's battles.

    Returns:
        The candidates whose lifespan is compatible with, or silent about,
        the battles. An abstaining gate keeps the candidate.
    """
    return [c for c in candidates if lifespan_verdict(c, years) is not False]


def _tiebreak_bonus(candidate: Candidate, group: MentionGroup) -> float:
    """Score adjustments used only to order candidates against each other.

    Args:
        candidate: The entity.
        group: The mention group, for the polities it fought for.

    Returns:
        A small bonus for a military occupation and for a citizenship that
        matches a side's polity. Applied after the threshold test, so it can
        break a tie but cannot make a match.
    """
    bonus = _MILITARY_BONUS if candidate.is_military else 0.0

    country = fold(candidate.country)
    if country:
        polities = {fold(p) for p in group.polities}
        if any(polity and (country in polity or polity in country) for polity in polities):
            bonus += _POLITY_BONUS

    return bonus


def _fuzzy_confidence(score: float, threshold: float) -> float:
    """Map a fuzzy score onto a confidence for ``battle_commanders``.

    Args:
        score: The raw WRatio of the accepted match.
        threshold: The spec's ``fuzzy_threshold``.

    Returns:
        A confidence rising from the floor at the threshold to the ceiling at
        a perfect score. Never 1.0: a fuzzy match is never certain.
    """
    span = max(1.0, 100.0 - threshold)
    fraction = min(1.0, max(0.0, (score - threshold) / span))
    width = _FUZZY_CONFIDENCE_CEILING - _FUZZY_CONFIDENCE_FLOOR
    return round(_FUZZY_CONFIDENCE_FLOOR + width * fraction, 3)


def _ranked(
    group: MentionGroup,
    candidates: Sequence[Candidate],
) -> list[tuple[float, float, Candidate]]:
    """Score and order candidates for one group.

    Args:
        group: The mention group.
        candidates: The viable candidates.

    Returns:
        (raw score, ranked score, candidate) triples, best first. Ties break
        on Q-id so a re-run orders them identically.
    """
    scored: list[tuple[float, float, Candidate]] = []
    for candidate in candidates:
        raw = best_score(group.keys, candidate_keys(candidate))
        scored.append((raw, raw + _tiebreak_bonus(candidate, group), candidate))

    scored.sort(key=lambda row: (-row[1], -row[0], row[2].qid))
    return scored


def rank_for_prompt(group: MentionGroup, candidates: Sequence[Candidate]) -> list[Candidate]:
    """Order the date-viable candidates by plausibility, best first.

    The LLM prompt shows only the first few candidates, so the order decides
    who the model can choose. Source order is the endpoint's, effectively
    arbitrary, and cut Julius Caesar (11th of 11 for his own name) and
    Hannibal Barca (15th of 15) from their own prompts.

    Args:
        group: The mention group.
        candidates: Everything the candidate source returned.

    Returns:
        The candidates the date gate keeps, with name-exact before fuzzy,
        date-confirmed before date-silent, then the matcher's score with its
        military and polity tiebreaks, then the Wikimedia sitelink count as a
        notability signal, then Q-id so a re-run orders identically.
    """
    years = group.years
    group_keys = set(group.keys)

    def sort_key(row: tuple[float, float, Candidate]) -> tuple[int, int, float, float, int, str]:
        raw, ranked, candidate = row
        exact = bool(group_keys & set(candidate_keys(candidate)))
        confirmed = lifespan_verdict(candidate, years) is True
        return (
            0 if exact else 1,
            0 if confirmed else 1,
            -ranked,
            -raw,
            -candidate.sitelinks,
            candidate.qid,
        )

    return [
        c for _, _, c in sorted(_ranked(group, viable_candidates(candidates, years)), key=sort_key)
    ]


def _careers_overlap(left: Sequence[int], right: Sequence[int]) -> bool:
    """Whether two sets of battle years could belong to one career.

    Args:
        left: Astronomical years of one group's battles.
        right: Astronomical years of another's.

    Returns:
        True when the two year ranges lie within one career span of each
        other.
    """
    if not left or not right:
        return True
    return (
        min(left) <= max(right) + _CAREER_SPAN_YEARS
        and min(right) <= max(left) + _CAREER_SPAN_YEARS
    )


def match_group(
    group: MentionGroup,
    candidates: Sequence[Candidate],
    *,
    threshold: float = 85.0,
    margin: float = AMBIGUITY_MARGIN,
) -> Decision:
    """Resolve one mention group against Wikidata candidates.

    Args:
        group: The mention group.
        candidates: Everything the candidate source returned for it.
        threshold: The spec's ``fuzzy_threshold``, 0..100.
        margin: How far the best match must beat the runner-up.

    Returns:
        A decision. ``linked`` and ``new`` are final; ``ambiguous`` means the
        LLM step should look at it.
    """
    years = group.years
    viable = viable_candidates(candidates, years)
    rejected = len(candidates) - len(viable)

    if rejected:
        logger.debug(
            "candidates_rejected_on_lifespan",
            group=group.key,
            rejected=rejected,
            years=sorted(set(years))[:5],
        )

    if not viable:
        return Decision(
            status="new",
            canonical_name=group.display_name,
            method="new_entity",
            confidence=_NEW_ENTITY_CONFIDENCE,
            candidates_considered=0,
            reasoning=(
                f"no candidate survived the date gate ({rejected} rejected)"
                if rejected
                else "no candidate returned"
            ),
        )

    scored = _ranked(group, viable)
    top_raw, _, top = scored[0]
    runner_up = scored[1][2].label if len(scored) > 1 else None
    runner_up_raw = scored[1][0] if len(scored) > 1 else 0.0

    group_keys = set(group.keys)
    exact = [c for _, _, c in scored if group_keys & set(candidate_keys(c))]

    if len(exact) > 1:
        # The date gate abstains on a candidate with no dates, which is the
        # right call -- absence of a lifespan is not evidence against one --
        # but it means every dateless namesake survives. Wikidata has a great
        # many of those, and against live data they were dragging genuinely
        # unambiguous commanders into the LLM step: "Napoleon" in 1805 came
        # back as Q517 alongside a dateless NFT founder of the same name.
        #
        # So among exact matches, prefer the ones the dates positively
        # confirm. This concedes nothing on safety: a candidate the dates
        # contradict was already dropped, and if two are both confirmed alive
        # the group stays ambiguous and still goes to the model. It only
        # breaks the tie between evidence and no evidence.
        confirmed = [c for c in exact if lifespan_verdict(c, years) is True]
        if len(confirmed) == 1:
            logger.debug(
                "exact_match_resolved_on_dates",
                group=group.key,
                chosen=confirmed[0].qid,
                over=len(exact) - 1,
            )
            exact = confirmed

    if len(exact) == 1:
        candidate = exact[0]
        dated = lifespan_verdict(candidate, years) is True
        return Decision(
            status="linked",
            canonical_name=candidate.label,
            method="exact_wikidata",
            qid=candidate.qid,
            confidence=_EXACT_CONFIDENCE,
            score=100.0,
            runner_up=runner_up,
            candidates_considered=len(viable),
            candidate=candidate,
            reasoning=(
                "label or alias matched exactly, and the dates confirm it"
                if dated
                else "label or alias matched exactly"
            ),
        )

    if len(exact) > 1:
        return Decision(
            status="ambiguous",
            canonical_name=group.display_name,
            method="llm_disambiguation",
            score=top_raw,
            runner_up=runner_up,
            candidates_considered=len(viable),
            reasoning=(
                f"{len(exact)} candidates match the name exactly and the dates do not separate them"
            ),
        )

    if top_raw >= threshold:
        if len(scored) == 1 or top_raw - runner_up_raw >= margin:
            return Decision(
                status="linked",
                canonical_name=top.label,
                method="fuzzy_contextual",
                qid=top.qid,
                confidence=_fuzzy_confidence(top_raw, threshold),
                score=top_raw,
                runner_up=runner_up,
                candidates_considered=len(viable),
                candidate=top,
                reasoning=f"fuzzy match {top_raw:.1f} >= {threshold:.1f}, date-compatible",
            )
        return Decision(
            status="ambiguous",
            canonical_name=group.display_name,
            method="llm_disambiguation",
            score=top_raw,
            runner_up=runner_up,
            candidates_considered=len(viable),
            reasoning=f"top two within {top_raw - runner_up_raw:.1f} of each other",
        )

    if top_raw >= threshold - LLM_FLOOR_BELOW_THRESHOLD:
        return Decision(
            status="ambiguous",
            canonical_name=group.display_name,
            method="llm_disambiguation",
            score=top_raw,
            runner_up=runner_up,
            candidates_considered=len(viable),
            reasoning=f"best fuzzy match {top_raw:.1f} below threshold {threshold:.1f}",
        )

    return Decision(
        status="new",
        canonical_name=group.display_name,
        method="new_entity",
        confidence=_NEW_ENTITY_CONFIDENCE,
        score=top_raw,
        runner_up=runner_up,
        candidates_considered=len(viable),
        reasoning=f"best candidate scored {top_raw:.1f}, too far to be worth disambiguating",
    )


def match_against_identities(
    group: MentionGroup,
    identities: Sequence[Identity],
    *,
    threshold: float = 85.0,
    margin: float = AMBIGUITY_MARGIN,
) -> Decision | None:
    """Try to attach an unmatched group to an identity already resolved.

    This is the second half of ``agents/resolve.yaml`` step 2: a bare
    surname Wikidata did not answer for ("Bonaparte") often belongs to
    somebody the corpus has already resolved in full ("Napoleon Bonaparte").

    The era gate is different here, because a corpus identity has no birth or
    death dates to test -- only the years of the battles it appears in. Two
    identities whose battles never come within a career of each other are not
    the same commander.

    Args:
        group: The unmatched mention group.
        identities: Identities already resolved in this run.
        threshold: The spec's ``fuzzy_threshold``.
        margin: How far the best match must beat the runner-up.

    Returns:
        A decision when a confident corpus match exists, otherwise None,
        meaning the caller should keep the decision it already had.
    """
    years = group.years
    scored: list[tuple[float, Identity]] = []

    for identity in identities:
        folded = (fold(name) for name in (identity.canonical_name, *identity.aliases))
        score = best_score(group.keys, [key for key in folded if key])
        if score <= 0:
            continue
        if not _careers_overlap(years, identity.years):
            continue
        scored.append((score, identity))

    if not scored:
        return None

    scored.sort(key=lambda row: (-row[0], row[1].canonical_name))
    top_score, top = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else 0.0

    if top_score < threshold or (len(scored) > 1 and top_score - runner_up_score < margin):
        return None

    return Decision(
        status="linked" if top.qid else "new",
        canonical_name=top.canonical_name,
        method="corpus_fuzzy",
        qid=top.qid,
        confidence=round(_fuzzy_confidence(top_score, threshold) * _CORPUS_CONFIDENCE_FACTOR, 3),
        score=top_score,
        runner_up=scored[1][1].canonical_name if len(scored) > 1 else None,
        candidates_considered=len(scored),
        # Carried so the per-mention lifespan guard can judge the link; a
        # corpus identity with no Wikidata lifespan leaves it a no-op.
        candidate=top.candidate,
        reasoning=f"matched an identity already resolved in this corpus at {top_score:.1f}",
    )
