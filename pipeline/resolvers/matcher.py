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
from pipeline.resolvers.records import Candidate, Decision, Identity, MentionGroup

__all__ = [
    "AMBIGUITY_MARGIN",
    "LIFESPAN_SLACK_YEARS",
    "LLM_FLOOR_BELOW_THRESHOLD",
    "MAX_PLAUSIBLE_AGE_YEARS",
    "best_score",
    "lifespan_verdict",
    "match_against_identities",
    "match_group",
    "viable_candidates",
]

logger = structlog.get_logger()

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


def lifespan_verdict(candidate: Candidate, years: Sequence[int]) -> bool | None:
    """Judge whether a candidate could have commanded in these battles.

    Args:
        candidate: The entity.
        years: Astronomical years of the group's battles.

    Returns:
        True when at least one battle falls inside the candidate's lifespan,
        False when every battle falls outside it, and None when the question
        cannot be asked -- no dated battle, or no dated life. None is not a
        pass: it means the gate abstains and the name has to carry the match
        alone.
    """
    if not years:
        return None
    if candidate.birth_year is None and candidate.death_year is None:
        return None

    # A half-known lifespan is bounded at the missing end rather than left
    # open. Wikidata carries many ancients with a birth and no death -- often
    # an explicit "no value" snak -- and an unbounded upper end made them
    # immortal: Q1576150, a Carthaginian commander born about 300 BC with no
    # recorded death, was judged alive at the Battle of Lissa in 1811, and
    # judged *positively*, so the exact-match rule below preferred it over
    # dateless namesakes and linked it at 0.95 confidence. That is a false
    # merge, which is the one error nothing downstream can detect.
    #
    # The cap is deliberately far beyond any real career, like
    # _CAREER_SPAN_YEARS: this gate separates centuries, it does not police age.
    low = candidate.birth_year
    high = candidate.death_year
    if low is None and high is not None:
        low = high - MAX_PLAUSIBLE_AGE_YEARS
    if high is None and low is not None:
        high = low + MAX_PLAUSIBLE_AGE_YEARS

    low = None if low is None else low - LIFESPAN_SLACK_YEARS
    high = None if high is None else high + LIFESPAN_SLACK_YEARS

    for year in years:
        if low is not None and year < low:
            continue
        if high is not None and year > high:
            continue
        return True
    return False


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
                f"{len(exact)} candidates match the name exactly and the dates "
                "do not separate them"
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
        reasoning=f"matched an identity already resolved in this corpus at {top_score:.1f}",
    )
