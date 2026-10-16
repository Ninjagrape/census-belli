"""
Deterministic parsing of military conflict infoboxes.

The infobox is the best-structured thing on a battle article and the
cheapest to read, so it is parsed first and its values outrank anything an
LLM later reports for the same field.

Wikipedia does not use one template for every engagement. Land battles use
``Infobox military conflict``, but naval, aerial and siege articles reach
for variants that name the same quantity differently: a naval infobox
reports ``ships1`` where a land one reports ``strength1``, and a siege
infobox reports ``garrison``. Those are mapped onto the schema's vocabulary
here, along with the branch each implies, so the merger never has to know
which template a number came from.

Both input forms are handled. The crawl stage stores rendered HTML, whose
infobox is a table of human-readable row labels; a wikitext source, when one
is available, is parsed with mwparserfromhell instead and yields more,
because the template names the variant and the campaignboxes are visible.
Both feed the same field mapping, so everything downstream sees one shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import mwparserfromhell
import structlog
from bs4 import BeautifulSoup

from pipeline.extractors.dates import parse_date_text
from pipeline.extractors.infobox_html import find_infobox_table, html_infobox_fields
from pipeline.extractors.records import (
    BattleFacts,
    CasualtyReport,
    CommanderMention,
    Provenance,
    SideExtraction,
    SourceExtraction,
    TroopReport,
)
from pipeline.extractors.textnorm import (
    Quantity,
    classify_figure,
    infer_casualty_type,
    is_equipment_count,
    is_labelled_figure,
    is_orphan_figure,
    parse_quantities,
    split_list_items,
    strip_markup,
)

__all__ = [
    "INFOBOX_STRENGTH_DEFAULT_SCOPE",
    "InfoboxVariant",
    "campaign_names",
    "detect_variant",
    "parse_article_infobox",
    "parse_infobox",
    "parse_infobox_html",
    "parse_result",
]

logger = structlog.get_logger()


# The military conflict infobox documents ``strength`` as the forces
# committed to the engagement, so a strength field whose wording says
# nothing else is read as ``engaged``. Wording that does say otherwise still
# wins: see textnorm.resolve_scope. Prose in the article body gets no such
# default, because a sentence carries no field semantics.
INFOBOX_STRENGTH_DEFAULT_SCOPE: Final[str] = "engaged"

# A component below this is far likelier a count of guns, ships or animals
# than of men, so it is kept as a report but never summed into a derived total.
_MIN_SUMMED_HEADCOUNT: Final[float] = 100.0


@dataclass(frozen=True)
class InfoboxVariant:
    """One infobox template family and how its fields map to the schema.

    Attributes:
        name: Short identifier, e.g. ``"naval"``.
        battle_type: The ``battle_type`` enum value the template implies, or
            None when the template does not determine it.
        strength_fields: Field stems that report force size, mapped to the
            ``troop_branch`` they imply. ``None`` means read the branch from
            the wording.
        casualty_fields: Field stems that report losses.
    """

    name: str
    battle_type: str | None
    strength_fields: dict[str, str | None]
    casualty_fields: tuple[str, ...]


_BASE_STRENGTH: Final[dict[str, str | None]] = {
    "strength": None,
    "forces": None,
    "troops": None,
    "units": None,
}

_BASE_CASUALTIES: Final[tuple[str, ...]] = ("casualties", "losses", "casualties and losses")

_VARIANTS: Final[dict[str, InfoboxVariant]] = {
    # "Infobox military conflict" and friends are Wikipedia's universal battle
    # templates: Actium, Trafalgar and Midway all use the plain one. The
    # template name therefore carries no evidence about the domain, so
    # battle_type is left for classify rather than guessed as "field", which
    # would silently label every naval and air battle a land engagement and
    # corrupt the battle_type covariate in agents/model.yaml for exactly the
    # commanders whose ranking depends on it.
    "generic": InfoboxVariant(
        name="generic",
        battle_type=None,
        strength_fields=dict(_BASE_STRENGTH),
        casualty_fields=_BASE_CASUALTIES,
    ),
    "naval": InfoboxVariant(
        name="naval",
        battle_type="naval",
        strength_fields={**_BASE_STRENGTH, "ships": "naval", "vessels": "naval"},
        casualty_fields=(*_BASE_CASUALTIES, "ships lost"),
    ),
    "aerial": InfoboxVariant(
        name="aerial",
        battle_type="aerial",
        strength_fields={**_BASE_STRENGTH, "aircraft": "air", "planes": "air"},
        casualty_fields=(*_BASE_CASUALTIES, "aircraft lost"),
    ),
    "siege": InfoboxVariant(
        # A siege template says a siege happened but not whose side the
        # article is written from, and the schema distinguishes
        # siege_offensive from siege_defensive. Leaving it unset logs a
        # missing field rather than guessing a direction.
        name="siege",
        battle_type=None,
        strength_fields={**_BASE_STRENGTH, "garrison": None, "besiegers": None},
        casualty_fields=_BASE_CASUALTIES,
    ),
    "rendered": InfoboxVariant(
        # Rendered HTML carries row labels, not a template name, so which
        # variant produced it is unknowable. The field names are read
        # generously and battle_type is left for a later stage.
        name="rendered",
        battle_type=None,
        strength_fields={**_BASE_STRENGTH, "ships": "naval", "aircraft": "air", "garrison": None},
        casualty_fields=(*_BASE_CASUALTIES, "ships lost", "aircraft lost"),
    ),
}

_TEMPLATE_VARIANTS: Final[dict[str, str]] = {
    "infobox military conflict": "generic",
    "infobox military operation": "generic",
    "infobox battle": "generic",
    "infobox war": "generic",
    "infobox naval conflict": "naval",
    "infobox naval battle": "naval",
    "infobox naval engagement": "naval",
    "infobox military naval conflict": "naval",
    "infobox air battle": "aerial",
    "infobox aerial engagement": "aerial",
    "infobox military aerial conflict": "aerial",
    "infobox siege": "siege",
    "infobox military siege": "siege",
}

_NAME_FIELDS: Final[tuple[str, ...]] = ("conflict", "battle_name", "name")
_COMBATANT_FIELDS: Final[tuple[str, ...]] = ("combatant", "belligerents", "side")
_COMMANDER_FIELDS: Final[tuple[str, ...]] = ("commander", "commanders", "leader")

# combatant1a / commander2b group sub-allies under a numbered side.
_SIDE_SUFFIX_RE: Final = re.compile(r"^(?P<stem>[a-z_ ]+?)(?P<index>[1-9])(?P<sub>[a-z])?$")

_QUALIFIER_WORDS: Final = "decisive|pyrrhic|marginal|tactical|strategic|minor|major|narrow"
_QUALIFIER_RE: Final = re.compile(rf"\b(?:{_QUALIFIER_WORDS})\b", re.IGNORECASE)
# The victor span may hold qualifiers on either side of the victor's name
# ("Roman pyrrhic victory", "Strategic Soviet victory"); they are split out after.
_RESULT_RE: Final = re.compile(
    r"(?P<who>[A-Za-zÀ-ɏ'’\- ]{2,60}?)\s+victory",
    re.IGNORECASE,
)
# "Victory for Octavian" puts the victor after the word rather than before it.
_VICTORY_FOR_RE: Final = re.compile(
    r"\bvictory\s+for\s+(?:the\s+)?(?P<who>[A-Za-zÀ-ɏ'’\- ]{2,60}?)(?=\s*(?:[,;.\n(]|$))",
    re.IGNORECASE,
)
# Cross-references an editor leaves in a result field: "(see § Aftermath)".
_RESULT_DEBRIS_RE: Final = re.compile(r"\((?:see|cf\.?)\b[^)]*\)", re.IGNORECASE)
_INDECISIVE_RE: Final = re.compile(
    r"\b(?:inconclusive|indecisive|stalemate|draw|status quo ante)\b", re.IGNORECASE
)

_TERRAIN_RE: Final = re.compile(
    r"\b(?:mountain(?:ous)?|hill[sy]?|forest(?:ed)?|wood(?:ed|land)|marsh(?:y|land)?|swamp"
    r"|desert|plain[s]?|river|ford|coastal|open ground|ridge|valley|pass|urban|steppe"
    r"|snow|ice|jungle)\b",
    re.IGNORECASE,
)
_FORTIFIED_RE: Final = re.compile(
    r"\b(?:fort(?:ress|ified|ification)?s?|citadel|redoubt|earthworks?|walls?|bastion"
    r"|entrenchments?|siege works?|stockade)\b",
    re.IGNORECASE,
)


def detect_variant(template_name: str) -> InfoboxVariant | None:
    """Identify which infobox family a template belongs to.

    Args:
        template_name: The raw template name, e.g. ``"Infobox military conflict"``.

    Returns:
        The matching variant, or None if the template is not a conflict infobox.
    """
    key = re.sub(r"\s+", " ", template_name.strip().lower())
    variant_name = _TEMPLATE_VARIANTS.get(key)
    if variant_name is None:
        return None
    return _VARIANTS[variant_name]


def campaign_names(wikitext: str) -> list[str]:
    """List the campaigns a battle's campaignboxes place it in.

    Args:
        wikitext: The full article source.

    Returns:
        Campaign or war names, in template order, without duplicates.
    """
    found: list[str] = []
    for template in mwparserfromhell.parse(wikitext).filter_templates(recursive=True):
        raw = re.sub(r"\s+", " ", str(template.name).strip())
        lowered = raw.lower()
        if not lowered.startswith(("campaignbox", "campaign box")):
            continue
        name = raw.split(" ", 1)[1].strip() if " " in raw else ""
        if not name and template.params:
            name = strip_markup(str(template.params[0].value)).strip()
        if name and name not in found:
            found.append(name)
    return found


# ─── Field collection ────────────────────────────────────────────────────────


def _template_fields(template: mwparserfromhell.nodes.Template) -> dict[str, str]:
    """Read every parameter of an infobox template as cleaned plain text.

    Args:
        template: The infobox template node.

    Returns:
        Normalised field name to cleaned value, blank fields omitted.
    """
    fields: dict[str, str] = {}
    for param in template.params:
        name = re.sub(r"\s+", " ", str(param.name).strip().lower())
        value = strip_markup(str(param.value))
        if name and value:
            fields[name] = value
    return fields


def _indexed_fields(
    fields: dict[str, str], stems: tuple[str, ...]
) -> dict[int, list[tuple[str, str]]]:
    """Collect side-indexed fields, folding sub-lettered variants into their side.

    ``combatant1a`` and ``combatant1b`` name separate allied contingents on
    side 1. They are kept as separate entries so their labels survive, but
    under the same side index, because the battle has two sides regardless of
    how many banners flew on each.

    Args:
        fields: Every infobox field, normalised.
        stems: Field stems to look for, e.g. ``("combatant", "side")``.

    Returns:
        Side index to a list of (field name, value) pairs.
    """
    collected: dict[int, list[tuple[str, str]]] = {}

    for name, value in fields.items():
        match = _SIDE_SUFFIX_RE.match(name)
        if match is None or match.group("stem").strip() not in stems:
            continue
        collected.setdefault(int(match.group("index")), []).append((name, value))

    for entries in collected.values():
        entries.sort(key=lambda pair: pair[0])

    return collected


# ─── Field interpretation ────────────────────────────────────────────────────


def _parse_result(result_text: str) -> tuple[str | None, str | None]:
    """Read a victor and an outcome level out of an infobox result field.

    Args:
        result_text: Cleaned text of the ``result`` field.

    Returns:
        A (victor, outcome_level) pair; either may be None when the field
        does not say.
    """
    if not result_text:
        return None, None

    if _INDECISIVE_RE.search(result_text):
        return None, "indecisive"

    result_text = _RESULT_DEBRIS_RE.sub("", result_text)
    match = _RESULT_RE.search(result_text)
    span = re.sub(r"\s+", " ", match.group("who")).strip(" -–—") if match else ""
    qualifiers = [q.lower() for q in _QUALIFIER_RE.findall(span)]
    who = re.sub(r"\s+", " ", _QUALIFIER_RE.sub(" ", span)).strip(" -–—")

    if not who:
        # "Decisive victory for Octavian": the qualifier filled the span and
        # the victor follows the word "victory".
        victory_for = _VICTORY_FOR_RE.search(result_text)
        if victory_for is None:
            return (None, None) if match is None else (None, _level(qualifiers))
        who = re.sub(r"\s+", " ", victory_for.group("who")).strip(" -–—")
        if match is None:
            return (who or None), "victory"

    return (who or None), _level(qualifiers)


def _level(qualifiers: list[str]) -> str:
    """Map victory qualifiers to an outcome level; decisive and pyrrhic win out."""
    if "decisive" in qualifiers:
        return "decisive_victory"
    if "pyrrhic" in qualifiers:
        return "pyrrhic_victory"
    return "victory"


parse_result = _parse_result


def _commander_mentions(
    value: str, side_label: str, provenance: Provenance, first_order: int = 0
) -> list[CommanderMention]:
    """Split a commander field into one mention per named person.

    Args:
        value: Cleaned text of a ``commanderN`` field.
        side_label: The side these commanders served.
        provenance: Where the field came from.
        first_order: ``infobox_order`` of the first mention, so a side with
            several commander fields keeps one running order.

    Returns:
        One mention per item. Roles stay ``unclear``: an infobox lists who
        was present, not who commanded tactically, and deciding that is the
        classify stage's job.
    """
    mentions: list[CommanderMention] = []

    for item in split_list_items(value):
        note_match = re.search(
            r"\((KIA|WIA|POW|MIA|DOW|executed|surrendered)\)", item, re.IGNORECASE
        )
        note = note_match.group(1) if note_match else ""
        name = re.sub(r"\((?:KIA|WIA|POW|MIA|DOW|executed|surrendered)\)", "", item, flags=re.I)
        name = name.strip(" \t†*-–—,")
        if not name or name.lower() in ("unknown", "various", "none"):
            continue
        mentions.append(
            CommanderMention(
                side_label=side_label,
                name=name,
                provenance=provenance,
                apparent_role="unclear",
                role_evidence=value[:500],
                note=note.upper() if note else "",
                infobox_order=first_order + len(mentions),
            )
        )

    return mentions


def _classified_branch(
    quantity: Quantity, implied_branch: str | None, *, first_line: bool
) -> tuple[str, bool]:
    """Pick the branch a headcount is stored under.

    A line naming an arm keeps that arm. A line naming a contingent,
    nationality or unit is a component, stored as ``other`` so reconcile does
    not read "25,000 Romans" as the Roman side's total. Only an explicit total
    or a lone unqualified figure stays ``total``.

    Args:
        quantity: The parsed figure.
        implied_branch: Branch the field itself implies (``ships1`` is naval).
        first_line: Whether the figure is on the first line of its field.

    Returns:
        The branch, and whether the figure counts equipment rather than men.
    """
    equipment = implied_branch is not None or is_equipment_count(quantity)
    if implied_branch:
        return implied_branch, equipment
    role = classify_figure(quantity, first_in_field=first_line)
    if role == "component":
        return "other", equipment
    return quantity.branch, equipment


def _derived_total(
    reports: list[TroopReport], unsafe: set[int], side_label: str, provenance: Provenance
) -> TroopReport | None:
    """Sum a side's components into one lower-bound total when it has none.

    Components may be incomplete, so the sum is a lower bound and is marked
    as derived in ``extracted_context``. Contingents and arms of service are
    separate partitions of the same side, so each is summed on its own and the
    larger wins; adding both would count the same men twice. Within the arms
    partition only the largest figure per arm is used, because "8,050 gunners
    and engineers" and a nested "7,232 gunners" describe overlapping men.

    Args:
        reports: The side's reports, none of which is a total.
        unsafe: Indexes into ``reports`` that must not be summed: equipment
            counts, figures behind a label that may name a phase or date, and
            numbers too small to be a headcount.
        side_label: The side the sum describes.
        provenance: Where the figures came from.

    Returns:
        The derived report, or None when fewer than two components exist.
    """
    contingents: list[TroopReport] = []
    arms: dict[str, TroopReport] = {}
    for index, report in enumerate(reports):
        if index in unsafe or report.is_upper_bound:
            continue
        if report.branch == "other":
            contingents.append(report)
        elif report.branch != "total":
            held = arms.get(report.branch)
            if held is None or report.reported_value > held.reported_value:
                arms[report.branch] = report

    partitions = [contingents, list(arms.values())]
    best = max(partitions, key=lambda part: sum(r.reported_value for r in part))
    if len(best) < 2:
        return None

    scopes = {r.scope for r in best}
    return TroopReport(
        side_label=side_label,
        reported_value=sum(r.reported_value for r in best),
        provenance=provenance,
        branch="total",
        scope=scopes.pop() if len(scopes) == 1 else "unknown",
        is_estimate=True,
        is_lower_bound=True,
        extracted_context=(
            f"derived: sum of {len(best)} component figures, a lower bound "
            "because components may be incomplete"
        ),
        page_or_section="infobox:derived_total",
    )


def _side_troop_reports(
    fields: list[tuple[str, str]],
    variant: InfoboxVariant,
    side_label: str,
    provenance: Provenance,
) -> list[TroopReport]:
    """Turn a side's strength fields into troop reports.

    Args:
        fields: (field name, value) pairs for this side.
        variant: The infobox family, which decides the branch some fields imply.
        side_label: The side these numbers describe.
        provenance: Where the fields came from.

    Returns:
        One report per number found. Ranges yield two, flagged as bounds.
        Component and arm figures are never stored as ``total``. When a side
        has no total but several components, one derived lower-bound total is
        appended; when it has no total and a single component, that figure is
        promoted to the total.
    """
    reports: list[TroopReport] = []
    equipment: set[int] = set()
    unsafe: set[int] = set()
    has_orphan = False

    for field_name, value in fields:
        stem = _SIDE_SUFFIX_RE.match(field_name)
        stem_name = stem.group("stem").strip() if stem else field_name
        if stem_name not in variant.strength_fields:
            continue
        implied_branch = variant.strength_fields[stem_name]

        quantities = parse_quantities(value, default_scope=INFOBOX_STRENGTH_DEFAULT_SCOPE)
        first_context = quantities[0].context if quantities else ""
        for quantity in quantities:
            branch, is_equipment = _classified_branch(
                quantity, implied_branch, first_line=quantity.context == first_context
            )
            if is_equipment:
                equipment.add(len(reports))
            # A label before the number may name a phase or a date ("Skirmish:
            # 20,000", "Battle: 40,000"), and a tiny count is more likely guns
            # or animals than men; neither is safe to add into a total.
            if (
                is_equipment
                or is_labelled_figure(quantity)
                or quantity.value < _MIN_SUMMED_HEADCOUNT
            ):
                unsafe.add(len(reports))
            has_orphan = has_orphan or is_orphan_figure(quantity)
            reports.append(
                TroopReport(
                    side_label=side_label,
                    reported_value=quantity.value,
                    provenance=provenance,
                    branch=branch,
                    scope=quantity.scope,
                    is_estimate=quantity.is_estimate,
                    is_upper_bound=quantity.is_upper_bound,
                    is_lower_bound=quantity.is_lower_bound,
                    extracted_context=quantity.context,
                    page_or_section=f"infobox:{field_name}",
                )
            )

    if reports and not any(r.branch == "total" for r in reports):
        _complete_total(
            reports, equipment, unsafe, side_label, provenance, may_derive=not has_orphan
        )

    return reports


def _complete_total(
    reports: list[TroopReport],
    equipment: set[int],
    unsafe: set[int],
    side_label: str,
    provenance: Provenance,
    *,
    may_derive: bool,
) -> None:
    """Give a side without a total either a promoted lone figure or a derived sum.

    Args:
        reports: The side's reports, modified in place.
        equipment: Indexes of reports that count equipment, not men.
        unsafe: Indexes of reports that must not be summed.
        side_label: The side the reports describe.
        provenance: Where the figures came from.
        may_derive: False when the field holds bare figures under headings
            the page dropped; their structure is unknown, so no sum is made.
    """
    components = [r for i, r in enumerate(reports) if r.branch == "other" and i not in equipment]
    arms = [
        r
        for i, r in enumerate(reports)
        if r.branch not in ("other", "total") and i not in equipment
    ]
    lines = {(r.page_or_section, r.extracted_context) for r in components}
    # "50,000 French troops" on the French side is that side's own figure
    # even though a nationality was named; with nothing else to compare it to,
    # it is the unqualified figure for the side.
    if len(lines) == 1 and not arms:
        for report in components:
            report.branch = "total"
        return

    if not may_derive:
        return
    derived = _derived_total(reports, unsafe, side_label, provenance)
    if derived is not None:
        reports.append(derived)


def _side_casualty_reports(
    fields: list[tuple[str, str]],
    variant: InfoboxVariant,
    side_label: str,
    provenance: Provenance,
) -> list[CasualtyReport]:
    """Turn a side's casualty fields into casualty reports.

    Args:
        fields: (field name, value) pairs for this side.
        variant: The infobox family.
        side_label: The side these losses describe.
        provenance: Where the fields came from.

    Returns:
        One report per number found.
    """
    reports: list[CasualtyReport] = []

    for field_name, value in fields:
        stem = _SIDE_SUFFIX_RE.match(field_name)
        stem_name = stem.group("stem").strip() if stem else field_name
        if stem_name not in variant.casualty_fields:
            continue

        for quantity in parse_quantities(value):
            reports.append(
                CasualtyReport(
                    side_label=side_label,
                    reported_value=quantity.value,
                    provenance=provenance,
                    casualty_type=infer_casualty_type(quantity.context),
                    is_estimate=quantity.is_estimate,
                    extracted_context=quantity.context,
                )
            )

    return reports


def _dedupe_part_of(names: list[str]) -> list[str]:
    """Drop duplicate war and campaign names differing only by a leading article.

    Args:
        names: Names from the ``partof`` field and the campaignboxes.

    Returns:
        The names in order, without duplicates.
    """
    kept: list[str] = []
    seen: set[str] = set()

    for name in names:
        cleaned = re.sub(r"^the\s+", "", name.strip(), flags=re.IGNORECASE)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        kept.append(cleaned)

    return kept


def _build_extraction(
    fields: dict[str, str],
    variant: InfoboxVariant,
    provenance: Provenance,
    *,
    campaigns: list[str],
    fallback_name: str,
) -> SourceExtraction:
    """Assemble a source extraction from collected infobox fields.

    Args:
        fields: Normalised field name to value.
        variant: The infobox family.
        provenance: Where the fields came from.
        campaigns: Campaign names from campaignboxes, if any were readable.
        fallback_name: Battle name to use when the infobox names none.

    Returns:
        What the infobox says about the battle.
    """
    place = fields.get("place", "")
    result_text = fields.get("result", "")
    victor, outcome_level = _parse_result(result_text)

    name = next((fields[key] for key in _NAME_FIELDS if fields.get(key)), "")
    part_of = fields.get("partof", "")

    terrain_source = f"{place} {fields.get('territory', '')}"
    date_text = fields.get("date") or None
    parsed_date = parse_date_text(date_text)
    if date_text and parsed_date is None:
        logger.warning("infobox_date_unparseable", date=date_text[:80], battle=fallback_name)
    facts = BattleFacts(
        name=name or fallback_name or None,
        date_start=parsed_date.date_start if parsed_date else None,
        date_end=parsed_date.date_end if parsed_date else None,
        date_precision=parsed_date.precision if parsed_date else None,
        date_circa=parsed_date.circa if parsed_date else None,
        location_name=place or None,
        battle_type=variant.battle_type,
        terrain=sorted({m.group(0).lower() for m in _TERRAIN_RE.finditer(terrain_source)}),
        fortified=True if _FORTIFIED_RE.search(f"{place} {result_text}") else None,
        victor=victor,
        outcome_level=outcome_level,
        outcome_evidence=result_text or None,
        part_of=_dedupe_part_of(([part_of] if part_of else []) + campaigns),
    )

    combatants = _indexed_fields(fields, _COMBATANT_FIELDS)
    commanders = _indexed_fields(fields, _COMMANDER_FIELDS)
    strengths = _indexed_fields(fields, tuple(variant.strength_fields))
    casualties = _indexed_fields(fields, variant.casualty_fields)

    sides: list[SideExtraction] = []
    for index in sorted(set(combatants) | set(commanders) | set(strengths)):
        # A combatant field often holds a coalition as a bulleted list. The
        # first entry names the side; the rest are recorded as aliases so the
        # resolve stage can still match a source that names an ally instead.
        labels: list[str] = []
        for _, value in combatants.get(index, []):
            labels.extend(split_list_items(value))
        label = labels[0] if labels else f"side {index}"

        side = SideExtraction(
            label=label,
            aliases=labels[1:],
            polity=label if labels else None,
        )
        for _, value in commanders.get(index, []):
            side.commanders.extend(
                _commander_mentions(value, label, provenance, len(side.commanders))
            )
        side.troop_reports = _side_troop_reports(
            strengths.get(index, []), variant, label, provenance
        )
        side.casualty_reports = _side_casualty_reports(
            casualties.get(index, []), variant, label, provenance
        )
        sides.append(side)

    notes: list[str] = []
    if variant.battle_type is None:
        notes.append(
            f"{variant.name} infobox does not determine battle_type; left unset for classify"
        )

    logger.debug(
        "infobox_parsed",
        variant=variant.name,
        sides=len(sides),
        troop_reports=sum(len(s.troop_reports) for s in sides),
        source=provenance.source_ref or provenance.source_title,
    )

    return SourceExtraction(
        provenance=provenance,
        facts=facts,
        sides=sides,
        notes=notes,
        unparsed_date_text=date_text if date_text and parsed_date is None else None,
    )


# ─── Entry points ────────────────────────────────────────────────────────────


def parse_infobox(
    wikitext: str,
    *,
    source_ref: str = "",
    source_title: str = "",
) -> SourceExtraction | None:
    """Parse a battle article's conflict infobox from MediaWiki source.

    Args:
        wikitext: The full MediaWiki source of a battle article.
        source_ref: URL or file path identifying the article.
        source_title: Human-readable article title.

    Returns:
        What the infobox says, or None when the article has no conflict
        infobox. A None return is a fact about the article, not an error:
        the caller logs it as a missing source and carries on with the body
        and Wikidata passes.
    """
    found: tuple[mwparserfromhell.nodes.Template, InfoboxVariant] | None = None
    for template in mwparserfromhell.parse(wikitext).filter_templates(recursive=True):
        variant = detect_variant(str(template.name))
        if variant is not None:
            found = (template, variant)
            break

    if found is None:
        logger.info("no_conflict_infobox_found", source=source_ref or source_title)
        return None

    template, variant = found
    return _build_extraction(
        _template_fields(template),
        variant,
        Provenance(
            source_type="wikipedia_infobox",
            extraction_method="infobox_parser",
            source_ref=source_ref,
            source_title=source_title,
        ),
        campaigns=campaign_names(wikitext),
        fallback_name=source_title,
    )


def parse_infobox_html(
    html: str,
    *,
    source_ref: str = "",
    source_title: str = "",
) -> SourceExtraction | None:
    """Parse a battle article's conflict infobox from rendered HTML.

    This is what the crawl stage actually stores. A rendered infobox gives
    the same quantities as the template but names them with human labels and
    does not say which template variant produced it, so ``battle_type`` is
    left for a later stage rather than guessed.

    Args:
        html: The article's rendered HTML.
        source_ref: URL or file path identifying the article.
        source_title: Human-readable article title.

    Returns:
        What the infobox says, or None when the page has no infobox table.
    """
    table = find_infobox_table(BeautifulSoup(html, "html.parser"))
    if table is None:
        logger.info("no_rendered_infobox_found", source=source_ref or source_title)
        return None

    fields = html_infobox_fields(table)

    if not fields:
        logger.info("rendered_infobox_had_no_readable_rows", source=source_ref)
        return None

    return _build_extraction(
        fields,
        _VARIANTS["rendered"],
        Provenance(
            source_type="wikipedia_infobox",
            extraction_method="infobox_parser",
            source_ref=source_ref,
            source_title=source_title,
        ),
        campaigns=[],
        fallback_name=source_title,
    )


def parse_article_infobox(
    raw: str,
    *,
    is_html: bool,
    source_ref: str = "",
    source_title: str = "",
) -> SourceExtraction | None:
    """Parse an article's infobox from whichever form the crawl stored.

    Args:
        raw: The article source.
        is_html: True for rendered HTML, False for MediaWiki source.
        source_ref: URL or file path identifying the article.
        source_title: Human-readable article title.

    Returns:
        What the infobox says, or None when the article has none.
    """
    if is_html:
        return parse_infobox_html(raw, source_ref=source_ref, source_title=source_title)
    return parse_infobox(raw, source_ref=source_ref, source_title=source_title)
