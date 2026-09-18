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
    infer_casualty_type,
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
]

logger = structlog.get_logger()


# The military conflict infobox documents ``strength`` as the forces
# committed to the engagement, so a strength field whose wording says
# nothing else is read as ``engaged``. Wording that does say otherwise still
# wins: see textnorm.resolve_scope. Prose in the article body gets no such
# default, because a sentence carries no field semantics.
INFOBOX_STRENGTH_DEFAULT_SCOPE: Final[str] = "engaged"


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
    "land": InfoboxVariant(
        name="land",
        battle_type="field",
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
    "infobox military conflict": "land",
    "infobox military operation": "land",
    "infobox battle": "land",
    "infobox war": "land",
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

_RESULT_RE: Final = re.compile(
    r"(?P<qualifier>decisive|pyrrhic|marginal|tactical|strategic|minor|major|narrow)?\s*"
    r"(?P<who>[A-Za-zÀ-ɏ'’\- ]{2,60}?)\s+victory",
    re.IGNORECASE,
)
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

    match = _RESULT_RE.search(result_text)
    if match is None:
        return None, None

    qualifier = (match.group("qualifier") or "").lower()
    who = re.sub(r"\s+", " ", match.group("who")).strip(" -–—")

    if qualifier == "decisive":
        level = "decisive_victory"
    elif qualifier == "pyrrhic":
        level = "pyrrhic_victory"
    else:
        level = "victory"

    return (who or None), level


def _commander_mentions(
    value: str, side_label: str, provenance: Provenance
) -> list[CommanderMention]:
    """Split a commander field into one mention per named person.

    Args:
        value: Cleaned text of a ``commanderN`` field.
        side_label: The side these commanders served.
        provenance: Where the field came from.

    Returns:
        One mention per item. Roles stay ``unclear``: an infobox lists who
        was present, not who commanded tactically, and deciding that is the
        classify stage's job.
    """
    mentions: list[CommanderMention] = []

    for item in split_list_items(value):
        note_match = re.search(r"\((KIA|WIA|POW|MIA|executed|surrendered)\)", item, re.IGNORECASE)
        note = note_match.group(1) if note_match else ""
        name = re.sub(r"\((?:KIA|WIA|POW|MIA|executed|surrendered)\)", "", item, flags=re.I)
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
            )
        )

    return mentions


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
    """
    reports: list[TroopReport] = []

    for field_name, value in fields:
        stem = _SIDE_SUFFIX_RE.match(field_name)
        stem_name = stem.group("stem").strip() if stem else field_name
        if stem_name not in variant.strength_fields:
            continue
        implied_branch = variant.strength_fields[stem_name]

        for quantity in parse_quantities(value, default_scope=INFOBOX_STRENGTH_DEFAULT_SCOPE):
            reports.append(
                TroopReport(
                    side_label=side_label,
                    reported_value=quantity.value,
                    provenance=provenance,
                    branch=implied_branch or quantity.branch,
                    scope=quantity.scope,
                    is_estimate=quantity.is_estimate,
                    is_upper_bound=quantity.is_upper_bound,
                    is_lower_bound=quantity.is_lower_bound,
                    extracted_context=quantity.context,
                    page_or_section=f"infobox:{field_name}",
                )
            )

    return reports


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
    facts = BattleFacts(
        name=name or fallback_name or None,
        date_start=fields.get("date") or None,
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
            side.commanders.extend(_commander_mentions(value, label, provenance))
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

    return SourceExtraction(provenance=provenance, facts=facts, sides=sides, notes=notes)


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
