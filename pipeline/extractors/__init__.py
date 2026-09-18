"""
Extractors for the extract stage.

Four sources feed one battle record, and they disagree. The infobox is
structured and cheap; Wikidata is curated and authoritative on dates and
coordinates; the article body holds everything the infobox had no field for
and needs a model to read; fetched citations hold the same, less reliably.

Every extractor returns the same shape --
:class:`~pipeline.extractors.records.SourceExtraction` -- so the merger can
compare them without knowing which parser produced what. The merger prefers
structured sources for scalar fields and records the dissent, and keeps
every troop number as its own report, because the reconcile stage models
source disagreement and cannot model reports it never sees.

Typical use::

    from pipeline.extractors import merge_battle, parse_infobox

    infobox = parse_infobox(wikitext, source_ref=url, source_title=title)
    battle = merge_battle(slug, [s for s in (infobox, wikidata) if s])
"""

from __future__ import annotations

from pipeline.extractors.article import (
    DEFAULT_MAX_PASSAGE_CHARS,
    ArticleExtraction,
    Passage,
    build_passages,
    clean_article_text,
    extract_from_passages,
    iter_batches,
    payload_to_extraction,
    render_template,
)
from pipeline.extractors.infobox import (
    INFOBOX_STRENGTH_DEFAULT_SCOPE,
    campaign_names,
    detect_variant,
    parse_article_infobox,
    parse_infobox,
    parse_infobox_html,
)
from pipeline.extractors.merger import SOURCE_PRIORITY, merge_battle, missing_fields
from pipeline.extractors.records import (
    BattleExtraction,
    BattleFacts,
    CasualtyReport,
    CommanderMention,
    Disagreement,
    ExtractionFailure,
    MissingField,
    Provenance,
    SideExtraction,
    SourceExtraction,
    TroopReport,
    to_jsonable,
    to_schema_command_role,
)
from pipeline.extractors.store import ExtractWriteCounts, write_battle
from pipeline.extractors.textnorm import (
    Quantity,
    infer_branch,
    infer_scope,
    normalise_label,
    parse_quantities,
    resolve_scope,
    strip_markup,
)
from pipeline.extractors.wikidata_mapper import map_entity, wikidata_time_to_date

__all__ = [
    "DEFAULT_MAX_PASSAGE_CHARS",
    "INFOBOX_STRENGTH_DEFAULT_SCOPE",
    "SOURCE_PRIORITY",
    "ArticleExtraction",
    "BattleExtraction",
    "BattleFacts",
    "CasualtyReport",
    "CommanderMention",
    "Disagreement",
    "ExtractWriteCounts",
    "ExtractionFailure",
    "MissingField",
    "Passage",
    "Provenance",
    "Quantity",
    "SideExtraction",
    "SourceExtraction",
    "TroopReport",
    "build_passages",
    "campaign_names",
    "clean_article_text",
    "detect_variant",
    "extract_from_passages",
    "infer_branch",
    "infer_scope",
    "iter_batches",
    "map_entity",
    "merge_battle",
    "missing_fields",
    "normalise_label",
    "parse_article_infobox",
    "parse_infobox",
    "parse_infobox_html",
    "parse_quantities",
    "payload_to_extraction",
    "render_template",
    "resolve_scope",
    "strip_markup",
    "to_jsonable",
    "to_schema_command_role",
    "wikidata_time_to_date",
    "write_battle",
]
