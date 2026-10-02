"""
Command-role and missingness classification for the classify stage.

``agents/classify.yaml`` describes two unrelated jobs run over the same
already-resolved data: turning each side's commanders into a command
hierarchy (:mod:`pipeline.classifiers.roles`), and labelling why a missing
field is missing (:mod:`pipeline.classifiers.missingness`). A third module,
:mod:`pipeline.classifiers.battle_type`, does the inference handover.md §4.2
made necessary once the infobox parser stopped guessing 'field' for every
battle template it did not recognise.

Every module here is pure: dataclasses in, dataclasses out, no database and
no network. :mod:`pipeline.classifiers.requests` is the one module that
builds something a network call needs (an :class:`pipeline.llm.base.LLMRequest`),
but building one is still free of a database or a live call -- the stage
runner and ``scripts/llm_offline.py`` are what send it.

Typical use::

    from pipeline.classifiers import classify_side_roles, infer_battle_type

    decisions = classify_side_roles(side)
    if all(d.attribution_method == "default_split" for d in decisions):
        request = build_role_request(side, excerpt, spec)
"""

from __future__ import annotations

from pipeline.classifiers.battle_type import extract_categories, infer_battle_type
from pipeline.classifiers.load import (
    ARTICLE_SUFFIXES,
    HTML_SUFFIXES,
    find_article_path,
    find_unclassified_battle_type_log_ids,
    load_battle_wikipedia_urls,
    load_battles_needing_type,
    load_missingness_inputs,
    load_side_groups,
    load_troop_evidence,
    read_article_text,
)
from pipeline.classifiers.missingness import IMPUTE_FIELDS, MissingnessInput, classify_missingness
from pipeline.classifiers.records import (
    APPARENT_ROLES,
    BATTLE_TYPES,
    COMMAND_ROLES,
    MISSINGNESS_CLASSES,
    BattleTypeDecision,
    ClassifyCounts,
    CommanderRow,
    MissingnessDecision,
    RoleDecision,
    SideGroup,
    to_jsonable,
)
from pipeline.classifiers.requests import build_role_request, select_excerpt
from pipeline.classifiers.roles import (
    COMMAND_ROLE_RESPONSE_SCHEMA,
    apply_llm_classification,
    classify_side_roles,
    normalize_name,
)
from pipeline.classifiers.store import (
    write_battle_type_decisions,
    write_missingness_decisions,
    write_role_decisions,
)

__all__ = [
    "APPARENT_ROLES",
    "ARTICLE_SUFFIXES",
    "BATTLE_TYPES",
    "COMMAND_ROLE_RESPONSE_SCHEMA",
    "COMMAND_ROLES",
    "HTML_SUFFIXES",
    "IMPUTE_FIELDS",
    "MISSINGNESS_CLASSES",
    "BattleTypeDecision",
    "ClassifyCounts",
    "CommanderRow",
    "MissingnessDecision",
    "MissingnessInput",
    "RoleDecision",
    "SideGroup",
    "apply_llm_classification",
    "build_role_request",
    "classify_missingness",
    "classify_side_roles",
    "extract_categories",
    "find_article_path",
    "find_unclassified_battle_type_log_ids",
    "infer_battle_type",
    "load_battle_wikipedia_urls",
    "load_battles_needing_type",
    "load_missingness_inputs",
    "load_side_groups",
    "load_troop_evidence",
    "normalize_name",
    "read_article_text",
    "select_excerpt",
    "to_jsonable",
    "write_battle_type_decisions",
    "write_missingness_decisions",
    "write_role_decisions",
]
