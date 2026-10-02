"""
Unit tests for building the command-role LLM request.

The property worth guarding here is the one downstream idempotency depends
on: two calls with the same inputs must build a request that hashes the
same, and changing the excerpt must change the hash. Get that wrong and
``scripts/llm_offline.py --stage classify`` will export requests the live
stage never matches, or the audit cache will never hit.
"""

from __future__ import annotations

import pytest

from pipeline.classifiers.records import CommanderRow, SideGroup
from pipeline.classifiers.requests import build_role_request, select_excerpt
from pipeline.llm.base import request_hash

SPEC: dict[str, object] = {
    "command_role_prompt": {
        "system": "You are a military history analyst classifying command relationships.",
        "user_template": (
            "Battle: {battle_name} ({date})\n"
            "Side: {side_label}\n\n"
            "Commanders listed:\n"
            "{commanders_formatted}\n\n"
            "Extracted role evidence:\n"
            "{role_evidence_formatted}\n\n"
            "Full article excerpt (relevant sections):\n"
            "{article_excerpt}\n\n"
            "Classify each commander. Return JSON:\n"
            "{{\n"
            '  "classifications": [\n'
            "    {{\n"
            '      "name": "...",\n'
            '      "command_role": "...",\n'
            '      "hierarchy_rank": 0,\n'
            '      "reports_to": null or "name of superior",\n'
            '      "attribution_weight": 0.0-1.0,\n'
            '      "confidence": 0.0-1.0,\n'
            '      "reasoning": "..."\n'
            "    }}\n"
            "  ],\n"
            '  "needs_review": true/false,\n'
            '  "review_reason": "..." or null\n'
            "}}\n"
        ),
    },
    "params": {"llm_max_tokens": 2048, "llm_temperature": 0.0},
}


def _side() -> SideGroup:
    return SideGroup(
        battle_id=1,
        side_id=2,
        side_label="Forces of Octavian",
        battle_name="Battle of Actium",
        year_astronomical=-30,
        commanders=(
            CommanderRow(
                bc_id=1,
                battle_id=1,
                side_id=2,
                general_id=11,
                name="Marcus Vipsanius Agrippa",
                apparent_role="unclear",
                role_evidence="Agrippa commanded the fleet",
                listing_order=0,
            ),
            CommanderRow(
                bc_id=2,
                battle_id=1,
                side_id=2,
                general_id=12,
                name="Octavian",
                apparent_role="unclear",
                role_evidence="",
                listing_order=1,
            ),
        ),
    )


def test_build_role_request_renders_the_literal_json_braces_correctly() -> None:
    request = build_role_request(_side(), "Agrippa commanded the fleet.", SPEC)

    assert '"classifications": [' in request.user
    assert "{{" not in request.user
    assert "}}" not in request.user


def test_build_role_request_substitutes_all_placeholders() -> None:
    request = build_role_request(_side(), "an excerpt about Agrippa", SPEC)

    assert "Battle of Actium" in request.user
    assert "31 BC" in request.user  # astronomical year -30 == 31 BC
    assert "Forces of Octavian" in request.user
    assert "Marcus Vipsanius Agrippa" in request.user
    assert "Agrippa commanded the fleet" in request.user
    assert "an excerpt about Agrippa" in request.user


def test_undated_battle_renders_date_as_unknown() -> None:
    side = SideGroup(
        battle_id=1,
        side_id=2,
        side_label="Side",
        battle_name="Battle of Somewhere",
        year_astronomical=None,
        commanders=_side().commanders,
    )
    request = build_role_request(side, "excerpt", SPEC)
    assert "(unknown)" in request.user


def test_ad_year_renders_without_a_bc_suffix() -> None:
    side = SideGroup(
        battle_id=1,
        side_id=2,
        side_label="Side",
        battle_name="Battle of Hastings",
        year_astronomical=1066,
        commanders=_side().commanders,
    )
    request = build_role_request(side, "excerpt", SPEC)
    assert "(1066)" in request.user


def test_missing_prompt_in_spec_raises() -> None:
    with pytest.raises(ValueError):
        build_role_request(_side(), "excerpt", {"command_role_prompt": {}})


def test_request_hash_is_stable_across_identical_builds() -> None:
    request_a = build_role_request(_side(), "the same excerpt", SPEC)
    request_b = build_role_request(_side(), "the same excerpt", SPEC)

    hash_a = request_hash(request_a, "anthropic", "claude-sonnet-4-6")
    hash_b = request_hash(request_b, "anthropic", "claude-sonnet-4-6")

    assert hash_a == hash_b


def test_request_hash_changes_when_the_excerpt_changes() -> None:
    request_a = build_role_request(_side(), "excerpt one", SPEC)
    request_b = build_role_request(_side(), "excerpt two", SPEC)

    hash_a = request_hash(request_a, "anthropic", "claude-sonnet-4-6")
    hash_b = request_hash(request_b, "anthropic", "claude-sonnet-4-6")

    assert hash_a != hash_b


def test_request_hash_is_unaffected_by_metadata() -> None:
    # metadata is local bookkeeping (battle_id/side_id) and is explicitly
    # excluded from the hash by pipeline.llm.base.request_hash; two
    # requests for different sides with the same prompt content should
    # still hash identically if that is ever wanted for a dedup check.
    request = build_role_request(_side(), "excerpt", SPEC)
    assert request.metadata == {"battle_id": 1, "side_id": 2}


# ─── select_excerpt ─────────────────────────────────────────────────────────


def test_select_excerpt_prefers_passages_mentioning_a_commander_by_surname() -> None:
    passages = [
        "The weather that day was clear and the sea calm.",
        "Agrippa commanded the fleet while Octavian remained aboard a transport.",
        "The campaign continued for several more months afterward.",
    ]
    excerpt = select_excerpt(passages, ["Marcus Vipsanius Agrippa"], max_chars=1000)

    assert "Agrippa commanded the fleet" in excerpt
    assert "weather that day" not in excerpt


def test_select_excerpt_matches_full_name_too() -> None:
    passages = ["Marcus Vipsanius Agrippa was born around 64 BC."]
    excerpt = select_excerpt(passages, ["Marcus Vipsanius Agrippa"], max_chars=1000)
    assert "born around 64 BC" in excerpt


def test_select_excerpt_keeps_passages_in_article_order() -> None:
    passages = [
        "First: Agrippa is mentioned here.",
        "Second: Octavian is mentioned here.",
    ]
    excerpt = select_excerpt(passages, ["Agrippa", "Octavian"], max_chars=1000)

    assert excerpt.index("First") < excerpt.index("Second")


def test_select_excerpt_caps_at_max_chars() -> None:
    passages = ["Agrippa " + "x" * 500, "Agrippa " + "y" * 500]
    excerpt = select_excerpt(passages, ["Agrippa"], max_chars=100)
    assert len(excerpt) <= 100


def test_select_excerpt_falls_back_to_the_top_of_the_article_when_no_name_matches() -> None:
    passages = ["Nobody named here.", "Still nobody named here."]
    excerpt = select_excerpt(passages, ["Agrippa"], max_chars=1000)
    assert excerpt  # never empty when passages exist
    assert "Nobody named here" in excerpt


def test_select_excerpt_of_no_passages_is_empty() -> None:
    assert select_excerpt([], ["Agrippa"], max_chars=1000) == ""
