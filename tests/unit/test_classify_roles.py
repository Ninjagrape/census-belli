"""
Unit tests for command-role classification.

Three deterministic paths (one commander, consistent existing roles,
fallback split) and one LLM-response mapping path. The properties that
matter across all of them: every side's weights sum to 1.0, a
``reports_to`` link never crosses sides, and an LLM answer that does not
fit the schema degrades to 'unknown' plus review rather than being trusted.
"""

from __future__ import annotations

import math

import pytest

from pipeline.classifiers.records import CommanderRow, RoleDecision, SideGroup
from pipeline.classifiers.roles import (
    apply_llm_classification,
    classify_side_roles,
    normalize_name,
)


def _side(*commanders: CommanderRow, side_id: int = 1) -> SideGroup:
    return SideGroup(
        battle_id=1,
        side_id=side_id,
        side_label="Test Side",
        battle_name="Battle of Test",
        year_astronomical=100,
        commanders=tuple(commanders),
    )


def _commander(
    bc_id: int,
    name: str,
    *,
    command_role: str = "unknown",
    apparent_role: str = "unclear",
    listing_order: int = 0,
    role_evidence: str = "",
) -> CommanderRow:
    return CommanderRow(
        bc_id=bc_id,
        battle_id=1,
        side_id=1,
        general_id=bc_id * 10,
        name=name,
        command_role=command_role,
        apparent_role=apparent_role,
        listing_order=listing_order,
        role_evidence=role_evidence,
    )


def _weights_sum(decisions: list[RoleDecision]) -> float:
    return sum(d.attribution_weight for d in decisions)


# ─── Single commander ──────────────────────────────────────────────────────


def test_a_single_commander_is_the_field_commander_with_full_weight() -> None:
    side = _side(_commander(1, "Agrippa"))

    decisions = classify_side_roles(side)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.command_role == "field_commander"
    assert decision.hierarchy_rank == 0
    assert decision.attribution_weight == 1.0
    assert decision.attribution_method == "rule_single"
    assert decision.needs_review is False
    assert decision.reports_to_bc_id is None


def test_an_empty_side_yields_no_decisions() -> None:
    side = _side()
    assert classify_side_roles(side) == []


# ─── Consistent existing roles ─────────────────────────────────────────────


def test_consistent_roles_are_kept_with_rank_and_reports_to_derived() -> None:
    side = _side(
        _commander(1, "Octavian", command_role="sovereign", listing_order=0),
        _commander(2, "Agrippa", command_role="field_commander", listing_order=1),
    )

    decisions = classify_side_roles(side)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].command_role == "sovereign"
    assert by_id[1].hierarchy_rank == 1
    assert by_id[1].reports_to_bc_id is None  # only subordinates get reports_to
    assert by_id[2].command_role == "field_commander"
    assert by_id[2].hierarchy_rank == 0
    assert by_id[2].attribution_method == "rule_roles"
    assert by_id[2].needs_review is False


def test_subordinates_report_to_the_field_commander() -> None:
    side = _side(
        _commander(1, "Field Cdr", command_role="field_commander"),
        _commander(2, "Wing Cdr A", command_role="subordinate"),
        _commander(3, "Wing Cdr B", command_role="subordinate"),
    )

    decisions = classify_side_roles(side)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[2].reports_to_bc_id == 1
    assert by_id[3].reports_to_bc_id == 1
    assert by_id[1].reports_to_bc_id is None


def test_sovereign_gets_less_weight_than_the_field_commander() -> None:
    side = _side(
        _commander(1, "Octavian", command_role="sovereign"),
        _commander(2, "Agrippa", command_role="field_commander"),
    )

    decisions = classify_side_roles(side)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].attribution_weight < by_id[2].attribution_weight
    assert 0.05 <= by_id[1].attribution_weight <= 0.2


def test_two_field_commanders_is_not_consistent_and_falls_back() -> None:
    side = _side(
        _commander(1, "Consul A", command_role="field_commander"),
        _commander(2, "Consul B", command_role="field_commander"),
    )

    decisions = classify_side_roles(side)

    assert all(d.attribution_method == "default_split" for d in decisions)
    assert all(d.command_role == "unknown" for d in decisions)


def test_no_field_commander_at_all_is_not_consistent_and_falls_back() -> None:
    side = _side(
        _commander(1, "A", command_role="subordinate"),
        _commander(2, "B", command_role="subordinate"),
    )

    decisions = classify_side_roles(side)

    assert all(d.attribution_method == "default_split" for d in decisions)


def test_an_unknown_role_among_others_breaks_consistency() -> None:
    side = _side(
        _commander(1, "A", command_role="field_commander"),
        _commander(2, "B", command_role="unknown"),
    )

    decisions = classify_side_roles(side)

    assert all(d.attribution_method == "default_split" for d in decisions)


# ─── Fallback split ─────────────────────────────────────────────────────────


def test_two_commanders_with_no_signal_split_six_four_by_listing_order() -> None:
    side = _side(
        _commander(1, "Senior", listing_order=0),
        _commander(2, "Junior", listing_order=1),
    )

    decisions = classify_side_roles(side)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].attribution_weight == pytest.approx(0.6)
    assert by_id[2].attribution_weight == pytest.approx(0.4)
    assert all(d.command_role == "unknown" for d in decisions)
    assert all(d.needs_review for d in decisions)
    assert all(d.attribution_method == "default_split" for d in decisions)


def test_the_six_four_split_follows_listing_order_not_bc_id_order() -> None:
    # bc_id 2 is listed first (listing_order=0); the split should follow
    # listing order, not the bc_id numbering, since listing order is the
    # documented seniority proxy.
    side = _side(
        _commander(2, "Listed First", listing_order=0),
        _commander(1, "Listed Second", listing_order=1),
    )

    decisions = classify_side_roles(side)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[2].attribution_weight == pytest.approx(0.6)
    assert by_id[1].attribution_weight == pytest.approx(0.4)


def test_three_or_more_commanders_with_no_signal_split_equally() -> None:
    side = _side(
        _commander(1, "A", listing_order=0),
        _commander(2, "B", listing_order=1),
        _commander(3, "C", listing_order=2),
    )

    decisions = classify_side_roles(side)

    for d in decisions:
        assert d.attribution_weight == pytest.approx(1 / 3, abs=1e-6)


@pytest.mark.parametrize("n", range(1, 9))
def test_weights_always_sum_to_one_across_side_sizes(n: int) -> None:
    side = _side(*(_commander(i, f"Commander {i}", listing_order=i) for i in range(n)))
    decisions = classify_side_roles(side)
    assert _weights_sum(decisions) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "roles",
    [
        ("field_commander", "subordinate"),
        ("field_commander", "subordinate", "subordinate"),
        ("sovereign", "field_commander"),
        ("field_commander", "nominal", "subordinate", "subordinate"),
        ("theatre_commander", "subordinate"),
        ("supreme_commander", "sovereign", "field_commander"),
    ],
)
def test_weights_always_sum_to_one_for_consistent_role_combinations(
    roles: tuple[str, ...],
) -> None:
    side = _side(
        *(
            _commander(i, f"Commander {i}", command_role=role, listing_order=i)
            for i, role in enumerate(roles)
        )
    )
    decisions = classify_side_roles(side)
    assert _weights_sum(decisions) == pytest.approx(1.0, abs=1e-9)


# ─── normalize_name ─────────────────────────────────────────────────────────


def test_normalize_name_casefolds_and_collapses_whitespace() -> None:
    assert normalize_name("  Marcus   Vipsanius Agrippa ") == normalize_name(
        "marcus vipsanius agrippa"
    )
    assert normalize_name("AGRIPPA") == normalize_name("agrippa")


# ─── apply_llm_classification ──────────────────────────────────────────────


def _llm_side() -> SideGroup:
    return _side(
        _commander(1, "Octavian", listing_order=0),
        _commander(2, "Agrippa", listing_order=1),
    )


def test_llm_classification_maps_names_back_to_bc_ids() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {
                "name": "Octavian",
                "command_role": "sovereign",
                "hierarchy_rank": 1,
                "reports_to": None,
                "attribution_weight": 0.15,
                "confidence": 0.8,
                "reasoning": "present but did not command tactically",
            },
            {
                "name": "Agrippa",
                "command_role": "field_commander",
                "hierarchy_rank": 0,
                "reports_to": None,
                "attribution_weight": 0.85,
                "confidence": 0.9,
                "reasoning": "commanded the fleet",
            },
        ],
        "needs_review": False,
        "review_reason": None,
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].command_role == "sovereign"
    assert by_id[2].command_role == "field_commander"
    assert by_id[2].attribution_method == "llm"
    assert _weights_sum(decisions) == pytest.approx(1.0, abs=1e-9)


def test_llm_classification_is_case_and_whitespace_insensitive_on_names() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "  octavian  ", "command_role": "sovereign", "hierarchy_rank": 1,
             "attribution_weight": 0.1},
            {"name": "AGRIPPA", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 0.9},
        ],
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].command_role == "sovereign"
    assert by_id[2].command_role == "field_commander"


def test_reports_to_resolves_within_the_same_side_only() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": 0,
             "attribution_weight": 0.3, "reports_to": None},
            {"name": "Agrippa", "command_role": "subordinate", "hierarchy_rank": 1,
             "attribution_weight": 0.7, "reports_to": "Octavian"},
        ],
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[2].reports_to_bc_id == 1


def test_reports_to_a_name_not_on_this_side_resolves_to_none() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": 0,
             "attribution_weight": 0.3, "reports_to": "Mark Antony"},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 0.7, "reports_to": None},
        ],
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    # "Mark Antony" is not a commander on this side (he would be on the
    # opposing side's SideGroup, a different object entirely), so the
    # reference cannot be resolved and must not be guessed at.
    assert by_id[1].reports_to_bc_id is None


def test_a_self_reported_reports_to_resolves_to_none() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": 0,
             "attribution_weight": 0.5, "reports_to": "Octavian"},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 0.5, "reports_to": None},
        ],
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].reports_to_bc_id is None


def test_an_unrecognised_role_becomes_unknown_and_flags_review() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "emperor-for-life", "hierarchy_rank": 0,
             "attribution_weight": 0.5},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 0.5},
        ],
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].command_role == "unknown"
    assert by_id[1].needs_review is True


def test_missing_commander_name_falls_back_to_deterministic_and_flags_review() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 1.0},
        ],
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].command_role == "unknown"
    assert by_id[1].needs_review is True
    assert by_id[1].attribution_method == "default_split"
    assert by_id[2].command_role == "field_commander"
    # The fallback row is still included in the side's renormalised weights.
    assert _weights_sum(decisions) == pytest.approx(1.0, abs=1e-9)


def test_an_extra_name_not_on_the_side_is_ignored() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": 0,
             "attribution_weight": 0.3},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 0.7},
            {"name": "Mark Antony", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 1.0},
        ],
    }

    decisions = apply_llm_classification(side, data)

    assert {d.bc_id for d in decisions} == {1, 2}
    assert _weights_sum(decisions) == pytest.approx(1.0, abs=1e-9)


def test_weights_are_clamped_before_renormalising() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": 0,
             "attribution_weight": -5.0},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 50.0},
        ],
    }

    decisions = apply_llm_classification(side, data)
    by_id = {d.bc_id: d for d in decisions}

    assert by_id[1].attribution_weight == pytest.approx(0.0)
    assert by_id[2].attribution_weight == pytest.approx(1.0)


def test_all_zero_weights_renormalise_to_an_equal_split() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": 0,
             "attribution_weight": 0.0},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 0.0},
        ],
    }

    decisions = apply_llm_classification(side, data)

    for d in decisions:
        assert d.attribution_weight == pytest.approx(0.5)


def test_overall_needs_review_flag_propagates_to_every_matched_decision() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": 0,
             "attribution_weight": 0.3},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 0.7},
        ],
        "needs_review": True,
        "review_reason": "ambiguous article text",
    }

    decisions = apply_llm_classification(side, data)

    assert all(d.needs_review for d in decisions)


def test_malformed_response_fields_do_not_raise() -> None:
    side = _llm_side()
    data = {
        "classifications": [
            {"name": "Octavian", "command_role": "sovereign", "hierarchy_rank": "not-a-number",
             "attribution_weight": "also-not-a-number", "confidence": "nope"},
            {"name": "Agrippa", "command_role": "field_commander", "hierarchy_rank": 0,
             "attribution_weight": 1.0},
        ],
    }

    decisions = apply_llm_classification(side, data)

    assert len(decisions) == 2
    assert not math.isnan(_weights_sum(decisions))
    assert _weights_sum(decisions) == pytest.approx(1.0, abs=1e-9)


def test_empty_classifications_list_falls_back_for_every_commander() -> None:
    side = _llm_side()
    decisions = apply_llm_classification(side, {"classifications": []})

    assert len(decisions) == 2
    assert all(d.attribution_method == "default_split" for d in decisions)
    assert all(d.needs_review for d in decisions)
    assert _weights_sum(decisions) == pytest.approx(1.0, abs=1e-9)
