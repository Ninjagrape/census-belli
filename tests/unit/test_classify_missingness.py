"""
Unit tests for missingness classification.

The rule that must never break is the one about not overwriting a decision
another stage already made: resolve writes 'mnar' directly onto
``commander_general_id`` rows for a commander mention it could not resolve
(handover.md §12.3's sibling rule, for ``command_role``), and this module
must leave that alone even though 'commander_general_id' looks, at a
glance, like exactly the kind of field a missingness heuristic could have
an opinion about.
"""

from __future__ import annotations

from pipeline.classifiers.missingness import IMPUTE_FIELDS, MissingnessInput, classify_missingness


def _record(**overrides: object) -> MissingnessInput:
    defaults: dict[str, object] = {
        "log_id": 1,
        "field_name": "troop_total",
        "current_class": "unclassified",
        "year_astronomical": 500,
        "this_side_has_troops": False,
        "other_side_has_troops": False,
        "any_side_has_troops": False,
        "side_outcome": None,
    }
    defaults.update(overrides)
    return MissingnessInput(**defaults)  # type: ignore[arg-type]


def test_an_already_classified_row_is_never_touched() -> None:
    for existing_class in ("mnar", "mar", "mcar", "observed"):
        record = _record(current_class=existing_class, year_astronomical=100)
        assert classify_missingness(record) is None


def test_missingness_never_overwrites_resolves_mnar_on_commander_general_id() -> None:
    # commander_general_id is not in IMPUTE_FIELDS at all -- resolve owns it
    # end to end -- so this is refused before current_class is even looked at.
    record = _record(
        field_name="commander_general_id",
        current_class="mnar",
        year_astronomical=100,
    )
    assert "commander_general_id" not in IMPUTE_FIELDS
    assert classify_missingness(record) is None


def test_an_unclassified_commander_general_id_row_is_still_out_of_scope() -> None:
    # Even 'unclassified' commander_general_id rows are left alone: this
    # stage only ever writes to the five fields impute consumes.
    record = _record(field_name="commander_general_id", current_class="unclassified")
    assert classify_missingness(record) is None


def test_undated_battles_stay_unclassified() -> None:
    record = _record(year_astronomical=None, any_side_has_troops=False)
    assert classify_missingness(record) is None


def test_ancient_battle_with_no_troop_data_anywhere_is_mnar() -> None:
    record = _record(year_astronomical=-200, any_side_has_troops=False)

    decision = classify_missingness(record)

    assert decision is not None
    assert decision.missingness_class == "mnar"
    assert decision.log_id == record.log_id


def test_medieval_battle_at_the_1500_boundary_is_still_mnar() -> None:
    record = _record(year_astronomical=1499, any_side_has_troops=False)
    decision = classify_missingness(record)
    assert decision is not None
    assert decision.missingness_class == "mnar"


def test_year_1500_itself_is_outside_the_ancient_medieval_rule() -> None:
    # 1500 is the stated cutoff, exclusive: this alone shouldn't produce a
    # decision from the ancient/medieval heuristic (it may still match
    # nothing else and stay unclassified).
    record = _record(year_astronomical=1500, any_side_has_troops=False)
    decision = classify_missingness(record)
    assert decision is None


def test_only_the_winners_numbers_known_is_mnar() -> None:
    record = _record(
        year_astronomical=1600,
        side_outcome="loser",
        this_side_has_troops=False,
        other_side_has_troops=True,
        any_side_has_troops=True,
    )

    decision = classify_missingness(record)

    assert decision is not None
    assert decision.missingness_class == "mnar"


def test_1700_to_1914_missing_one_sides_numbers_is_mar() -> None:
    record = _record(
        year_astronomical=1815,
        side_outcome="winner",
        this_side_has_troops=False,
        other_side_has_troops=True,
        any_side_has_troops=True,
    )

    decision = classify_missingness(record)

    assert decision is not None
    assert decision.missingness_class == "mar"


def test_mar_era_boundaries_are_inclusive() -> None:
    for year in (1700, 1914):
        record = _record(
            year_astronomical=year,
            side_outcome="winner",
            this_side_has_troops=False,
            other_side_has_troops=True,
            any_side_has_troops=True,
        )
        decision = classify_missingness(record)
        assert decision is not None
        assert decision.missingness_class == "mar"


def test_losing_side_beats_the_era_rule_and_stays_mnar_in_the_mar_window() -> None:
    # Both heuristics could fire at once inside 1700-1914; "only the winner
    # is known" is checked first because it is the more specific claim.
    record = _record(
        year_astronomical=1800,
        side_outcome="loser",
        this_side_has_troops=False,
        other_side_has_troops=True,
        any_side_has_troops=True,
    )

    decision = classify_missingness(record)

    assert decision is not None
    assert decision.missingness_class == "mnar"


def test_a_battle_matching_no_heuristic_stays_unclassified() -> None:
    # 1950, both sides' numbers already known would not even generate a
    # missing_data_log row in practice, but a genuinely ambiguous modern
    # case with no matching heuristic should come back None rather than a
    # guess.
    record = _record(
        year_astronomical=1950,
        this_side_has_troops=False,
        other_side_has_troops=False,
        any_side_has_troops=False,
        side_outcome=None,
    )
    decision = classify_missingness(record)
    assert decision is None


def test_battle_type_terrain_and_fortified_are_mar_not_mcar() -> None:
    for field_name in ("battle_type", "terrain", "fortified"):
        record = _record(field_name=field_name, year_astronomical=1900)
        decision = classify_missingness(record)
        assert decision is not None
        assert decision.missingness_class == "mar"


def test_casualties_field_uses_the_same_troop_heuristics_as_troop_total() -> None:
    record = _record(field_name="casualties", year_astronomical=-500, any_side_has_troops=False)
    decision = classify_missingness(record)
    assert decision is not None
    assert decision.missingness_class == "mnar"


def test_impute_fields_matches_the_five_fields_impute_consumes() -> None:
    expected = {"troop_total", "casualties", "battle_type", "terrain", "fortified"}
    assert expected == IMPUTE_FIELDS
