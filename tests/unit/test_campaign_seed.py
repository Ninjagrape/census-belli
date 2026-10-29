"""Unit tests for the corpus3 seed writer's pure functions."""

from __future__ import annotations

from scripts.campaign_seed import (
    KEPT,
    NO_INFOBOX,
    NO_NAMED_COMMANDER,
    NO_SHARED_COMMANDER,
    normalise_target,
    parse_commanders,
    select_shared_commander,
)


def fs(*names: str) -> frozenset[str]:
    return frozenset(names)


class TestParseCommanders:
    def test_flagicon_and_refs_are_ignored(self) -> None:
        text = (
            "{{Infobox military conflict\n"
            "| conflict = Battle of X\n"
            "| commander1 = {{flagicon|France}} [[Napoleon]]<ref>[[Some Book]]</ref>\n"
            "| commander2 = {{flag|Austria}} [[Archduke Charles, Duke of Teschen|Charles]]\n"
            "}}\nBody with [[Not A Commander]]."
        )
        assert parse_commanders(text) == ["Archduke Charles, Duke of Teschen", "Napoleon"]

    def test_lettered_fields_and_lists(self) -> None:
        text = (
            "{{Infobox_military_conflict\n"
            "| commander1 = [[Lee]]\n"
            "| commander1a = [[Jackson_(general)#Early life|Jackson]]\n"
            "| commander2b = [[Grant]]\n"
            "| commander3 = {{plainlist|\n* [[Meade]]\n* [[File:Flag.svg]]\n}}\n"
            "| strength1 = [[Not Commander]]\n"
            "}}"
        )
        assert parse_commanders(text) == ["Grant", "Jackson (general)", "Lee", "Meade"]

    def test_piped_links_use_target_not_label(self) -> None:
        text = "{{Infobox battle|commander1=[[robert E. Lee|General Lee]]}}"
        assert parse_commanders(text) == ["Robert E. Lee"]

    def test_no_infobox_is_none_and_no_links_is_empty(self) -> None:
        assert parse_commanders("'''Saul''' was a king. {{Infobox royalty|name=Saul}}") is None
        assert parse_commanders("{{Infobox military conflict|commander1=Unknown}}") == []

    def test_only_first_infobox_is_read(self) -> None:
        text = (
            "{{Infobox military conflict|commander1=[[A]]}}"
            "{{Infobox military conflict|commander1=[[B]]}}"
        )
        assert parse_commanders(text) == ["A"]


def test_normalise_target() -> None:
    assert normalise_target("  george_Washington#Career ") == "George Washington"
    assert normalise_target("#Section") == ""


class TestSelectSharedCommander:
    def test_corpus2_anchored_candidate_is_kept_and_stranger_dropped(self) -> None:
        corpus2 = {"c2": fs("Napoleon")}
        candidates = {"a": fs("Napoleon", "Ney"), "b": fs("Obscure")}
        result = select_shared_commander(corpus2, candidates)
        assert result == {"a": KEPT, "b": NO_SHARED_COMMANDER}

    def test_no_commander_and_no_infobox_reasons(self) -> None:
        candidates = {"empty": fs(), "missing": None, "ok": fs("Napoleon")}
        result = select_shared_commander({"c2": fs("Napoleon")}, candidates)
        assert result == {"empty": NO_NAMED_COMMANDER, "missing": NO_INFOBOX, "ok": KEPT}

    def test_corpus2_battles_are_never_dropped(self) -> None:
        # Not part of the output: only candidates get a decision.
        assert select_shared_commander({"c2": fs("Loner")}, {}) == {}

    def test_chain_does_not_cascade_but_mutual_support_survives(self) -> None:
        # q hangs off p and r through different commanders; the stranger s is
        # dropped but nobody depended on it, so p, q, r all stay, anchored or not.
        candidates = {
            "p": fs("Y"),
            "q": fs("Y", "W"),
            "r": fs("W"),
            "s": fs("S"),
        }
        result = select_shared_commander({}, candidates)
        assert result == {"p": KEPT, "q": KEPT, "r": KEPT, "s": NO_SHARED_COMMANDER}

    def test_commander_counted_once_per_battle(self) -> None:
        # A single candidate cannot support itself.
        assert select_shared_commander({}, {"solo": fs("Z")}) == {"solo": NO_SHARED_COMMANDER}
