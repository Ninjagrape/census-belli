"""
Battle-type regressions built from the real pilot corpus.

The pilot typed Marathon, Malvern Hill, the Yom Kippur War and Operation
Torch as naval and 133 of 155 battles as unknown, because the stored URL never
matched the crawled article's filename (so no article was ever read) and
because a single ship string among many troop counts was enough for 'naval'.
Strings below are the real ``troop_reports.extracted_context`` values.
"""

from __future__ import annotations

from pipeline.classifiers.battle_type import extract_categories, infer_battle_type


def test_marathon_with_a_passing_fleet_is_field_not_naval() -> None:
    decision = infer_battle_type(
        76,
        "Battle of Marathon",
        "The Persians landed at Marathon bay. The Athenian army marched to meet them.",
        ["490s BC conflicts", "Amphibious operations"],
        ["cavalry", "infantry", "naval", "total"],
        [
            "10,000-11,000 men",
            ">100,000 armed oarsmen and sailors (reserve)",
            ">200 supply ships",
            "600 triremes",
            "25,000 infantry",
            "1,000 cavalry",
        ],
    )

    assert decision.battle_type == "field"


def test_malvern_hill_harrisons_landing_is_not_an_amphibious_signal() -> None:
    decision = infer_battle_type(
        75,
        "Battle of Malvern Hill",
        "McClellan's army fell back toward Harrison's Landing, where gunboats waited.",
        [],
        ["artillery", "naval", "total"],
        ["12-14 batteries", "54,000", "33 batteries, 171 guns (more in reserve)", "3 ships"],
    )

    assert decision.battle_type == "field"


def test_yom_kippur_war_with_naval_vessels_among_many_counts_is_field() -> None:
    decision = infer_battle_type(
        158,
        "Yom Kippur War",
        "Egyptian and Syrian forces attacked across the Suez Canal and the Golan Heights.",
        [],
        ["air", "armour", "artillery", "naval", "total"],
        [
            "375,000-415,000 troops",
            "650,000-850,000 troops (100,000+ crossed)",
            "140 helicopters",
            "120,000 troops",
            "104 naval vessels",
        ],
    )

    assert decision.battle_type == "field"


def test_operation_torch_is_amphibious_when_the_article_says_so_repeatedly() -> None:
    decision = infer_battle_type(
        141,
        "Operation Torch",
        "Torch was the first major Allied amphibious operation. The amphibious landings "
        "were followed by an amphibious assault on Oran.",
        [],
        ["air", "armour", "naval", "total"],
        ["107,000 troops", "125,000 troops", "350 warships", "11 submarines"],
    )

    assert decision.battle_type == "amphibious"


def test_a_fleet_only_report_set_is_naval_without_a_category() -> None:
    # Okpo: only naval and unsplit-total reports, with a "warships" figure.
    decision = infer_battle_type(
        92,
        "Battle of Okpo",
        "Yi Sun-sin's fleet attacked the Japanese anchored at Okpo.",
        [],
        ["naval", "total"],
        ["50 transports", "46 fishing boats", "43 warships", "2 small ships"],
    )

    assert decision.battle_type == "naval"


def test_an_aerial_category_on_a_campaign_with_ground_forces_is_field() -> None:
    # Stalingrad and Bagration carry "Aerial operations and battles of the
    # Eastern Front" as one category among dozens.
    decision = infer_battle_type(
        109,
        "Battle of Stalingrad",
        "The battle for the city lasted months.",
        ["Aerial operations and battles of the Eastern Front (World War II)"],
        ["air", "armour", "artillery", "total"],
        ["1,040,000 troops"],
    )

    assert decision.battle_type == "field"


def test_a_land_battle_with_a_read_article_and_no_signal_is_field() -> None:
    decision = infer_battle_type(
        20, "Battle of Austerlitz", "Napoleon defeated the allied army.", [], ["total"], []
    )

    assert decision.battle_type == "field"


def test_extract_categories_reads_rendered_html_category_links() -> None:
    html = (
        '<a href="/wiki/Category:Naval_battles_of_the_Imjin_War" title="Category:x">x</a>'
        '<a href="/wiki/Category:1592_in_Korea">y</a>'
        '<a href="/wiki/Category:Naval_battles_of_the_Imjin_War">dup</a>'
    )

    assert extract_categories(html) == ["Naval battles of the Imjin War", "1592 in Korea"]
