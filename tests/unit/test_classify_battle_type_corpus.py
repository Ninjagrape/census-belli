"""
Battle-type regressions built from the real pilot corpus.

The pilot typed Marathon, Malvern Hill, the Yom Kippur War and Operation
Torch as naval and 133 of 155 battles as unknown, because the stored URL never
matched the crawled article's filename (so no article was ever read) and
because a single ship string among many troop counts was enough for 'naval'.
Strings below are the real ``troop_reports.extracted_context`` values.
"""

from __future__ import annotations

import pytest
import structlog.testing

from pipeline.classifiers.battle_type import extract_categories, infer_battle_type, infer_campaign


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


def test_naval_vessels_among_many_counts_do_not_make_a_battle_naval() -> None:
    decision = infer_battle_type(
        158,
        "Crossing of the Suez Canal",
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
        "Landings at Torch",
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


# ─── Campaigns, amphibious, guerrilla, and the two category false positives ──


@pytest.mark.parametrize(
    "name",
    ["Six-Day War", "Yom Kippur War", "1982 Lebanon War", "1948 Arab–Israeli War"],
)
def test_a_war_title_without_battle_or_siege_is_a_campaign(name: str) -> None:
    decision = infer_campaign(1, name)

    assert decision is not None
    assert decision.battle_type == "campaign"


@pytest.mark.parametrize(
    "name",
    ["Battle of Warsaw", "Siege of Vienna", "Battle of Chancellorsville", "Raid on Entebbe"],
)
def test_battle_titles_and_look_alikes_are_not_campaigns(name: str) -> None:
    assert infer_campaign(1, name) is None


def test_a_war_title_naming_a_battle_stays_an_engagement() -> None:
    assert infer_campaign(1, "Battle of the War Memorial") is None


def test_a_span_over_180_days_is_a_campaign_and_180_is_not() -> None:
    assert infer_campaign(1, "Expedition Example", 181) is not None
    assert infer_campaign(1, "Expedition Example", 180) is None
    assert infer_campaign(1, "Expedition Example", 91) is None
    assert infer_campaign(1, "Expedition Example", None) is None


@pytest.mark.parametrize(
    ("name", "span"),
    [
        ("Battle of Verdun", 300),
        ("Battle of Stalingrad", 200),
        ("Siege of Tobruk", 241),
        ("Siege of Example", 365),
        ("Battles of Ypres", 400),
    ],
)
def test_battle_and_siege_titles_are_exempt_from_the_span_rule(name: str, span: int) -> None:
    assert infer_campaign(1, name, span) is None


def test_siege_of_leningrad_is_a_campaign_because_it_lasted_over_two_years() -> None:
    decision = infer_battle_type(1, "Siege of Leningrad", "text", [], [], [], 872)

    assert decision.battle_type == "campaign"


@pytest.mark.parametrize(
    ("name", "span"), [("Siege of Leningrad", 872), ("Sieges of Gibraltar", 1000)]
)
def test_a_siege_over_two_years_is_a_campaign(name: str, span: int) -> None:
    decision = infer_campaign(1, name, span)

    assert decision is not None
    assert decision.battle_type == "campaign"


def test_a_siege_under_two_years_is_still_a_siege() -> None:
    decision = infer_battle_type(1, "Siege of Tobruk", "text", [], [], [], 241)

    assert decision.battle_type == "siege_offensive"


def test_a_year_long_siege_such_as_rome_537_538_stays_a_siege() -> None:
    assert infer_campaign(1, "Siege of Rome (537–538)", 375) is None
    decision = infer_battle_type(1, "Siege of Rome (537–538)", "text", [], [], [], 375)

    assert decision.battle_type == "siege_offensive"


@pytest.mark.parametrize(
    ("name", "span"),
    [
        ("Operation Bagration", 10),
        ("Operation Barbarossa", 166),
        ("Operation Overlord", 80),
        ("Operation Torch", 0),
        ("Operation Torch", None),
    ],
)
def test_an_operation_title_is_a_campaign_whatever_its_span(name: str, span: int | None) -> None:
    decision = infer_campaign(1, name, span)

    assert decision is not None
    assert decision.battle_type == "campaign"


@pytest.mark.parametrize(
    "name",
    [
        "Battle of the Operation Hills",
        "Siege of Operation Fort",
        "Raid on Operation Base",
        "Operational Maneuver at Example",
        "Co-operation of Example",
        "Operation",
    ],
)
def test_operation_look_alikes_are_not_campaigns(name: str) -> None:
    assert infer_campaign(1, name) is None


def test_a_crisis_title_is_a_campaign_like_a_war() -> None:
    decision = infer_campaign(1, "Suez Crisis", None, "month")

    assert decision is not None
    assert decision.battle_type == "campaign"
    assert infer_campaign(1, "Battle of the Crisis Pass") is None


@pytest.mark.parametrize("precision", ["decade", "century", "millennium"])
def test_a_coarse_date_precision_is_a_campaign_even_for_a_battle_title(precision: str) -> None:
    decision = infer_campaign(122, "Battle of Moscow", None, precision)

    assert decision is not None
    assert decision.battle_type == "campaign"


@pytest.mark.parametrize("precision", ["day", "month", "year", None])
def test_finer_precisions_are_not_campaigns_on_precision_alone(precision: str | None) -> None:
    assert infer_campaign(1, "Battle of Cannae", None, precision) is None


def test_the_precision_reaches_infer_battle_type() -> None:
    decision = infer_battle_type(
        122, "Battle of Moscow", "text", [], [], [], None, date_precision="decade"
    )

    assert decision.battle_type == "campaign"


def test_an_end_date_before_the_start_is_not_a_campaign_signal() -> None:
    assert infer_campaign(1, "Battle of Ulm", -1) is None


@pytest.mark.parametrize(
    "name",
    ["Invasion of Poland", "Norman conquest of England", "Mongol invasion of Hungary"],
)
def test_invasion_and_conquest_titles_are_campaigns(name: str) -> None:
    decision = infer_campaign(1, name)

    assert decision is not None
    assert decision.battle_type == "campaign"


def test_a_battle_title_with_invasion_in_it_is_not_a_campaign() -> None:
    assert infer_campaign(1, "Battle of the Invasion Beaches") is None


def test_each_campaign_demotion_is_logged_with_its_reason() -> None:
    with structlog.testing.capture_logs() as logs:
        infer_campaign(7, "Invasion of Poland")
        infer_campaign(8, "Operation Example", 200)
        infer_campaign(9, "Battle of Verdun", 300)

    demoted = [entry for entry in logs if entry["event"] == "classify_campaign_demoted"]
    assert [entry["battle_id"] for entry in demoted] == [7, 8]
    assert all(entry["reason"] for entry in demoted)


def test_campaign_wins_over_a_naval_category() -> None:
    decision = infer_battle_type(
        1, "Falklands War", "text", ["Naval battles of the Falklands War"], ["naval"], ["10 ships"]
    )

    assert decision.battle_type == "campaign"


def test_operation_badr_is_not_a_siege_from_a_country_grouping_category() -> None:
    decision = infer_battle_type(
        2,
        "Action at Badr (1973)",
        "Egyptian forces crossed the Suez Canal and stormed the Bar Lev Line.",
        ["Sieges involving Israel", "Battles of the Yom Kippur War"],
        ["infantry", "armour", "total"],
        ["100,000 men", "1,000 tanks"],
    )

    assert decision.battle_type == "field"


def test_a_genuine_siege_category_still_makes_a_siege() -> None:
    decision = infer_battle_type(3, "Fall of Example", "text", ["Sieges of Example"], [], [])

    assert decision.battle_type == "siege_offensive"


def test_fort_donelson_is_field_when_land_reports_outweigh_the_naval_category() -> None:
    decision = infer_battle_type(
        4,
        "Battle of Fort Donelson",
        "Grant's army besieged the fort while gunboats bombarded it.",
        ["Naval battles of the American Civil War"],
        ["infantry", "infantry", "artillery", "naval"],
        ["27,000 men", "12,000 men", "100 guns", "7 gunboats"],
    )

    assert decision.battle_type == "field"


def test_actium_keeps_its_naval_category_when_naval_reports_match_land_ones() -> None:
    decision = infer_battle_type(
        5,
        "Battle of Actium",
        "The fleets met off the coast of Greece.",
        ["Naval battles of the Roman Republic"],
        ["infantry", "naval"],
        ["80,000 legionaries", "400 ships"],
    )

    assert decision.battle_type == "naval"


def test_inchon_is_amphibious_when_the_article_says_so_repeatedly_up_front() -> None:
    article = (
        "The Battle of Inchon was an amphibious invasion. The amphibious assault began "
        "at dawn. A second amphibious wave followed. " + "Troops moved inland. " * 50
    )

    decision = infer_battle_type(6, "Battle of Inchon", article, [], [], [])

    assert decision.battle_type == "amphibious"


def test_one_amphibious_mention_is_not_enough() -> None:
    decision = infer_battle_type(
        7, "Battle of Example", "An amphibious feint was tried. " + "Fighting. " * 50, [], [], []
    )

    assert decision.battle_type == "field"


def test_guerrilla_needs_a_category_or_repeated_mentions() -> None:
    by_category = infer_battle_type(
        8, "Battle of Example", "Some text.", ["Guerrilla warfare"], [], []
    )
    by_text = infer_battle_type(
        9, "Ambush at Example", "Guerrillas struck. " * 4 + "The guerrilla band fled.", [], [], []
    )
    passing = infer_battle_type(
        10, "Battle of Example", "A guerrilla band watched from the hills.", [], [], []
    )

    assert by_category.battle_type == "guerrilla"
    assert by_text.battle_type == "guerrilla"
    assert passing.battle_type == "field"


def test_a_naval_category_beats_guerrilla_text() -> None:
    decision = infer_battle_type(
        11,
        "Naval Action",
        "Guerrilla " * 6,
        ["Naval battles of the Example War"],
        ["naval"],
        ["10 ships"],
    )

    assert decision.battle_type == "naval"


def test_siege_defensive_and_combined_are_never_produced() -> None:
    cases = [
        infer_battle_type(1, "Siege of Example", "defended", [], [], []),
        infer_battle_type(2, "Battle of Example", "text", ["Sieges of Example"], [], []),
        infer_battle_type(3, "Battle of Example", "text", [], ["infantry", "naval"], ["1 ship"]),
    ]

    produced = {d.battle_type for d in cases}
    assert not produced & {"siege_defensive", "combined"}


def test_a_sieges_involving_category_counts_when_the_lead_says_siege() -> None:
    decision = infer_battle_type(
        12,
        "Defence of Example",
        "The siege of the town lasted a month.",
        ["Sieges involving France"],
        [],
        [],
    )

    assert decision.battle_type == "siege_offensive"


def test_a_sieges_involving_category_is_ignored_when_the_lead_is_silent() -> None:
    article = "Forces advanced. " * 200 + "A siege followed."

    decision = infer_battle_type(14, "Action Example", article, ["Sieges involving Israel"], [], [])

    assert decision.battle_type == "field"


def test_a_long_article_with_a_few_guerrilla_mentions_is_not_guerrilla() -> None:
    article = "Guerrillas struck. " * 4 + "The line held. " * 2000

    decision = infer_battle_type(15, "Battle of Example", article, [], [], [])

    assert decision.battle_type == "field"


def test_guerrilla_mentions_that_never_reach_the_lead_do_not_count() -> None:
    article = "Fighting went on. " * 100 + "Guerrillas struck. " * 6

    decision = infer_battle_type(16, "Battle of Example", article, [], [], [])

    assert decision.battle_type == "field"


def test_a_long_article_with_a_few_amphibious_mentions_is_not_amphibious() -> None:
    article = "An amphibious plan. " * 3 + "The line held. " * 3000

    decision = infer_battle_type(
        17, "Battle of Example", article, [], ["infantry", "naval"], ["5,000 men", "10 ships"]
    )

    assert decision.battle_type != "amphibious"


def test_actium_like_land_reports_on_both_sides_still_type_naval_when_ships_dominate() -> None:
    decision = infer_battle_type(
        18,
        "Battle of Actium",
        "The fleets met off the coast of Greece.",
        ["Naval battles of the Roman Republic"],
        ["infantry", "cavalry", "infantry", "cavalry", "naval", "naval"],
        ["80,000 infantry", "12,000 cavalry", "400 ships", "230 ships", "500 vessels"],
    )

    assert decision.battle_type == "naval"
