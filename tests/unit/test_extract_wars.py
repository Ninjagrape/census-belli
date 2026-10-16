"""Unit tests for choosing a battle's war and campaign from its part_of names."""

from __future__ import annotations

import pytest

from pipeline.extractors.wars import WarCampaign, classify_part_of


@pytest.mark.parametrize(
    ("part_of", "expected"),
    [
        (["Q6271", "Second Punic War"], WarCampaign("Second Punic War", None)),
        (
            ["Q8676", "American Civil War", "Appomattox campaign"],
            WarCampaign("American Civil War", "Appomattox campaign"),
        ),
        # A name carrying both words is a campaign, not a war.
        (
            ["Western Desert campaign of World War II"],
            WarCampaign(
                None,
                "Western Desert campaign of World War II",
            ),
        ),
        # A combined label glued from two other entries is dropped.
        (
            ["Third Silesian War", "Third Silesian War and Seven Years' War"],
            WarCampaign("Third Silesian War", None),
        ),
        (["Dnieper-Carpathian offensive"], WarCampaign(None, "Dnieper-Carpathian offensive")),
        (["Q6271"], WarCampaign(None, None)),
        (["Khalid ibn al-Walid"], WarCampaign(None, None)),
        ([], WarCampaign(None, None)),
        (["French invasion of Russia"], WarCampaign("French invasion of Russia", None)),
    ],
)
def test_classify_part_of(part_of: list[str], expected: WarCampaign) -> None:
    assert classify_part_of(part_of) == expected
