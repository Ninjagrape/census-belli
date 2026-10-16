"""Guards on the worked examples in agents/classify.yaml.

A few-shot that breaks its own schema teaches the model to break it, and one
whose weights do not sum to 1.0 teaches it to be rejected, so each example is
parsed and checked against the same rules the stage applies to an answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.classifiers.records import COMMAND_ROLES

SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "agents" / "classify.yaml"
EXPECTED_BATTLES = (
    "Actium",
    "Munda",
    "Mutina",
    "Chaeronea",
    "Salamis",
    "Mohi",
    "Königgrätz",
)


def _examples() -> list[tuple[str, dict[str, Any]]]:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    system: str = spec["command_role_prompt"]["system"]
    found = []
    for block in re.split(r"\n\s*\n(?=\s*Battle of )", system):
        header = re.match(r"\s*(Battle of [^\n]+)\n", block)
        answer = re.search(r"Answer: (\{.*\})\s*$", block, re.DOTALL)
        if header and answer:
            found.append((header.group(1), json.loads(answer.group(1))))
    return found


def test_every_requested_battle_has_an_example() -> None:
    headers = " ".join(header for header, _ in _examples())

    for battle in EXPECTED_BATTLES:
        assert battle in headers


@pytest.mark.parametrize(("header", "answer"), _examples())
def test_each_example_matches_the_answer_schema(header: str, answer: dict[str, Any]) -> None:
    rows = answer["classifications"]

    assert answer["is_defender"] in (True, False, None)
    assert sum(r["attribution_weight"] for r in rows) == pytest.approx(1.0, abs=0.01), header
    assert any(r["hierarchy_rank"] == 0 for r in rows), header
    names = {r["name"] for r in rows}
    for row in rows:
        assert row["command_role"] in COMMAND_ROLES
        assert row["reports_to"] is None or row["reports_to"] in names


def test_the_prompt_describes_is_defender_and_the_unresolved_line() -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    prompt = spec["command_role_prompt"]

    assert "is_defender" in prompt["system"]
    assert "{unresolved_formatted}" in prompt["user_template"]
    assert '"is_defender": true/false/null' in prompt["user_template"]
