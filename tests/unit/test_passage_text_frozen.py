"""Freeze the LLM passage text the extract stage builds from the wikitext fixtures.

The hashes below were frozen at 2808819. The passage text is part of every
offline LLM cache key, so a change to ``build_passages`` output (through
``strip_markup``, ``clean_article_text``, ``split_list_items`` or the article
splitter) re-keys the cache and forces every cached request to be re-answered.
If this test fails, the change leaked outside the infobox path; fix the change
rather than the hashes unless re-keying the cache is the deliberate intent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline.extractors.article import build_passages

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "extract"

EXPECTED_HASHES: dict[str, str] = {
    "actium_naval": "182286f67d28611b9f9651d5d355bbac5c491c434556a2dfac4f0f8ce5b263ba",
    "alesia_siege": "773b395819165b9ffebdde905924621972dbdb42496f1bc17b4a2bc2478ee009",
    "austerlitz": "8b632ed204751f47f56de482dd2c53d24bb442030becbb541ef98a60c67e3fde",
    "contested_field": "1c89b0858d93cfc2b83af6ffe832ec8bc9e85ea51cdaa8aa09bd640e2bcb7689",
    "croix_dorade_strength": "699ad1dfcabfd285ee0a64649e3586e2eee56f0059051c2575cfe76fa3e27f04",
    "jena_auerstedt": "db91035144785c55f4cbeac19c245895cb40b82e52379301bff5f31421008188",
    "la_suffel_headings": "240898cf1a539b6116353f8ac4b1fa98c6893cfb9ffef5c7d016ceca73f22a54",
    "okinawa_headings": "b143b7f8358778094b261fb9f467feba392812fbe721b068018680271b066d06",
    "okinawa_strength": "128940002cbaa586978a4cdb29538bdfa2b354d763dcb9ba33114340d81b6af1",
    "quatre_bras_headings": "3ee4f8aa678fd40bf71d06774ca397bb9d1f82e75df6c2a38d95c990953351b6",
    "timor_headings": "5fc59e4cddeabca9ba31c389a799f6329dc43a1c4f92ca01c8e8f856fcfe5ccb",
    "tomaszow_headings": "eddc0d95b1689aad61731b57f9ac246fa5f05a4f5cf01fed2ee199342f92eddf",
}


def _passage_hash(stem: str) -> str:
    """Hash the canonical serialisation of a fixture's passages.

    Args:
        stem: Fixture file stem, also used as the battle name.

    Returns:
        Hex sha256 of the JSON-serialised passage tuples.
    """
    raw = (FIXTURES / f"{stem}.wikitext").read_text(encoding="utf-8")
    passages = build_passages(stem, raw)
    serialised = json.dumps(
        [(p.section, p.text, p.index, p.total) for p in passages], ensure_ascii=False
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def test_every_wikitext_fixture_is_frozen() -> None:
    """A new fixture must get a hash, so the freeze covers it."""
    stems = {path.stem for path in FIXTURES.glob("*.wikitext")}
    assert stems == set(EXPECTED_HASHES)


@pytest.mark.parametrize("stem", sorted(EXPECTED_HASHES))
def test_passage_text_is_byte_identical(stem: str) -> None:
    """Passage text for each fixture matches the hash frozen at 2808819."""
    assert _passage_hash(stem) == EXPECTED_HASHES[stem]
