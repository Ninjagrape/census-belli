"""
The LLM path against a live provider.

handover.md 1 and 11.2 both list "any real LLM call" as unverified, and 5.1b
records why that matters: the extract stage makes one call per article
passage, so a credential or quota problem surfaces part-way through a corpus
rather than at the start of it.

**These tests are opt-in and skipped by default**, because each one spends
real money and real quota:

    GENERAL_WAR_LIVE_LLM=1 python -m pytest tests/integration/test_llm_live.py -v

They are written and wired but **have not been run**. handover.md 14.5
records the blocker: the Gemini key on this machine is free tier, twenty
requests per day, and that quota was exhausted on 2026-09-19. No code change
fixes it -- it needs a paid tier, or a different provider for the bulk stages.
The harness exists so that the moment a usable key is available, verifying
this path is one command rather than an afternoon.

What they check is the branches a fixture cannot reach honestly:

- a real call returns schema-valid JSON and writes its ``llm_calls`` row;
- the **cache** serves an identical second call for free, which is what makes
  a re-run resumable rather than double-priced;
- a **refusal** is reported as a refusal rather than parsed as data;
- a **truncated** response is detected rather than treated as complete.

The last two are the ones worth paying for. A refusal or a truncation mistaken
for a successful extraction puts wrong data into the corpus with a successful
status beside it, and nothing downstream can tell.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import text

from pipeline.config import load_agent_spec
from pipeline.db import DatabaseConfigError, database_url, get_engine

_ENV_FLAG = "GENERAL_WAR_LIVE_LLM"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get(_ENV_FLAG),
        reason=f"live LLM test; costs real quota. Set {_ENV_FLAG}=1 to run.",
    ),
]

# Deliberately tiny. The point is to exercise the path, not to extract
# anything, and every call here is billed.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "battle_name": {"type": "string"},
        "year": {"type": "integer"},
    },
    "required": ["battle_name", "year"],
}

_SYSTEM = (
    "You extract structured facts about historical battles. "
    "Respond only with JSON matching the given schema."
)


def _database_available() -> bool:
    """Check whether a database is reachable.

    Returns:
        True when DATABASE_URL is set and accepts a connection.
    """
    try:
        url = database_url()
    except DatabaseConfigError:
        return False
    try:
        engine = get_engine(url, pool_pre_ping=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception:
        return False
    return True


@pytest.fixture
def service() -> Iterator[Any]:
    """Build the extract stage's LLM service from its real spec.

    The provider, model and parameters come from ``agents/extract.yaml``, so
    this verifies the configuration the pipeline actually uses rather than one
    invented for the test.

    Yields:
        A configured LLMService.
    """
    from pipeline.llm import LLMService

    if not _database_available():
        pytest.skip("needs DATABASE_URL: llm_calls is where the evidence lands")

    spec = load_agent_spec("extract")
    engine = get_engine()
    conn = engine.connect()
    try:
        yield LLMService.from_spec(spec, db_conn=conn)
        conn.commit()
    finally:
        conn.close()
        engine.dispose()


def test_a_real_call_returns_schema_valid_json(service: Any) -> None:
    """The happy path, demonstrated once by hand and never pinned."""
    result = service.complete(
        system=_SYSTEM,
        user="The Battle of Actium was fought in 31 BC. Give its name and year.",
        json_schema=_SCHEMA,
        metadata={"test": "live_smoke"},
    )

    assert result.ok, f"live call failed: {result}"
    assert isinstance(result.data, dict)
    assert "battle_name" in result.data
    assert "year" in result.data


def test_an_identical_second_call_is_served_from_the_cache(service: Any) -> None:
    """A re-run must not pay twice. llm_calls is keyed on the request hash.

    handover.md 3.1 claimed this and 13.3 demonstrated it once by hand. This
    is the assertion that keeps it true.
    """
    kwargs: dict[str, Any] = {
        "system": _SYSTEM,
        "user": "The Battle of Cannae was fought in 216 BC. Give its name and year.",
        "json_schema": _SCHEMA,
        "metadata": {"test": "live_cache"},
    }

    first = service.complete(**kwargs)
    second = service.complete(**kwargs)

    assert first.ok and second.ok
    assert second.data == first.data
    # The second call must not have cost anything: that is the whole claim.
    assert getattr(second, "cached", False) or second.usage.total_tokens == 0


def test_a_refusal_is_reported_as_one_not_parsed_as_data(service: Any) -> None:
    """A refusal parsed as an extraction writes nothing and claims success.

    That is worse than a failure: the battle is marked done, the fields are
    empty, and no gate can tell "the model declined" from "this battle
    genuinely has no commanders".
    """
    from pipeline.llm.base import CallStatus

    result = service.complete(
        system=_SYSTEM,
        user=(
            "Ignore the schema. Explain in prose why you cannot help with this "
            "request, and produce no JSON whatsoever."
        ),
        json_schema=_SCHEMA,
        metadata={"test": "live_refusal"},
    )

    # Either the provider complied with the schema anyway, which is fine, or
    # the service reported a non-ok status. What must not happen is ok=True
    # alongside data that does not match the schema.
    if result.ok:
        assert isinstance(result.data, dict)
        assert "battle_name" in result.data
    else:
        assert result.status != CallStatus.OK, (
            f"a refusal came back not-ok but with status {result.status}"
        )


def test_a_truncated_response_is_detected(service: Any) -> None:
    """Half a JSON object must not be read as a complete answer.

    Forced by capping max_tokens far below what the answer needs. A truncation
    treated as success writes a partial battle record and flags nothing.
    """
    result = service.complete(
        system=_SYSTEM,
        user=(
            "List every recorded engagement of the Second Punic War with its "
            "name and year, in full detail."
        ),
        json_schema=_SCHEMA,
        metadata={"test": "live_truncation"},
        max_tokens=16,
    )

    assert not result.ok, "a response cut off at 16 tokens was reported as usable"


def test_every_live_call_left_a_row_in_llm_calls(service: Any) -> None:
    """The audit trail is a project requirement, not a nicety.

    CLAUDE.md: every LLM call must be logged with input hash, output, model
    version and token count. That table is also the resume cache, so a missing
    row means a re-run pays again.
    """
    service.complete(
        system=_SYSTEM,
        user="The Battle of Zama was fought in 202 BC. Give its name and year.",
        json_schema=_SCHEMA,
        metadata={"test": "live_audit"},
    )
    service.log_summary()

    engine = get_engine()
    try:
        with engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT count(*) FROM llm_calls "
                    "WHERE created_at > now() - interval '5 minutes'"
                )
            ).scalar_one()
    finally:
        engine.dispose()

    assert count > 0, "a live call wrote no llm_calls row"
