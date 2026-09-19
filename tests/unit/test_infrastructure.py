"""
Unit tests for the process-level infrastructure: env loading and logging.

Both are the kind of code whose failure is silent. An env loader that reads
nothing leaves DATABASE_URL unset, and the integration suite then *skips*
rather than fails -- handover.md 1 records that a green run with skips looks
almost identical to a green run without. A logging configuration that never
runs is not an error either: structlog falls back to a default, so the
pipeline keeps logging, just not the way anyone decided.

Nothing here touches the real ``.env``. It holds live API keys, and the
project's rule is that a secret is confirmed set, never printed.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from pipeline.config import load_env
from pipeline.logging_config import DEFAULT_LEVEL, configure, is_configured

# Synthetic throughout. A test that needed a real credential to pass would be
# a test nobody could run.
_FAKE_URL = "postgresql+psycopg://user:password@localhost:5432/general_war_test"
_FAKE_KEY = "sk-not-a-real-key-0000"


@pytest.fixture(autouse=True)
def restore_logging_configuration() -> Iterator[None]:
    """Put structlog back as it was after every test in this module.

    ``configure()`` is process-global by nature, so a test that sets the level
    to ERROR silences every later test in the *suite* that asserts on captured
    log output. Three tests in test_quality.py and test_resolve.py failed
    exactly that way before this fixture existed, and they failed only when
    run together with this file -- the most annoying kind of test pollution,
    because each module passes on its own.

    Yields:
        None. The previous configuration is restored on teardown.
    """
    import pipeline.logging_config as logging_config

    saved = structlog.get_config()
    was_configured = logging_config.is_configured()
    try:
        yield
    finally:
        structlog.configure(**saved)
        logging_config._configured = was_configured


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the variables these tests set, so one cannot leak into another.

    Args:
        monkeypatch: pytest's environment patcher, which restores on teardown.
    """
    for name in ("GENERAL_WAR_TEST_URL", "GENERAL_WAR_TEST_KEY", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)


def write_env(path: Path, body: str, encoding: str) -> Path:
    """Write an env file in a given encoding.

    Args:
        path: Directory to write into.
        body: The file's text.
        encoding: Codec name, e.g. ``utf-8`` or ``utf-16``.

    Returns:
        The written file's path.
    """
    target = path / ".env"
    target.write_bytes(body.encode(encoding))
    return target


def test_a_plain_utf8_env_file_is_loaded(tmp_path: Path, clean_env: None) -> None:
    """The ordinary case: a UTF-8 file sets variables that were not set."""
    env = write_env(
        tmp_path,
        f"GENERAL_WAR_TEST_URL={_FAKE_URL}\nGENERAL_WAR_TEST_KEY={_FAKE_KEY}\n",
        "utf-8",
    )

    applied = load_env(env)

    assert sorted(applied) == ["GENERAL_WAR_TEST_KEY", "GENERAL_WAR_TEST_URL"]
    assert os.environ["GENERAL_WAR_TEST_URL"] == _FAKE_URL


def test_a_utf16_env_file_loads_without_a_bom_on_the_first_key(
    tmp_path: Path, clean_env: None
) -> None:
    """PowerShell writes UTF-16, and the BOM must not become part of a name.

    This is not hypothetical: the .env on this project's development machine
    is UTF-16-LE with a BOM, written by a PowerShell redirect. python-dotenv
    assumes UTF-8 and raised UnicodeDecodeError on the very first byte, so the
    file had never once been read. Decoding with an explicit-endianness codec
    fixes the crash but leaves a variable called "\\ufeffGENERAL_WAR_TEST_URL",
    which nothing looks up -- the same failure wearing a different hat.
    """
    env = write_env(tmp_path, f"GENERAL_WAR_TEST_URL={_FAKE_URL}\n", "utf-16")

    applied = load_env(env)

    assert applied == ["GENERAL_WAR_TEST_URL"]
    assert "﻿GENERAL_WAR_TEST_URL" not in os.environ
    assert os.environ["GENERAL_WAR_TEST_URL"] == _FAKE_URL


def test_a_utf8_bom_env_file_loads_cleanly(tmp_path: Path, clean_env: None) -> None:
    """Notepad's "UTF-8" writes a BOM too, with the same consequence."""
    env = write_env(tmp_path, f"GENERAL_WAR_TEST_URL={_FAKE_URL}\n", "utf-8-sig")

    assert load_env(env) == ["GENERAL_WAR_TEST_URL"]
    assert os.environ["GENERAL_WAR_TEST_URL"] == _FAKE_URL


def test_an_exported_variable_wins_over_the_file(tmp_path: Path, clean_env: None) -> None:
    """A deliberately exported value must not be overridden by a stale file.

    CI injects secrets as real environment variables, and a developer who
    exports DATABASE_URL to point at a scratch database means it.
    """
    os.environ["GENERAL_WAR_TEST_URL"] = "exported-wins"
    env = write_env(tmp_path, f"GENERAL_WAR_TEST_URL={_FAKE_URL}\n", "utf-8")

    assert load_env(env) == []
    assert os.environ["GENERAL_WAR_TEST_URL"] == "exported-wins"

    # ...unless the caller asks for the opposite explicitly.
    assert load_env(env, override=True) == ["GENERAL_WAR_TEST_URL"]
    assert os.environ["GENERAL_WAR_TEST_URL"] == _FAKE_URL


def test_a_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    """The deployed case is real variables and no file at all."""
    assert load_env(tmp_path / "nonexistent.env") == []


def test_an_unreadable_env_file_does_not_take_the_run_down(tmp_path: Path) -> None:
    """Graceful degradation: the variables may well be exported already.

    A stack trace out of dotenv's internals says nothing about which file was
    at fault or what to do, and killing a pipeline run over it is worse than
    proceeding with whatever the environment already holds.
    """
    broken = tmp_path / ".env"
    broken.write_bytes(b"\x80\x81\x82 not text at all\n")

    assert load_env(broken) == []


def test_load_env_returns_names_and_never_values(tmp_path: Path, clean_env: None) -> None:
    """A secret is confirmed set, never printed. That includes what we return."""
    env = write_env(tmp_path, f"GENERAL_WAR_TEST_KEY={_FAKE_KEY}\n", "utf-8")

    applied = load_env(env)

    assert applied == ["GENERAL_WAR_TEST_KEY"]
    assert _FAKE_KEY not in " ".join(applied)


# ─── Logging configuration ───────────────────────────────────────────────────


def test_configure_actually_configures_structlog() -> None:
    """Nothing called structlog.configure before, so the format was an accident."""
    configure(level="INFO", json_logs=True, force=True)

    assert is_configured()
    assert structlog.is_configured()


def test_the_level_filters_what_is_emitted(capsys: pytest.CaptureFixture[str]) -> None:
    """The level is the difference between a readable run and 40,000 lines."""
    configure(level="WARNING", json_logs=True, force=True)
    logger = structlog.get_logger()

    logger.info("should_not_appear")
    logger.warning("should_appear")

    err = capsys.readouterr().err
    assert "should_not_appear" not in err
    assert "should_appear" in err


def test_an_unrecognised_level_warns_rather_than_raising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typo in an environment variable must not stop a pipeline run."""
    configure(level="VERBOSE", json_logs=True, force=True)

    assert is_configured()
    assert "log_level_unrecognised" in capsys.readouterr().err


def test_json_output_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """A long run has to be greppable afterwards, which the default is not."""
    configure(level="INFO", json_logs=True, force=True)
    structlog.get_logger().info("battle_extracted", slug="battle_of_actium", sides=2)

    line = [ln for ln in capsys.readouterr().err.splitlines() if "battle_extracted" in ln][-1]
    decoded = json.loads(line)

    assert decoded["event"] == "battle_extracted"
    assert decoded["slug"] == "battle_of_actium"
    assert decoded["level"] == "info"
    assert decoded["timestamp"].endswith("Z")


def test_noisy_libraries_are_held_at_warning() -> None:
    """httpx logs every request at INFO; a crawl would bury our own lines."""
    configure(level="DEBUG", json_logs=True, force=True)

    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("sqlalchemy.engine").level >= logging.WARNING


def test_a_second_call_is_a_no_op_without_force(capsys: pytest.CaptureFixture[str]) -> None:
    """An imported stage must not restyle the logs of the run that imported it."""
    configure(level="ERROR", json_logs=True, force=True)
    configure(level="DEBUG", json_logs=True)  # no force: must not take effect

    structlog.get_logger().info("still_filtered")

    assert "still_filtered" not in capsys.readouterr().err


def test_the_default_level_is_the_documented_one() -> None:
    """DEFAULT_LEVEL is referenced in the README and .env.example."""
    assert DEFAULT_LEVEL == "INFO"
