"""An override the loader's per-row rules would drop is a curation error."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.reconcilers.load import DEFAULT_MIN_PLAUSIBLE_TROOPS, _build_troop_reports
from pipeline.reconcilers.records import ReconcileCounts


class _Rows:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def execute(self, _statement: Any) -> _Rows:
        return _Rows(self._rows)


def _row(url: str, *, branch: str = "total", scope: str | None = "army") -> tuple[Any, ...]:
    return (1, 7, 3, 9, "manual_entry", url, branch, 50_000.0, scope, True, False, False, "", -200)


def test_an_override_the_row_rules_would_drop_raises_naming_url_side_and_reason() -> None:
    conn = _FakeConn([_row("override:ov-0042", branch="naval")])

    with pytest.raises(ValueError, match=r"override:ov-0042.*side 7.*naval"):
        _build_troop_reports(
            conn,
            ReconcileCounts(),
            min_plausible_troops=DEFAULT_MIN_PLAUSIBLE_TROOPS,
        )


def test_an_ordinary_dropped_row_is_still_dropped_quietly() -> None:
    counts = ReconcileCounts()
    conn = _FakeConn([_row("https://example/x", branch="naval")])

    assert _build_troop_reports(conn, counts, min_plausible_troops=100.0) == []
    assert counts.excluded_naval == 1
