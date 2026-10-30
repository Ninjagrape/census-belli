"""Battle lookup in the resolve store's missing-data logging, no database."""

from __future__ import annotations

from typing import Any

from pipeline.resolvers import Mention, ResolveCounts
from pipeline.resolvers.store import _battle_id_for, log_unresolved

# Two articles share a display name, so load_battle_index leaves the name key out.
_BATTLES = {"Battle_of_Alexandria_(30_BC)": 1, "Battle_of_Alexandria_(1801)": 2}


class _Conn:
    """Records the parameters of every statement executed."""

    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []

    def execute(self, _stmt: Any, params: dict[str, Any] | None = None) -> None:
        if params is not None:
            self.params.append(params)


def _mention(slug: str) -> Mention:
    return Mention(
        battle_slug=slug,
        battle_name="Battle of Alexandria",
        side_label="Rome",
        name="Unknown officer",
    )


def test_battle_id_for_prefers_slug_when_name_is_shared() -> None:
    assert _battle_id_for(_mention("Battle_of_Alexandria_(1801)"), _BATTLES) == 2


def test_battle_id_for_falls_back_to_name_without_slug() -> None:
    assert _battle_id_for(_mention(""), {"Battle of Alexandria": 7}) == 7


def test_battle_id_for_unknown_battle_is_none() -> None:
    assert _battle_id_for(_mention(""), _BATTLES) is None


def test_unresolved_mention_with_slug_is_logged_not_homeless() -> None:
    conn = _Conn()
    counts = ResolveCounts()
    homeless = log_unresolved(
        conn,
        [_mention("Battle_of_Alexandria_(30_BC)")],
        _BATTLES,
        {},
        counts,
        non_person=[_mention("Battle_of_Alexandria_(1801)")],
    )
    assert homeless == []
    assert counts.battles_not_in_db == 0
    assert counts.missing_rows == 2
    assert sorted(p["battle_id"] for p in conn.params if "battle_id" in p) == [1, 2]
