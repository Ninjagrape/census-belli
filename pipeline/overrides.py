"""
Manual overrides: hand-curated, cited corrections that stages apply on every run.

``config/overrides.yaml`` holds a list of entries. Each stage applies the
entries that belong to it *after* its own logic, so a correction survives a
re-run instead of being overwritten by it. The design is in
``docs/handover/07-phase2-overrides-design.md``. This module only loads,
validates and reports; applying an entry is the owning stage's job.

File format::

    version: 1
    entries:
      - id: ov-0001                  # ^ov-\\d{4}$, unique in the file
        kind: outcome                # one of KINDS_BY_STAGE
        target: {battle: battle_of_borodino}
        value: {"French Empire": victory, "Russian Empire": defeat}
        source: "Chandler, The Campaigns of Napoleon (1966), p. 807"
        note: "why"
        reviewed: {by: lee, on: 2026-10-17}

Kinds, their stage, exact target keys and value shape:

======================  ========  =========================  ================================
kind                    stage     target keys                value
======================  ========  =========================  ================================
``outcome``             extract   battle                     {side_label: outcome_level}, >= 2
``battle_date``         extract   battle                     {start, end|null, precision}
``side_polity``         extract   battle, side               non-empty str
``commander_link``      resolve   battle, side, mention      Q-id or the string ``none``
``general_merge``       resolve   general                    Q-id (the survivor)
``battle_type``         classify  battle                     battle_type enum member
``attribution``         classify  battle, side               list of {general, role, rank,
                                                             weight, reports_to}
======================  ========  =========================  ================================

``battle`` is ``battles.source_slug``, ``side`` is ``battle_sides.side_label``,
``mention`` is the commander name as the extract stage wrote it, and a
``general`` key is a Wikidata Q-id or a corpus-local ``canonical_name``.
Targets are never database ids. Enum members are read from
``config/schema.sql``, the single source of truth.

Every violation raises :class:`OverrideError` naming the entry id, at load
time, before any stage writes. Two entries with the same kind and the same
normalised target (stripped, case-folded) are a duplicate-target error.

Public API:

* :func:`load_overrides` -> :class:`OverrideSet` (``for_stage``, ``of_kind``,
  ``len``). A missing file yields an empty set.
* :class:`OverrideEntry`: ``id, kind, stage, target, value, source, note,
  reviewed_by, reviewed_on``. ``target`` is a read-only mapping of str to str;
  ``value`` is the normalised value (for ``attribution`` every item carries
  all five keys, ``reports_to`` being None when absent). Entries compare and
  hash by identity.
* :class:`OverrideReport`: per stage run, call ``applied(entry)`` or
  ``unmatched(entry, reason)`` for every entry of the stage, then
  ``write(root)`` to produce ``overrides_<stage>.json`` for the
  ``overrides_gate`` quality check. Set ``limited=True`` for a ``--limit``
  run, where unmatched entries are expected.
* :func:`manual_source_id`: the ``sources`` row manual provenance hangs off.
* :func:`overrides_path`: the file a stage run reads (spec param
  ``overrides_path`` or :data:`DEFAULT_PATH`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import structlog
import yaml
from sqlalchemy import text

from pipeline.extractors.dates import PRECISIONS

__all__ = [
    "DEFAULT_PATH",
    "KINDS_BY_STAGE",
    "OverrideEntry",
    "OverrideError",
    "OverrideReport",
    "OverrideSet",
    "load_overrides",
    "manual_source_id",
    "overrides_path",
]

logger = structlog.get_logger()

DEFAULT_PATH: Final[Path] = Path("config/overrides.yaml")

KINDS_BY_STAGE: Final[dict[str, tuple[str, ...]]] = {
    "extract": ("outcome", "battle_date", "side_polity"),
    "resolve": ("commander_link", "general_merge"),
    "classify": ("battle_type", "attribution"),
}

_STAGE_OF_KIND: Final[dict[str, str]] = {
    kind: stage for stage, kinds in KINDS_BY_STAGE.items() for kind in kinds
}

_TARGET_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "outcome": ("battle",),
    "battle_date": ("battle",),
    "side_polity": ("battle", "side"),
    "commander_link": ("battle", "side", "mention"),
    "general_merge": ("general",),
    "battle_type": ("battle",),
    "attribution": ("battle", "side"),
}

_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {"id", "kind", "target", "value", "source", "note", "reviewed"}
)
_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^ov-\d{4}$")
_QID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^Q[1-9]\d*$")
_SCHEMA_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "config" / "schema.sql"
_ATTRIBUTION_KEYS: Final[frozenset[str]] = frozenset(
    {"general", "role", "rank", "weight", "reports_to"}
)
_DATE_VALUE_KEYS: Final[frozenset[str]] = frozenset({"start", "end", "precision"})
_MIN_OUTCOME_SIDES: Final[int] = 2


class OverrideError(ValueError):
    """The overrides file is malformed or contradicts itself."""


def _schema_enum(name: str) -> frozenset[str]:
    """Read an enum's labels from ``config/schema.sql``.

    Args:
        name: The Postgres enum type name.

    Returns:
        The enum's labels.

    Raises:
        OverrideError: If the schema file has no such enum.
    """
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    match = re.search(rf"CREATE TYPE {name} AS ENUM \((.*?)\);", sql, re.DOTALL)
    if match is None:
        raise OverrideError(f"enum {name} not found in {_SCHEMA_PATH}")
    # Drop SQL comments first: they can contain quoted words.
    body = re.sub(r"--[^\n]*", "", match.group(1))
    return frozenset(re.findall(r"'([^']+)'", body))


OUTCOME_LEVELS: Final[frozenset[str]] = _schema_enum("outcome_level")
BATTLE_TYPES: Final[frozenset[str]] = _schema_enum("battle_type")
COMMAND_ROLES: Final[frozenset[str]] = _schema_enum("command_role")


@dataclass(frozen=True, eq=False)
class OverrideEntry:
    """One validated override. Compares and hashes by identity."""

    id: str
    kind: str
    stage: str
    target: Mapping[str, str]
    value: Any
    source: str
    note: str
    reviewed_by: str
    reviewed_on: str


@dataclass(frozen=True)
class OverrideSet:
    """All entries of one overrides file, in file order."""

    entries: tuple[OverrideEntry, ...] = ()
    path: Path = DEFAULT_PATH

    def __len__(self) -> int:
        return len(self.entries)

    def for_stage(self, stage: str) -> list[OverrideEntry]:
        """Return the entries a stage owns.

        Args:
            stage: A stage name; unknown stages own nothing.

        Returns:
            The stage's entries in file order.
        """
        return [e for e in self.entries if e.stage == stage]

    def of_kind(self, kind: str) -> list[OverrideEntry]:
        """Return the entries of one kind, in file order.

        Args:
            kind: An override kind.

        Returns:
            The matching entries.
        """
        return [e for e in self.entries if e.kind == kind]


# ─── Validation ──────────────────────────────────────────────────────────────


def _fail(label: str, problem: str) -> OverrideError:
    return OverrideError(f"{label}: {problem}")


def _non_empty_str(label: str, what: str, raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise _fail(label, f"{what} must be a non-empty string")
    return raw.strip()


def _exact_keys(label: str, what: str, raw: Any, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _fail(label, f"{what} must be a mapping")
    if set(raw) != keys:
        raise _fail(
            label, f"{what} keys must be exactly {sorted(keys)}, got {sorted(map(str, raw))}"
        )
    return raw


def _general_key(label: str, what: str, raw: Any) -> str:
    """A general is a Q-id or a corpus-local canonical name."""
    return _non_empty_str(label, what, raw)


def _qid(label: str, what: str, raw: Any) -> str:
    if not isinstance(raw, str) or not _QID_PATTERN.match(raw):
        raise _fail(label, f"{what} must be a Wikidata Q-id like Q517, got {raw!r}")
    return raw


def _date_text(label: str, what: str, raw: Any) -> str:
    """Normalise a date or ISO string to an ISO string."""
    if isinstance(raw, datetime):
        raise _fail(label, f"{what} must be a date, not a datetime")
    if isinstance(raw, date):
        return raw.isoformat()
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw.strip()).isoformat()
        except ValueError:
            pass
    raise _fail(label, f"{what} must be an ISO date like 2026-10-17, got {raw!r}")


def _validate_target(label: str, kind: str, raw: Any) -> dict[str, str]:
    keys = _TARGET_KEYS[kind]
    target = _exact_keys(label, "target", raw, frozenset(keys))
    return {key: _non_empty_str(label, f"target.{key}", target[key]) for key in keys}


def _validate_outcome(label: str, raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or len(raw) < _MIN_OUTCOME_SIDES:
        raise _fail(label, f"value must map at least {_MIN_OUTCOME_SIDES} side labels to outcomes")
    value: dict[str, str] = {}
    for side, outcome in raw.items():
        side_label = _non_empty_str(label, "value side label", side)
        if outcome not in OUTCOME_LEVELS:
            raise _fail(label, f"outcome {outcome!r} for {side_label!r} is not an outcome_level")
        value[side_label] = outcome
    return value


def _validate_battle_date(label: str, raw: Any) -> dict[str, str | None]:
    value = _exact_keys(label, "value", raw, _DATE_VALUE_KEYS)
    # An unquoted 1812-09-07 arrives as a date; the extract stage wants text.
    start = value["start"].isoformat() if isinstance(value["start"], date) else value["start"]
    end = value["end"].isoformat() if isinstance(value["end"], date) else value["end"]
    start = _non_empty_str(label, "value.start", start)
    if end is not None:
        end = _non_empty_str(label, "value.end", end)
    if value["precision"] not in PRECISIONS:
        raise _fail(label, f"value.precision must be one of {sorted(PRECISIONS)}")
    return {"start": start, "end": end, "precision": value["precision"]}


def _validate_commander_link(label: str, raw: Any) -> str:
    if raw == "none":
        return "none"
    return _qid(label, "value", raw)


def _validate_enum_value(label: str, raw: Any, members: frozenset[str], what: str) -> str:
    if raw not in members:
        raise _fail(label, f"value {raw!r} is not a {what}")
    return str(raw)


def _validate_attribution_item(label: str, index: int, item: Any) -> dict[str, Any]:
    where = f"value[{index}]"
    if not isinstance(item, dict):
        raise _fail(label, f"{where} must be a mapping")
    unknown = set(item) - _ATTRIBUTION_KEYS
    missing = {"general", "role", "rank", "weight"} - set(item)
    if unknown or missing:
        raise _fail(label, f"{where} keys: unknown {sorted(unknown)}, missing {sorted(missing)}")
    if item["role"] not in COMMAND_ROLES:
        raise _fail(label, f"{where}.role {item['role']!r} is not a command_role")
    rank = item["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise _fail(label, f"{where}.rank must be an integer >= 0")
    weight = item["weight"]
    if isinstance(weight, bool) or not isinstance(weight, int | float) or not 0 <= weight <= 1:
        raise _fail(label, f"{where}.weight must be a number in [0, 1]")
    reports_to = item.get("reports_to")
    if reports_to is not None:
        reports_to = _general_key(label, f"{where}.reports_to", reports_to)
    return {
        "general": _general_key(label, f"{where}.general", item["general"]),
        "role": item["role"],
        "rank": rank,
        "weight": float(weight),
        "reports_to": reports_to,
    }


def _validate_attribution(label: str, raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise _fail(label, "value must be a non-empty list")
    items = [_validate_attribution_item(label, i, item) for i, item in enumerate(raw)]
    generals = [item["general"] for item in items]
    if len(set(generals)) != len(generals):
        raise _fail(label, "a general appears twice in value")
    for item in items:
        parent = item["reports_to"]
        if parent is not None and (parent == item["general"] or parent not in generals):
            raise _fail(
                label, f"{item['general']!r} reports_to {parent!r}, which is not another general"
            )
    return items


def _validate_value(label: str, kind: str, target: dict[str, str], raw: Any) -> Any:
    if kind == "outcome":
        return _validate_outcome(label, raw)
    if kind == "battle_date":
        return _validate_battle_date(label, raw)
    if kind == "side_polity":
        return _non_empty_str(label, "value", raw)
    if kind == "commander_link":
        return _validate_commander_link(label, raw)
    if kind == "general_merge":
        survivor = _qid(label, "value", raw)
        if target["general"] == survivor:
            raise _fail(label, "a general cannot be merged into itself")
        return survivor
    if kind == "battle_type":
        return _validate_enum_value(label, raw, BATTLE_TYPES, "battle_type")
    return _validate_attribution(label, raw)


def _validate_reviewed(label: str, raw: Any) -> tuple[str, str]:
    # YAML 1.1 reads an unquoted `on` key as boolean True, which is how the
    # documented `reviewed: {by: lee, on: 2026-10-17}` arrives from safe_load.
    if isinstance(raw, dict):
        raw = {("on" if key is True else key): item for key, item in raw.items()}
    reviewed = _exact_keys(label, "reviewed", raw, frozenset({"by", "on"}))
    by = _non_empty_str(label, "reviewed.by", reviewed["by"])
    return by, _date_text(label, "reviewed.on", reviewed["on"])


def _validate_entry(index: int, raw: Any, seen_ids: set[str]) -> OverrideEntry:
    label = f"entry #{index}"
    if not isinstance(raw, dict):
        raise _fail(label, "must be a mapping")
    if isinstance(raw.get("id"), str):
        label = raw["id"]
    if set(raw) != _ENTRY_KEYS:
        raise _fail(
            label,
            f"keys must be exactly {sorted(_ENTRY_KEYS)}; "
            f"missing {sorted(_ENTRY_KEYS - set(raw))}, "
            f"unknown {sorted(map(str, set(raw) - _ENTRY_KEYS))}",
        )
    entry_id = raw["id"]
    if not isinstance(entry_id, str) or not _ID_PATTERN.match(entry_id):
        raise _fail(label, "id must match ov-NNNN")
    if entry_id in seen_ids:
        raise _fail(label, "duplicate id")
    seen_ids.add(entry_id)
    kind = raw["kind"]
    if not isinstance(kind, str) or kind not in _STAGE_OF_KIND:
        raise _fail(label, f"unknown kind {kind!r}; expected one of {sorted(_STAGE_OF_KIND)}")
    source = _non_empty_str(label, "source (the citation is required)", raw["source"])
    if not isinstance(raw["note"], str):
        raise _fail(label, "note must be a string")
    target = _validate_target(label, kind, raw["target"])
    value = _validate_value(label, kind, target, raw["value"])
    reviewed_by, reviewed_on = _validate_reviewed(label, raw["reviewed"])
    return OverrideEntry(
        id=entry_id,
        kind=kind,
        stage=_STAGE_OF_KIND[kind],
        target=MappingProxyType(target),
        value=value,
        source=source,
        note=raw["note"],
        reviewed_by=reviewed_by,
        reviewed_on=reviewed_on,
    )


def _target_key(entry: OverrideEntry) -> tuple[str, tuple[tuple[str, str], ...]]:
    return entry.kind, tuple((k, v.casefold()) for k, v in sorted(entry.target.items()))


def load_overrides(path: Path = DEFAULT_PATH) -> OverrideSet:
    """Load and validate the overrides file.

    Args:
        path: The YAML file. A missing file is an empty set.

    Returns:
        The validated entries, in file order.

    Raises:
        OverrideError: On any schema violation, naming the entry and problem.
    """
    if not path.exists():
        logger.info("overrides_file_absent", path=str(path))
        return OverrideSet(path=path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OverrideError(f"{path}: cannot read overrides: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"version", "entries"}:
        raise OverrideError(f"{path}: top level must have exactly the keys version and entries")
    version = document["version"]
    if isinstance(version, bool) or version != 1:
        raise OverrideError(f"{path}: version must be 1, got {version!r}")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise OverrideError(f"{path}: entries must be a list")
    seen_ids: set[str] = set()
    seen_targets: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
    entries: list[OverrideEntry] = []
    for index, raw in enumerate(raw_entries):
        entry = _validate_entry(index, raw, seen_ids)
        key = _target_key(entry)
        if key in seen_targets:
            raise _fail(entry.id, f"duplicate target, already overridden by {seen_targets[key]}")
        seen_targets[key] = entry.id
        entries.append(entry)
    return OverrideSet(entries=tuple(entries), path=path)


def overrides_path(params: dict[str, Any]) -> Path:
    """Return the overrides file a stage run should read.

    Args:
        params: The stage spec's params.

    Returns:
        ``params["overrides_path"]`` when set, else :data:`DEFAULT_PATH`.
    """
    configured = params.get("overrides_path")
    return Path(str(configured)) if configured else DEFAULT_PATH


# ─── Reporting ───────────────────────────────────────────────────────────────


@dataclass
class OverrideReport:
    """What one stage run did with its overrides, for the ``overrides_gate`` check.

    Args:
        stage: The stage name; must be a key of :data:`KINDS_BY_STAGE`.
        overrides: The loaded set; the stage's entries are counted.
        limited: True for a run restricted with ``--limit``, where unmatched
            entries are expected.
    """

    stage: str
    overrides: OverrideSet
    limited: bool = False
    _applied: dict[str, int] = field(init=False, repr=False)
    _unmatched: list[dict[str, str]] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.stage not in KINDS_BY_STAGE:
            raise ValueError(f"unknown stage {self.stage!r}")
        self._applied = dict.fromkeys(KINDS_BY_STAGE[self.stage], 0)

    def applied(self, entry: OverrideEntry) -> None:
        """Record that an entry changed (or confirmed) its target."""
        self._applied[entry.kind] += 1

    def unmatched(self, entry: OverrideEntry, reason: str) -> None:
        """Record that an entry found nothing to apply to, and why."""
        self._unmatched.append({"id": entry.id, "kind": entry.kind, "reason": reason})

    def write(self, root: Path) -> Path:
        """Write ``overrides_<stage>.json`` under ``root`` and log the counts.

        Args:
            root: The directory for processed output.

        Returns:
            The path written.
        """
        payload = {
            "stage": self.stage,
            "file": str(self.overrides.path),
            "entries": len(self.overrides.for_stage(self.stage)),
            "applied": dict(self._applied),
            "unmatched": list(self._unmatched),
            "limited": self.limited,
            "written_at": datetime.now().astimezone().isoformat(),
        }
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"overrides_{self.stage}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "overrides_applied",
            stage=self.stage,
            entries=payload["entries"],
            applied=payload["applied"],
            unmatched=len(self._unmatched),
            limited=self.limited,
        )
        return path


# ─── Provenance ──────────────────────────────────────────────────────────────

_FIND_MANUAL_SOURCE = text(
    "SELECT source_id FROM sources WHERE source_type = CAST('manual_entry' AS source_type) "
    "AND title = :title AND url IS NULL ORDER BY source_id LIMIT 1"
)
_INSERT_MANUAL_SOURCE = text(
    "INSERT INTO sources (source_type, title, url) "
    "VALUES (CAST('manual_entry' AS source_type), :title, NULL) RETURNING source_id"
)


def manual_source_id(conn: Any, entry: OverrideEntry) -> int:
    """Find or create the ``sources`` row that carries an entry's citation.

    Args:
        conn: An open database connection.
        entry: The override whose citation is the row's title.

    Returns:
        The source id; the same citation always yields the same row.
    """
    params = {"title": entry.source}
    row = conn.execute(_FIND_MANUAL_SOURCE, params).fetchone()
    if row is None:
        row = conn.execute(_INSERT_MANUAL_SOURCE, params).fetchone()
    return int(row[0])
