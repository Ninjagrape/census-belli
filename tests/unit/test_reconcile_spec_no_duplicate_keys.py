"""agents/reconcile.yaml has no duplicate mapping keys (PyYAML keeps the last silently)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

SPEC = Path(__file__).resolve().parents[2] / "agents" / "reconcile.yaml"


class _StrictLoader(yaml.SafeLoader):
    """A loader that raises on a repeated key within one mapping."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def test_reconcile_spec_has_no_duplicate_keys() -> None:
    with SPEC.open(encoding="utf-8") as handle:
        yaml.load(handle, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass


def test_the_strict_loader_does_flag_a_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate key 'a'"):
        yaml.load("a: 1\na: 2\n", Loader=_StrictLoader)  # noqa: S506
