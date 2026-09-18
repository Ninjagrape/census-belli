"""
The contract every pipeline stage implements.

The orchestrator imports stage modules by name and calls ``run(spec)`` on
them. That is a loose arrangement: a typo in a function name, or a stage
that quietly takes different arguments, surfaces only when the stage is
reached -- which for the model stage is after hours of upstream work.

``check_conforms`` verifies the contract at import time instead, so a broken
stage fails before the pipeline starts rather than partway through.

A stage module must expose::

    def run(spec: dict, context: StageContext | None = None) -> None: ...

``context`` is optional so that a stage needing nothing but its spec can
declare ``run(spec)`` and still conform.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "StageConformanceError",
    "StageContext",
    "StageRunner",
    "check_conforms",
    "conforms",
]


@dataclass
class StageContext:
    """Shared services handed to a stage at run time.

    Passing these rather than letting stages construct their own keeps one
    connection pool and one LLM budget per run, and lets tests substitute
    doubles without patching module globals.

    Attributes:
        db_conn: An open database connection, or None when the stage does not
            touch the database.
        dry_run: Plan the work and log it without writing or calling out.
        limit: Process at most this many records. For smoke tests, and for
            costing an LLM stage before committing to the full corpus.
    """

    db_conn: Any = None
    dry_run: bool = False
    limit: int | None = None


@runtime_checkable
class StageRunner(Protocol):
    """Structural type for a stage module."""

    def run(self, spec: dict[str, Any]) -> None:
        """Execute the stage.

        Args:
            spec: The loaded agent spec, with overrides already applied.
        """
        ...


class StageConformanceError(TypeError):
    """Raised when a stage module does not implement the runner contract."""


def check_conforms(module: ModuleType) -> None:
    """Verify a stage module implements the contract.

    Args:
        module: The imported stage module.

    Raises:
        StageConformanceError: If ``run`` is missing, is not callable, or
            cannot accept a single positional spec argument.
    """
    name = getattr(module, "__name__", repr(module))

    run = getattr(module, "run", None)
    if run is None:
        raise StageConformanceError(
            f"{name} defines no run(); a stage module must expose run(spec: dict) -> None"
        )

    if not callable(run):
        raise StageConformanceError(f"{name}.run is {type(run).__name__}, not callable")

    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError):
        # Builtins and some C callables have no introspectable signature.
        # Callability is as far as the check can go.
        return

    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    accepts_var_positional = any(
        p.kind is p.VAR_POSITIONAL for p in signature.parameters.values()
    )

    if not positional and not accepts_var_positional:
        raise StageConformanceError(
            f"{name}.run{signature} takes no positional argument; it must accept "
            "the spec as its first argument"
        )

    required_after_first = [p for p in positional[1:] if p.default is inspect.Parameter.empty]
    if required_after_first:
        names = [p.name for p in required_after_first]
        raise StageConformanceError(
            f"{name}.run{signature} requires {names} beyond the spec. The "
            "orchestrator calls run(spec), so any further parameter needs a default."
        )


def conforms(module: ModuleType) -> bool:
    """Report whether a stage module implements the contract.

    Args:
        module: The imported stage module.

    Returns:
        True if the module conforms.
    """
    try:
        check_conforms(module)
    except StageConformanceError:
        return False
    return True
