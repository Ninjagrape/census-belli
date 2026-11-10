"""Where the pipeline's data tree lives.

One setting, the ``GENERAL_WAR_DATA_ROOT`` environment variable, decides the
root of ``raw``, ``processed``, ``imputed``, ``logs``, ``llm_requests`` and
``llm_responses``, so a second corpus can run from the same checkout. The
variable is read at call time, never at import time. ``data/backups`` is not
under the root: it stays fixed and shared.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DATA_ROOT_ENV", "DEFAULT_DATA_ROOT", "data_path", "data_root"]

DATA_ROOT_ENV = "GENERAL_WAR_DATA_ROOT"
DEFAULT_DATA_ROOT = "data"


def data_root() -> Path:
    """Return the data root for this process.

    Returns:
        ``$GENERAL_WAR_DATA_ROOT`` when set and non-empty, else ``data``
        (relative to the working directory).
    """
    value = os.environ.get(DATA_ROOT_ENV, "").strip()
    return Path(value or DEFAULT_DATA_ROOT)


def data_path(*parts: str) -> Path:
    """Join path parts onto the data root.

    Args:
        *parts: Path components below the root, e.g. ``"processed", "reconcile"``.

    Returns:
        The joined path, not created on disk.
    """
    return data_root().joinpath(*parts)
