"""Atomic file writes, so a reader never sees a half-written output."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

__all__ = ["remove_if_present", "write_bytes_atomic", "write_text_atomic"]


def write_bytes_atomic(path: Path, data: bytes) -> Path:
    """Write ``data`` to a temp file beside ``path``, then ``os.replace`` it into place.

    Args:
        path: Destination file; parents are created.
        data: The bytes to write.

    Returns:
        The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return path


def write_text_atomic(path: Path, text: str) -> Path:
    """Atomic UTF-8 text write; see :func:`write_bytes_atomic`.

    Args:
        path: Destination file.
        text: The text, written with ``\\n`` newlines untouched.

    Returns:
        The path.
    """
    return write_bytes_atomic(path, text.encode("utf-8"))


def remove_if_present(path: Path) -> bool:
    """Delete a file if it exists.

    Args:
        path: The file.

    Returns:
        Whether a file was removed.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
