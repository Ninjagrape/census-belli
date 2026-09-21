"""
Offline LLM processing for the General WAR pipeline.

When no API key is available, or when the user wants to process requests
through a Claude Pro subscription instead of billed API calls, this module
serialises pending requests to files and imports completed responses back
into the ``llm_calls`` cache.

Workflow:

1. Run a stage's export: collects every request the stage would make,
   checks the ``llm_calls`` cache, and writes uncached requests to a JSONL
   file under ``data/llm_requests/``.
2. Process them via Claude.ai (upload the file), a Claude Code session
   (run ``/process-llm-batch``), or any other means.
3. Import the response JSONL into ``llm_calls``.
4. Re-run the pipeline stage. Every request hits the cache.

The ``request_hash`` is the correlation key: a request exported with hash
*X* must come back with hash *X*, and the next pipeline run will find it
in ``llm_calls`` by that hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from pipeline.llm.base import (
    CallStatus,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from pipeline.llm.base import (
    request_hash as compute_hash,
)
from pipeline.llm.call_log import find_completed_call, log_call

__all__ = [
    "ExportResult",
    "ImportResult",
    "REQUEST_DIR",
    "RESPONSE_DIR",
    "export_pending",
    "import_responses",
    "pending_count",
]

logger = structlog.get_logger()

REQUEST_DIR: Path = Path("data/llm_requests")
RESPONSE_DIR: Path = Path("data/llm_responses")


@dataclass(frozen=True)
class ExportResult:
    """Summary of an export run."""

    path: Path
    exported: int
    cached: int


@dataclass(frozen=True)
class ImportResult:
    """Summary of an import run."""

    imported: int
    skipped: int
    errors: int


def export_pending(
    stage: str,
    provider: str,
    model: str,
    requests: list[LLMRequest],
    conn: Any = None,
    output_dir: Path | None = None,
) -> ExportResult:
    """Export uncached requests to a JSONL file.

    Each line is a self-contained JSON object with the full prompt,
    schema, and the ``request_hash`` the import step will use to
    correlate the response.

    Args:
        stage: Pipeline stage name, e.g. ``"extract"``.
        provider: Provider from the agent spec, used for hash computation.
        model: Model identifier from the agent spec.
        requests: The requests to check and potentially export.
        conn: Database connection for cache lookups. Without one, every
            request is treated as uncached.
        output_dir: Override the default ``data/llm_requests/``.

    Returns:
        An :class:`ExportResult` with the output path and counts.
    """
    out = output_dir or REQUEST_DIR
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{stage}_{timestamp}.jsonl"

    exported = 0
    cached = 0

    with path.open("w", encoding="utf-8") as f:
        for request in requests:
            digest = compute_hash(request, provider, model)

            if conn is not None and find_completed_call(conn, digest) is not None:
                cached += 1
                continue

            line = {
                "request_hash": digest,
                "stage": stage,
                "provider": provider,
                "model": model,
                "system": request.system,
                "user": request.user,
                "json_schema": request.json_schema,
                "schema_name": request.schema_name,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "metadata": request.metadata,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            exported += 1

    if exported == 0:
        path.unlink(missing_ok=True)

    logger.info(
        "llm_requests_exported",
        stage=stage,
        exported=exported,
        cached=cached,
        path=str(path) if exported else None,
    )

    return ExportResult(path=path, exported=exported, cached=cached)


def import_responses(
    path: Path,
    conn: Any,
    *,
    provider: str = "offline",
    model: str = "claude-pro-subscription",
) -> ImportResult:
    """Import completed responses into ``llm_calls``.

    Each line must be a JSON object with at least:

    - ``request_hash``: the hash from the export file
    - ``data``: the extracted structured data (a dict)

    Optionally:

    - ``stage``: stage name for the audit row

    Responses whose ``request_hash`` already has a successful row in
    ``llm_calls`` are silently skipped, making re-imports safe.

    Args:
        path: Path to the response JSONL file.
        conn: Database connection for writing to ``llm_calls``.
        provider: Provider name recorded on the audit row. Defaults to
            ``"offline"`` to distinguish manual processing from API calls.
        model: Model name recorded on the audit row.

    Returns:
        An :class:`ImportResult` with counts.
    """
    imported = 0
    skipped = 0
    errors = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError as e:
                logger.warning("import_line_unparseable", line=lineno, error=str(e))
                errors += 1
                continue

            digest = line.get("request_hash")
            data = line.get("data")

            if not digest or not isinstance(data, dict):
                logger.warning(
                    "import_line_missing_fields",
                    line=lineno,
                    has_hash=bool(digest),
                    has_data=isinstance(data, dict),
                )
                errors += 1
                continue

            if find_completed_call(conn, digest) is not None:
                skipped += 1
                continue

            request = LLMRequest(
                system="",
                user="",
                json_schema={},
                metadata=line.get("metadata", {}),
            )

            response = LLMResponse(
                status=CallStatus.OK,
                provider=provider,
                model=model,
                request_hash=digest,
                data=data,
                usage=TokenUsage(),
                cost_usd=0.0,
            )

            stage_name = line.get("stage", "unknown")
            call_id = log_call(conn, stage_name, request, response)
            if call_id is not None:
                imported += 1
            else:
                errors += 1

    logger.info(
        "llm_responses_imported",
        imported=imported,
        skipped=skipped,
        errors=errors,
        path=str(path),
    )

    return ImportResult(imported=imported, skipped=skipped, errors=errors)


def pending_count(
    provider: str,
    model: str,
    requests: list[LLMRequest],
    conn: Any = None,
) -> tuple[int, int]:
    """Count how many requests are cached vs pending.

    Args:
        provider: Provider name for hash computation.
        model: Model identifier for hash computation.
        requests: The requests to check.
        conn: Database connection for cache lookups.

    Returns:
        ``(pending, cached)`` counts.
    """
    pending = 0
    cached = 0

    for request in requests:
        digest = compute_hash(request, provider, model)
        if conn is not None and find_completed_call(conn, digest) is not None:
            cached += 1
        else:
            pending += 1

    return pending, cached
