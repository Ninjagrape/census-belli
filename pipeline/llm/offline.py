"""
Offline LLM processing for the General WAR pipeline.

Offline is the default mode for every LLM stage. Instead of calling an API, a
stage defers each uncached request: :class:`OfflineProvider` collects them and
writes one batch file under ``data/llm_requests/``. The user then answers the
file with Claude Code subagents (``/process-llm-batch``), which run under the
Claude subscription rather than billed API calls, and imports the answers
into the ``llm_calls`` cache.

Workflow:

1. Run the stage. Cached requests are answered from ``llm_calls``; the rest
   are written to ``data/llm_requests/<stage>_<timestamp>.jsonl`` and the
   orchestrator stops after the stage.
2. Answer the file (``/process-llm-batch <file>``).
3. Import the response JSONL into ``llm_calls``.
4. Re-run the stage. Every request now hits the cache.

The ``request_hash`` is the correlation key. :class:`OfflineProvider` reports
the spec's real provider and model, so the hash equals the one the API path
computes: an answer obtained offline is a cache hit for an ``llm_mode: api``
run and the reverse.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
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
from pipeline.llm.parsing import parse_structured

__all__ = [
    "IMPORT_PROVIDER",
    "ExportResult",
    "ImportResult",
    "OfflineProvider",
    "REQUEST_DIR",
    "RESPONSE_DIR",
    "ValidationReport",
    "agent_model_for",
    "answered_hashes",
    "build_chunks",
    "build_request_line",
    "export_pending",
    "import_responses",
    "index_requests",
    "pending_count",
    "validate_responses",
    "write_request_file",
]

logger = structlog.get_logger()

REQUEST_DIR: Path = Path("data/llm_requests")
RESPONSE_DIR: Path = Path("data/llm_responses")

# Recorded as ``llm_calls.provider`` on imported rows, so an audit query can
# tell an answer a Claude Code subagent produced from a billed API call.
IMPORT_PROVIDER = "claude_code"
DEFAULT_IMPORT_MODEL = "claude-code-subagent"

_HASH_PREFIX = 12


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


@dataclass
class ValidationReport:
    """Outcome of validating one response file against its requests."""

    path: Path
    ok: int = 0
    # (hash prefix, reason) per rejected line.
    errors: list[tuple[str, str]] = field(default_factory=list)
    answered: set[str] = field(default_factory=set)


# ─── Request lines ───────────────────────────────────────────────────────────


def agent_model_for(model: str) -> str:
    """Pick the Claude Code subagent tier that should answer requests for a model.

    Args:
        model: The spec's model identifier, e.g. ``"claude-haiku-5-5"``.

    Returns:
        ``"haiku"``, ``"sonnet"`` or ``"opus"``; ``"sonnet"`` for anything
        unrecognised, since a wrong tier up is safer than one down.
    """
    lowered = model.lower()
    for tier in ("haiku", "sonnet", "opus"):
        if tier in lowered:
            return tier
    return "sonnet"


def build_request_line(
    request: LLMRequest, digest: str, stage: str, provider: str, model: str
) -> dict[str, Any]:
    """Build the JSON object written for one deferred request.

    The single line format shared by :func:`export_pending` and
    :class:`OfflineProvider`, so a batch file means the same thing whichever
    path wrote it.

    Args:
        request: The request.
        digest: Its :func:`~pipeline.llm.base.request_hash`.
        stage: Pipeline stage name.
        provider: Provider from the agent spec.
        model: Model identifier from the agent spec.

    Returns:
        The line as a mapping.
    """
    return {
        "request_hash": digest,
        "stage": stage,
        "provider": provider,
        "model": model,
        "agent_model": agent_model_for(model),
        "system": request.system,
        "user": request.user,
        "json_schema": request.json_schema,
        "schema_name": request.schema_name,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "metadata": request.metadata,
    }


def write_request_file(
    stage: str, lines: Iterable[dict[str, Any]], output_dir: Path | None = None
) -> Path:
    """Write request lines to a new timestamped JSONL file.

    Args:
        stage: Stage name, used in the file name.
        lines: Lines from :func:`build_request_line`.
        output_dir: Override the default ``data/llm_requests/``.

    Returns:
        The path written. A second file in the same second gets a numeric
        suffix rather than overwriting the first.
    """
    out = output_dir or REQUEST_DIR
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{stage}_{timestamp}.jsonl"
    suffix = 2
    while path.exists():
        path = out / f"{stage}_{timestamp}_{suffix}.jsonl"
        suffix += 1

    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


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
    lines: list[dict[str, Any]] = []
    cached = 0
    seen: set[str] = set()

    for request in requests:
        digest = compute_hash(request, provider, model)
        if digest in seen:
            continue
        if conn is not None and find_completed_call(conn, digest) is not None:
            cached += 1
            continue
        seen.add(digest)
        lines.append(build_request_line(request, digest, stage, provider, model))

    path = (output_dir or REQUEST_DIR) / f"{stage}_none.jsonl"
    if lines:
        path = write_request_file(stage, lines, output_dir)

    logger.info(
        "llm_requests_exported",
        stage=stage,
        exported=len(lines),
        cached=cached,
        path=str(path) if lines else None,
    )

    return ExportResult(path=path, exported=len(lines), cached=cached)


# ─── Deferring provider ──────────────────────────────────────────────────────


class OfflineProvider:
    """
    A provider that defers every request instead of calling out.

    It reports the spec's real ``llm_provider`` and ``llm_model`` as its
    ``name`` and ``model``, so the request hash matches what the API client
    would compute and the ``llm_calls`` cache is shared between modes.
    Requests accumulate in memory until :meth:`flush` writes the batch file.
    """

    def __init__(self, name: str, model: str, *, output_dir: Path | None = None) -> None:
        """Initialise the provider.

        Args:
            name: The spec's ``llm_provider``, e.g. ``"anthropic"``.
            model: The spec's ``llm_model``.
            output_dir: Override the default ``data/llm_requests/``.
        """
        self.name = name
        self.model = model
        self._output_dir = output_dir
        self._pending: dict[str, LLMRequest] = {}

    @property
    def pending(self) -> int:
        """How many distinct requests are waiting to be flushed."""
        return len(self._pending)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Record a request for offline processing.

        Args:
            request: The request that found no cached answer.

        Returns:
            A response with status DEFERRED. No network call is made.
        """
        digest = compute_hash(request, self.name, self.model)
        self._pending.setdefault(digest, request)
        return LLMResponse(
            status=CallStatus.DEFERRED,
            provider=self.name,
            model=self.model,
            request_hash=digest,
            usage=TokenUsage(),
            cost_usd=0.0,
            attempts=0,
            error="pending offline processing",
        )

    def flush(self, stage: str) -> Path | None:
        """Write the pending requests to a batch file and clear the list.

        Args:
            stage: Stage name, used in the file name and on each line.

        Returns:
            The batch file path, or None when nothing was pending.
        """
        if not self._pending:
            return None
        lines = [
            build_request_line(request, digest, stage, self.name, self.model)
            for digest, request in self._pending.items()
        ]
        path = write_request_file(stage, lines, self._output_dir)
        self._pending.clear()
        return path


# ─── Import and validation ───────────────────────────────────────────────────


def index_requests(
    request_files: Iterable[Path] | None = None, request_dir: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Index request lines by hash.

    Args:
        request_files: Explicit files to index. Without it, every
            ``*.jsonl`` directly under ``request_dir`` is indexed.
        request_dir: Directory to scan; defaults to :data:`REQUEST_DIR`.

    Returns:
        Hash to request line. Unreadable lines are logged and skipped.
    """
    if request_files is None:
        directory = request_dir or REQUEST_DIR
        files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    else:
        files = list(request_files)

    index: dict[str, dict[str, Any]] = {}
    for path in files:
        for _lineno, line in _read_jsonl(path):
            digest = line.get("request_hash")
            if isinstance(digest, str) and digest:
                index.setdefault(digest, line)
    return index


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line number, object)`` for each readable JSON object line.

    Args:
        path: A JSONL file.

    Yields:
        Parsed lines. Blank and malformed lines are skipped, the latter with
        a warning.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning("jsonl_line_unparseable", path=str(path), line=lineno, error=str(e))
                continue
            if isinstance(decoded, dict):
                yield lineno, decoded


def _check_line(
    lineno: int, raw_line: str, index: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any] | None, str, str]:
    """Validate one response line.

    Args:
        lineno: Line number, for messages.
        raw_line: The stripped line text.
        index: Known requests by hash.

    Returns:
        ``(line, hash prefix, error)``. ``error`` is empty and ``line`` is the
        decoded object when the line is acceptable.
    """
    try:
        line = json.loads(raw_line)
    except json.JSONDecodeError as e:
        return None, f"line{lineno}", f"unparseable JSON: {e}"
    if not isinstance(line, dict):
        return None, f"line{lineno}", "line is not a JSON object"

    digest = line.get("request_hash")
    data = line.get("data")
    prefix = str(digest)[:_HASH_PREFIX] if digest else f"line{lineno}"
    if not digest or not isinstance(data, dict):
        return None, prefix, "missing request_hash or data object"

    request = index.get(str(digest))
    if request is None:
        return None, prefix, "request_hash matches no known request"

    schema = request.get("json_schema")
    _parsed, error = parse_structured(json.dumps(data), schema if isinstance(schema, dict) else {})
    if error is not None:
        return None, prefix, error
    return line, prefix, ""


def validate_responses(
    path: Path,
    *,
    request_files: Iterable[Path] | None = None,
    index: dict[str, dict[str, Any]] | None = None,
) -> ValidationReport:
    """Validate a response file without touching the database.

    Args:
        path: The response JSONL file.
        request_files: Explicit request files to validate against.
        index: A prebuilt request index, to avoid re-reading request files
            when validating many response files.

    Returns:
        A report of accepted lines, rejected lines and the answered hashes.
    """
    known = index if index is not None else index_requests(request_files)
    report = ValidationReport(path=path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            line, prefix, error = _check_line(lineno, raw, known)
            if line is None:
                report.errors.append((prefix, error))
                continue
            report.ok += 1
            report.answered.add(str(line["request_hash"]))
    return report


def import_responses(
    path: Path,
    conn: Any,
    *,
    provider: str = IMPORT_PROVIDER,
    model: str | None = None,
    request_files: Iterable[Path] | None = None,
    index: dict[str, dict[str, Any]] | None = None,
) -> ImportResult:
    """Import completed responses into ``llm_calls``.

    Each line must be a JSON object with:

    - ``request_hash``: the hash from the request file
    - ``data``: the extracted structured data (a dict)

    Optionally ``stage`` (falls back to the request's stage) and
    ``model_used``, which is recorded as the audit row's model. Each line's
    ``data`` is validated against the JSON schema of the request it answers;
    a line whose hash matches no known request, or whose data fails
    validation, is counted as an error and not imported.

    Responses whose ``request_hash`` already has a successful row in
    ``llm_calls`` are skipped, making re-imports safe.

    Args:
        path: Path to the response JSONL file.
        conn: Database connection for writing to ``llm_calls``.
        provider: Provider recorded on the audit row.
        model: Model recorded when a line carries no ``model_used``. Falls
            back to ``"claude-code-subagent"``.
        request_files: Explicit request files to resolve hashes against;
            by default every ``*.jsonl`` under :data:`REQUEST_DIR`.
        index: A prebuilt request index.

    Returns:
        An :class:`ImportResult` with counts.
    """
    known = index if index is not None else index_requests(request_files)
    fallback_model = model or DEFAULT_IMPORT_MODEL
    imported = 0
    skipped = 0
    errors = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue

            line, prefix, error = _check_line(lineno, raw, known)
            if line is None:
                logger.warning("import_line_rejected", line=lineno, hash=prefix, error=error)
                errors += 1
                continue

            digest = str(line["request_hash"])
            if find_completed_call(conn, digest) is not None:
                skipped += 1
                continue

            answered = known[digest]
            metadata = answered.get("metadata")
            request = LLMRequest(
                system="",
                user="",
                json_schema={},
                metadata=metadata if isinstance(metadata, dict) else {},
            )
            response = LLMResponse(
                status=CallStatus.OK,
                provider=provider,
                model=str(line.get("model_used") or fallback_model),
                request_hash=digest,
                data=line["data"],
                usage=TokenUsage(),
                cost_usd=0.0,
            )

            stage_name = str(line.get("stage") or answered.get("stage") or "unknown")
            if log_call(conn, stage_name, request, response) is not None:
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


# ─── Chunking ────────────────────────────────────────────────────────────────


def answered_hashes(response_dir: Path) -> set[str]:
    """Collect every hash that has a response line in a directory.

    Args:
        response_dir: A ``data/llm_responses/<stem>/`` directory.

    Returns:
        Hashes with a response carrying a data object. Not validated: this
        only decides what a split can skip, and ``validate`` is the gate.
    """
    answered: set[str] = set()
    if not response_dir.is_dir():
        return answered
    for path in sorted(response_dir.glob("*.jsonl")):
        for _lineno, line in _read_jsonl(path):
            digest = line.get("request_hash")
            if isinstance(digest, str) and isinstance(line.get("data"), dict):
                answered.add(digest)
    return answered


def build_chunks(
    request_file: Path, *, chunk_size: int, skip: set[str] | None = None
) -> list[dict[str, Any]]:
    """Group a request file's lines into compact chunks for subagents.

    A chunk carries the system prompt and schema once, then only each
    request's hash and user text, so a subagent does not re-read them per
    request. Requests are grouped by (system, schema, agent model) so a chunk
    never mixes prompts.

    Args:
        request_file: A batch file written by :class:`OfflineProvider` or
            :func:`export_pending`.
        chunk_size: Maximum requests per chunk.
        skip: Hashes to leave out, e.g. those already answered.

    Returns:
        Chunk objects in file order of their first request.

    Raises:
        ValueError: If ``chunk_size`` is below 1.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    skipped = skip or set()
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen: set[str] = set()

    for _lineno, line in _read_jsonl(request_file):
        digest = line.get("request_hash")
        if not isinstance(digest, str) or digest in skipped or digest in seen:
            continue
        seen.add(digest)
        schema = line.get("json_schema") or {}
        system = str(line.get("system") or "")
        agent_model = str(line.get("agent_model") or agent_model_for(str(line.get("model") or "")))
        key = (system, json.dumps(schema, sort_keys=True), agent_model)
        group = groups.setdefault(
            key,
            {
                "stage": line.get("stage"),
                "agent_model": agent_model,
                "system": system,
                "json_schema": schema,
                "schema_name": line.get("schema_name", "extraction"),
                "requests": [],
            },
        )
        group["requests"].append({"request_hash": digest, "user": line.get("user", "")})

    chunks: list[dict[str, Any]] = []
    for group in groups.values():
        reqs = group["requests"]
        for start in range(0, len(reqs), chunk_size):
            chunks.append({**group, "requests": reqs[start : start + chunk_size]})
    return chunks


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
