"""
CLI for the offline LLM processing workflow.

Usage::

    # Show what is pending for a stage
    python -m scripts.llm_offline status --stage extract

    # Export uncached requests to a JSONL file (a stage run also writes one)
    python -m scripts.llm_offline export --stage extract

    # Split a request file into compact chunks for Claude Code subagents
    python -m scripts.llm_offline split --file data/llm_requests/extract_<ts>.jsonl

    # Check responses against the request schemas, without touching the database
    python -m scripts.llm_offline validate --responses data/llm_responses/extract_<ts>/

    # Import completed responses
    python -m scripts.llm_offline import --dir data/llm_responses/extract_<ts>/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from pipeline.config import load_agent_spec, load_env
from pipeline.db import get_engine
from pipeline.llm.base import LLMRequest
from pipeline.llm.base import request_hash as compute_hash
from pipeline.llm.call_log import find_completed_call
from pipeline.llm.factory import llm_params
from pipeline.llm.offline import (
    REQUEST_DIR,
    RESPONSE_DIR,
    answered_hashes,
    build_chunks,
    export_pending,
    import_responses,
    index_requests,
    validate_responses,
)

logger = structlog.get_logger()


def _build_requests_from_spec(
    spec: dict[str, Any],
    config: dict[str, Any],
) -> list[LLMRequest]:
    """Build the LLM requests a stage would make from its input files.

    This reads the stage's input data and constructs the requests the
    stage runner would send, without actually running the stage. Each
    stage has different input formats, so this dispatches on stage name.

    Args:
        spec: The loaded agent spec.
        config: The LLM params extracted from the spec.

    Returns:
        A list of LLMRequest objects.
    """
    stage = spec.get("stage", "unknown")
    params = spec.get("params", {}) or {}
    processed_root = Path(params.get("processed_root", "data/processed"))

    if stage == "extract":
        return _extract_requests(spec, processed_root)
    if stage == "resolve":
        return _resolve_requests(spec, processed_root)
    if stage == "classify":
        return _classify_requests(spec, processed_root)

    logger.warning("stage_request_builder_not_implemented", stage=stage)
    return []


def _extract_requests(
    spec: dict[str, Any],
    processed_root: Path,
) -> list[LLMRequest]:
    """Build extract-stage requests from crawled battle files.

    Mirrors pipeline/stages/extract.py's _llm_sources exactly, so the
    request_hash computed here matches what a live extract run would compute
    and exported responses cache back correctly.
    """
    from pipeline.extractors import build_passages
    from pipeline.extractors.article import DEFAULT_MAX_PASSAGE_CHARS, render_template
    from pipeline.stages.extract import (
        RAW_ROOT,
        _citation_records,
        _read_text,
        discover_battles,
    )

    prompt = spec.get("prompt") or {}
    system_prompt = str(prompt.get("system") or "")
    user_template = str(prompt.get("user_template") or "")
    schema = prompt.get("output_schema") or {}
    if not system_prompt or not user_template or not isinstance(schema, dict):
        logger.warning("extract_spec_missing_prompt_parts")
        return []

    params = spec.get("params") or {}
    raw_root = Path(str(params.get("raw_root", RAW_ROOT)))
    max_passage_chars = int(params.get("max_passage_chars", DEFAULT_MAX_PASSAGE_CHARS))
    max_tokens = int(params.get("llm_max_tokens", 4096))
    temperature = params.get("llm_temperature")

    battles = discover_battles(raw_root)
    if not battles:
        logger.info("no_battles_to_export", raw_root=str(raw_root))
        return []

    def _render(passage: Any) -> str:
        return render_template(
            user_template,
            {
                "battle_name": passage.battle_name,
                "source_type": passage.source_type,
                "source_title": passage.source_title,
                "text_passage": passage.text,
            },
        )

    def _metadata(battle_slug: str, source_ref: str, passage: Any) -> dict[str, Any]:
        return {
            "battle_slug": battle_slug,
            "source_ref": source_ref,
            "passage_index": passage.index,
            "passage_total": passage.total,
            "passage_section": passage.section,
        }

    requests: list[LLMRequest] = []
    for battle in battles:
        if battle.article_path is not None:
            raw = _read_text(battle.article_path)
            if raw:
                for passage in build_passages(
                    battle.name,
                    raw,
                    source_title=battle.name,
                    max_chars=max_passage_chars,
                    is_html=battle.article_is_html,
                ):
                    requests.append(
                        LLMRequest(
                            system=system_prompt,
                            user=_render(passage),
                            json_schema=schema,
                            schema_name="extraction",
                            max_tokens=max_tokens,
                            temperature=temperature,
                            metadata=_metadata(
                                battle.slug, str(battle.article_path), passage
                            ),
                        )
                    )

        for path in battle.citation_paths:
            for record in _citation_records(path):
                if record.get("skipped"):
                    continue
                body = str(record.get("content") or record.get("text") or "")
                if not body.strip():
                    continue
                source_ref = str(record.get("url") or path)
                for passage in build_passages(
                    battle.name,
                    body,
                    source_type="web_secondary",
                    source_title=str(
                        record.get("title") or record.get("url") or path.name
                    ),
                    max_chars=max_passage_chars,
                ):
                    requests.append(
                        LLMRequest(
                            system=system_prompt,
                            user=_render(passage),
                            json_schema=schema,
                            schema_name="extraction",
                            max_tokens=max_tokens,
                            temperature=temperature,
                            metadata=_metadata(battle.slug, source_ref, passage),
                        )
                    )

    return requests


def _resolve_requests(
    spec: dict[str, Any],
    processed_root: Path,
) -> list[LLMRequest]:
    """Build resolve-stage LLM requests (disambiguation only).

    Most resolution is deterministic; these are the ambiguous-group
    requests that would go to the LLM.
    """
    logger.info(
        "resolve_requests_not_exportable",
        reason=(
            "resolve builds LLM requests dynamically from match results, "
            "so they cannot be pre-computed without running the deterministic "
            "matching first. Run the stage with candidate_source=local or "
            "candidate_source=none to identify which groups need the LLM, "
            "then export those."
        ),
    )
    return []


def _classify_requests(
    spec: dict[str, Any],
    processed_root: Path,
) -> list[LLMRequest]:
    """Build classify-stage command-role requests for every pending side.

    Delegates to ``pipeline.stages.classify.build_pending_role_requests``,
    the same function the live stage's offline and api LLM steps call, so a
    request built here and one built by a live run are always identical and
    hash the same way. Without that shared path this exporter could drift
    from the stage silently, and every response processed against the drift
    would come back a cache miss.

    Args:
        spec: The loaded ``agents/classify.yaml``.
        processed_root: Unused here -- classify reads from the database, not
            from ``data/processed``; kept for a uniform dispatch signature
            with ``_extract_requests``/``_resolve_requests``.

    Returns:
        One request per side still needing an LLM answer. An empty list,
        with a warning logged, when no database is reachable: unlike
        extract or resolve, classify has nothing to build a request from
        without one.
    """
    del processed_root  # classify's inputs are all in the database

    from pipeline.db import get_connection
    from pipeline.stages.classify import build_pending_role_requests

    params = spec.get("params", {}) or {}
    raw_root = Path(str(params.get("raw_root", "data/raw")))
    excerpt_max_chars = int(params.get("excerpt_max_chars", 6000))

    try:
        with get_connection() as conn:
            return [
                request
                for _side, request in build_pending_role_requests(
                    conn, raw_root, spec, excerpt_max_chars=excerpt_max_chars
                )
            ]
    except Exception as e:
        logger.warning("classify_requests_needs_database", error=str(e))
        return []


def cmd_status(args: argparse.Namespace) -> None:
    """Show cache status for a stage's requests."""
    load_env()
    spec = load_agent_spec(args.stage)
    config = llm_params(spec)
    provider = config["provider"]
    model = config["model"]

    requests = _build_requests_from_spec(spec, config)
    if not requests:
        print(f"No requests found for stage {args.stage!r}.")
        return

    eng = get_engine()
    with eng.connect() as conn:
        cached = 0
        pending = 0
        for req in requests:
            digest = compute_hash(req, provider, model)
            if find_completed_call(conn, digest) is not None:
                cached += 1
            else:
                pending += 1

    print(f"Stage: {args.stage}")
    print(f"Provider: {provider}, Model: {model}")
    print(f"Total requests: {len(requests)}")
    print(f"Cached: {cached}")
    print(f"Pending: {pending}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export uncached requests for a stage."""
    load_env()
    spec = load_agent_spec(args.stage)
    config = llm_params(spec)
    provider = config["provider"]
    model = config["model"]

    requests = _build_requests_from_spec(spec, config)
    if not requests:
        print(f"No requests found for stage {args.stage!r}.")
        return

    conn = None
    eng = None
    try:
        eng = get_engine()
        conn = eng.connect()
    except Exception as e:
        logger.warning("no_database_connection_all_treated_as_pending", error=str(e))

    output_dir = Path(args.output) if args.output else None
    result = export_pending(
        stage=args.stage,
        provider=provider,
        model=model,
        requests=requests,
        conn=conn,
        output_dir=output_dir,
    )

    if conn is not None:
        conn.close()
    if eng is not None:
        eng.dispose()

    print(f"Exported: {result.exported}")
    print(f"Cached (skipped): {result.cached}")
    if result.exported:
        print(f"File: {result.path}")
        print()
        print("Process these requests (/process-llm-batch), then import the responses:")
        print("  python -m scripts.llm_offline import --dir data/llm_responses/<stem>/")


def _response_files(file: str | None, directory: str | None) -> list[Path]:
    """Resolve ``--file`` / ``--dir`` to response files.

    Args:
        file: A single response file, or None.
        directory: A directory whose ``*.jsonl`` files are all used, or None.

    Returns:
        The files, sorted for a directory.
    """
    if file:
        return [Path(file)]
    if directory:
        return sorted(Path(directory).glob("*.jsonl"))
    return []


def cmd_import(args: argparse.Namespace) -> int:
    """Import completed responses into llm_calls."""
    load_env()
    files = _response_files(args.file, args.dir)
    missing = [f for f in files if not f.exists()]
    if not files or missing:
        target = ", ".join(str(m) for m in missing) or str(args.dir)
        print(f"No response files found: {target}", file=sys.stderr)
        return 1

    index = index_requests([Path(p) for p in args.requests] if args.requests else None)
    imported = skipped = errors = 0
    eng = get_engine()
    with eng.connect() as conn:
        for path in files:
            result = import_responses(path, conn, model=args.model, index=index)
            print(
                f"{path}: imported {result.imported}, skipped {result.skipped}, "
                f"errors {result.errors}"
            )
            imported += result.imported
            skipped += result.skipped
            errors += result.errors
        conn.commit()

    print(f"Imported: {imported}")
    print(f"Skipped (already cached): {skipped}")
    print(f"Errors: {errors}")
    return 0


@dataclass(frozen=True)
class SplitSummary:
    """What a split wrote."""

    chunk_dir: Path
    chunks: list[dict[str, Any]]
    requests: int


def run_split(
    request_file: Path,
    *,
    chunk_size: int,
    skip_answered: bool,
    response_dir: Path | None = None,
    conn: Any = None,
) -> SplitSummary:
    """Write a request file's compact chunks.

    Args:
        request_file: The batch file to split.
        chunk_size: Maximum requests per chunk.
        skip_answered: Drop hashes already completed in ``llm_calls`` (when
            ``conn`` is given) or present in the response directory.
        response_dir: Where responses for this file live; defaults to
            ``data/llm_responses/<request file stem>/``.
        conn: Database connection for the cache check, or None.

    Returns:
        The chunk directory and chunks written.
    """
    stem = request_file.stem
    skip: set[str] = set()
    if skip_answered:
        skip |= answered_hashes(response_dir or RESPONSE_DIR / stem)
        if conn is not None:
            skip |= {
                digest
                for digest in index_requests([request_file])
                if find_completed_call(conn, digest) is not None
            }

    chunks = build_chunks(request_file, chunk_size=chunk_size, skip=skip)

    chunk_dir = request_file.parent / "chunks" / stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    # A re-split must not leave chunks from an earlier, larger split behind.
    for stale in chunk_dir.glob("chunk_*.json"):
        stale.unlink()
    for number, chunk in enumerate(chunks, 1):
        path = chunk_dir / f"chunk_{number:03d}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")), "utf-8")

    return SplitSummary(
        chunk_dir=chunk_dir,
        chunks=chunks,
        requests=sum(len(c["requests"]) for c in chunks),
    )


def cmd_split(args: argparse.Namespace) -> int:
    """Split a request file into compact chunks for subagents."""
    load_env()
    request_file = Path(args.file)
    if not request_file.exists():
        print(f"File not found: {request_file}", file=sys.stderr)
        return 1

    conn = None
    eng = None
    if args.skip_answered:
        try:
            eng = get_engine()
            conn = eng.connect()
        except Exception as e:
            logger.warning("no_database_connection_skipping_only_by_response_files", error=str(e))

    try:
        summary = run_split(
            request_file,
            chunk_size=args.chunk_size,
            skip_answered=args.skip_answered,
            conn=conn,
        )
    finally:
        if conn is not None:
            conn.close()
        if eng is not None:
            eng.dispose()

    print(f"Chunk directory: {summary.chunk_dir}")
    print(f"Chunks: {len(summary.chunks)}")
    print(f"Requests: {summary.requests}")
    for number, chunk in enumerate(summary.chunks, 1):
        print(
            f"  chunk_{number:03d}.json  agent_model={chunk['agent_model']}  "
            f"requests={len(chunk['requests'])}"
        )
    return 0


def run_validate(
    responses: Path,
    *,
    request_files: Sequence[Path] | None = None,
    request_dir: Path | None = None,
) -> int:
    """Validate response files against their requests and print a report.

    Args:
        responses: A response file or a directory of ``*.jsonl`` files.
        request_files: Explicit request files; by default every ``*.jsonl``
            in ``request_dir``.
        request_dir: Request directory, defaulting to ``data/llm_requests/``.

    Returns:
        0 when every line is valid, 1 when any line failed or no response
        file was found.
    """
    directory = request_dir or REQUEST_DIR
    files = sorted(responses.glob("*.jsonl")) if responses.is_dir() else [responses]
    files = [f for f in files if f.exists()]
    if not files:
        print(f"No response files found at {responses}", file=sys.stderr)
        return 1

    index = index_requests(request_files, directory)

    stem = responses.name if responses.is_dir() else responses.parent.name
    matching = directory / f"{stem}.jsonl"
    expected = set(index_requests([matching])) if matching.exists() else set()

    answered: set[str] = set()
    failed = False
    for path in files:
        report = validate_responses(path, index=index)
        answered |= report.answered
        print(f"{path}: ok {report.ok}, errors {len(report.errors)}")
        for prefix, reason in report.errors:
            print(f"  {prefix}: {reason}")
            failed = True

    if expected:
        unanswered = sorted(expected - answered)
        print(f"Requests in {matching.name} with no valid response yet: {len(unanswered)}")
        for digest in unanswered[:20]:
            print(f"  {digest[:12]}")
        if len(unanswered) > 20:
            print(f"  ... and {len(unanswered) - 20} more")

    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate responses without writing to the database."""
    requests = [Path(p) for p in args.requests] if args.requests else None
    return run_validate(Path(args.responses), request_files=requests)


def cmd_list(args: argparse.Namespace) -> None:
    """List exported request files."""
    directory = Path(args.dir) if args.dir else REQUEST_DIR
    if not directory.exists():
        print(f"No request directory at {directory}")
        return

    files = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("No exported request files found.")
        return

    for f in files[:20]:
        count = sum(1 for _ in f.open("r", encoding="utf-8") if _.strip())
        print(f"  {f.name}  ({count} requests)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline LLM request export and response import.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show cache status for a stage")
    p_status.add_argument("--stage", required=True)
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="Export uncached requests")
    p_export.add_argument("--stage", required=True)
    p_export.add_argument("--output", help="Output directory override")
    p_export.set_defaults(func=cmd_export)

    p_split = sub.add_parser("split", help="Split a request file into subagent chunks")
    p_split.add_argument("--file", required=True, help="Request JSONL file")
    p_split.add_argument("--chunk-size", type=int, default=15, help="Requests per chunk")
    p_split.add_argument(
        "--skip-answered",
        action="store_true",
        help="Drop requests already in llm_calls or in data/llm_responses/<stem>/",
    )
    p_split.set_defaults(func=cmd_split)

    p_validate = sub.add_parser("validate", help="Validate responses, no database writes")
    p_validate.add_argument("--responses", required=True, help="Response file or directory")
    p_validate.add_argument("--requests", nargs="*", help="Request files (default: all)")
    p_validate.set_defaults(func=cmd_validate)

    p_import = sub.add_parser("import", help="Import completed responses")
    source = p_import.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="Response JSONL file")
    source.add_argument("--dir", help="Directory of response *.jsonl files")
    p_import.add_argument("--requests", nargs="*", help="Request files (default: all)")
    p_import.add_argument("--model", help="Model name for lines without model_used")
    p_import.set_defaults(func=cmd_import)

    p_list = sub.add_parser("list", help="List exported request files")
    p_list.add_argument("--dir", help="Directory to list")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    code = args.func(args)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
