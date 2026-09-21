"""
CLI for the offline LLM processing workflow.

Usage::

    # Show what is pending for a stage
    python -m scripts.llm_offline status --stage extract

    # Export uncached requests to a JSONL file
    python -m scripts.llm_offline export --stage extract

    # Import completed responses
    python -m scripts.llm_offline import --file data/llm_responses/extract_results.jsonl
"""

from __future__ import annotations

import argparse
import sys
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
    export_pending,
    import_responses,
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
    """Build classify-stage requests from resolved commanders."""
    logger.info(
        "classify_stage_not_yet_implemented",
        reason="classify is not yet built; export will work once it is",
    )
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
        print("Process these requests, then import the responses:")
        print("  python -m scripts.llm_offline import --file <response_file>.jsonl")


def cmd_import(args: argparse.Namespace) -> None:
    """Import completed responses into llm_calls."""
    load_env()
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    eng = get_engine()
    with eng.connect() as conn:
        result = import_responses(
            path,
            conn,
            provider=args.provider or "offline",
            model=args.model or "claude-pro-subscription",
        )
        conn.commit()

    print(f"Imported: {result.imported}")
    print(f"Skipped (already cached): {result.skipped}")
    print(f"Errors: {result.errors}")


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

    p_import = sub.add_parser("import", help="Import completed responses")
    p_import.add_argument("--file", required=True, help="Response JSONL file")
    p_import.add_argument("--provider", help="Provider name for audit trail")
    p_import.add_argument("--model", help="Model name for audit trail")
    p_import.set_defaults(func=cmd_import)

    p_list = sub.add_parser("list", help="List exported request files")
    p_list.add_argument("--dir", help="Directory to list")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
