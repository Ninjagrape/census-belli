"""Tests for the offline LLM export/import workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pipeline.llm.base import (
    LLMRequest,
)
from pipeline.llm.base import (
    request_hash as compute_hash,
)
from pipeline.llm.offline import export_pending, import_responses


def _make_request(user: str = "test passage", **kwargs: Any) -> LLMRequest:
    return LLMRequest(
        system="You are a test extractor.",
        user=user,
        json_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        schema_name="test",
        **kwargs,
    )


def _stub_conn(cached_hashes: set[str] | None = None) -> MagicMock:
    """A connection stub whose cache lookup honours a set of known hashes.

    Distinguishes SELECT (cache lookup, params has only ``input_hash``)
    from INSERT (log_call, params has ``stage``, ``provider``, etc.) by
    checking whether the params dict contains the ``stage`` key.
    """
    conn = MagicMock()
    cached = cached_hashes or set()

    def fake_execute(statement: Any, params: Any = None) -> Any:
        result = MagicMock()
        if params and "input_hash" in params and "stage" not in params:
            digest = params["input_hash"]
            if digest in cached:
                row = MagicMock()
                row.__getitem__ = lambda self, i: {"name": "cached"} if i == 0 else None
                result.fetchone.return_value = row
            else:
                result.fetchone.return_value = None
        else:
            row = MagicMock()
            row.__getitem__ = lambda self, i: 1 if i == 0 else None
            result.fetchone.return_value = row
        return result

    conn.execute.side_effect = fake_execute
    return conn


class TestExportPending:
    def test_exports_uncached_requests(self, tmp_path: Path) -> None:
        requests = [_make_request("passage 1"), _make_request("passage 2")]
        result = export_pending(
            stage="extract",
            provider="gemini",
            model="gemini-3.8-flash",
            requests=requests,
            conn=None,
            output_dir=tmp_path,
        )

        assert result.exported == 2
        assert result.cached == 0
        assert result.path.exists()

        lines = result.path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert "request_hash" in first
        assert first["stage"] == "extract"
        assert first["provider"] == "gemini"
        assert first["system"] == "You are a test extractor."

    def test_skips_cached_requests(self, tmp_path: Path) -> None:
        req = _make_request("cached passage")
        digest = compute_hash(req, "gemini", "gemini-3.8-flash")
        conn = _stub_conn(cached_hashes={digest})

        result = export_pending(
            stage="extract",
            provider="gemini",
            model="gemini-3.8-flash",
            requests=[req],
            conn=conn,
            output_dir=tmp_path,
        )

        assert result.exported == 0
        assert result.cached == 1
        assert not result.path.exists()

    def test_mixed_cached_and_pending(self, tmp_path: Path) -> None:
        cached_req = _make_request("cached")
        pending_req = _make_request("pending")
        digest = compute_hash(cached_req, "gemini", "gemini-3.8-flash")
        conn = _stub_conn(cached_hashes={digest})

        result = export_pending(
            stage="extract",
            provider="gemini",
            model="gemini-3.8-flash",
            requests=[cached_req, pending_req],
            conn=conn,
            output_dir=tmp_path,
        )

        assert result.exported == 1
        assert result.cached == 1

    def test_hash_is_deterministic(self, tmp_path: Path) -> None:
        req = _make_request("deterministic")
        result1 = export_pending(
            stage="extract",
            provider="gemini",
            model="gemini-3.8-flash",
            requests=[req],
            conn=None,
            output_dir=tmp_path / "run1",
        )
        result2 = export_pending(
            stage="extract",
            provider="gemini",
            model="gemini-3.8-flash",
            requests=[req],
            conn=None,
            output_dir=tmp_path / "run2",
        )

        line1 = json.loads(result1.path.read_text(encoding="utf-8").strip())
        line2 = json.loads(result2.path.read_text(encoding="utf-8").strip())
        assert line1["request_hash"] == line2["request_hash"]


class TestImportResponses:
    def test_imports_valid_responses(self, tmp_path: Path) -> None:
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(
            json.dumps({
                "request_hash": "abc123",
                "data": {"name": "Napoleon"},
                "stage": "extract",
            })
            + "\n",
            encoding="utf-8",
        )

        conn = _stub_conn()
        result = import_responses(response_file, conn)

        assert result.imported == 1
        assert result.errors == 0

    def test_skips_already_cached(self, tmp_path: Path) -> None:
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(
            json.dumps({
                "request_hash": "already_there",
                "data": {"name": "Wellington"},
                "stage": "extract",
            })
            + "\n",
            encoding="utf-8",
        )

        conn = _stub_conn(cached_hashes={"already_there"})
        result = import_responses(response_file, conn)

        assert result.imported == 0
        assert result.skipped == 1

    def test_reports_malformed_lines(self, tmp_path: Path) -> None:
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(
            "not valid json\n"
            + json.dumps({"request_hash": "x"}) + "\n"
            + json.dumps({"request_hash": "y", "data": {"ok": True}, "stage": "test"}) + "\n",
            encoding="utf-8",
        )

        conn = _stub_conn()
        result = import_responses(response_file, conn)

        assert result.errors == 2
        assert result.imported == 1

    def test_roundtrip_export_then_import(self, tmp_path: Path) -> None:
        """An exported request's hash survives the import path."""
        req = _make_request("roundtrip test")
        export_result = export_pending(
            stage="extract",
            provider="gemini",
            model="gemini-3.8-flash",
            requests=[req],
            conn=None,
            output_dir=tmp_path / "requests",
        )

        exported = json.loads(
            export_result.path.read_text(encoding="utf-8").strip()
        )
        digest = exported["request_hash"]

        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(
            json.dumps({
                "request_hash": digest,
                "data": {"name": "test result"},
                "stage": "extract",
            })
            + "\n",
            encoding="utf-8",
        )

        conn = _stub_conn()
        import_result = import_responses(response_file, conn)
        assert import_result.imported == 1
