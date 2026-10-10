"""Tests for the offline LLM provider and the export/import/split/validate workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pipeline.llm.base import (
    CallStatus,
    LLMConfigError,
    LLMRequest,
)
from pipeline.llm.base import (
    request_hash as compute_hash,
)
from pipeline.llm.factory import build_provider, llm_params
from pipeline.llm.offline import (
    OfflineProvider,
    agent_model_for,
    build_chunks,
    export_pending,
    import_responses,
    index_requests,
    validate_responses,
)
from pipeline.llm.service import LLMService
from scripts.llm_offline import run_split, run_validate

PROVIDER = "anthropic"
MODEL = "claude-haiku-5-5"
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def _make_request(user: str = "test passage", **kwargs: Any) -> LLMRequest:
    return LLMRequest(
        system="You are a test extractor.",
        user=user,
        json_schema=SCHEMA,
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


def _inserted(conn: MagicMock) -> list[dict[str, Any]]:
    """Parameters of every INSERT the stub connection saw."""
    return [
        call.args[1]
        for call in conn.execute.call_args_list
        if len(call.args) > 1 and call.args[1] and "stage" in call.args[1]
    ]


def _write_requests(directory: Path, stem: str, requests: list[LLMRequest]) -> Path:
    """Export requests to ``<directory>/<stem>.jsonl`` and return the path."""
    result = export_pending("extract", PROVIDER, MODEL, requests, output_dir=directory)
    target = directory / f"{stem}.jsonl"
    result.path.rename(target)
    return target


def _hashes(path: Path) -> list[str]:
    return list(index_requests([path]))


def _answer(digest: str, name: str = "x", **extra: Any) -> str:
    return json.dumps({"request_hash": digest, "data": {"name": name}, **extra})


class TestAgentModel:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-haiku-5-5", "haiku"),
            ("claude-sonnet-5-5", "sonnet"),
            ("claude-opus-5", "opus"),
            ("claude-fable-5", "sonnet"),
            ("", "sonnet"),
        ],
    )
    def test_maps_model_to_agent_tier(self, model: str, expected: str) -> None:
        assert agent_model_for(model) == expected


class TestOfflineProvider:
    def test_hash_equals_the_api_path_hash(self) -> None:
        request = _make_request("same prompt", max_tokens=2048)
        provider = OfflineProvider(PROVIDER, MODEL)

        response = provider.complete(request)

        assert response.request_hash == compute_hash(request, PROVIDER, MODEL)

    def test_returns_deferred_with_zero_usage(self) -> None:
        response = OfflineProvider(PROVIDER, MODEL).complete(_make_request())

        assert response.status is CallStatus.DEFERRED
        assert response.ok is False
        assert response.usage.total_tokens == 0
        assert response.cost_usd == 0.0
        assert response.status.needs_review is True

    def test_dedupes_identical_requests(self, tmp_path: Path) -> None:
        provider = OfflineProvider(PROVIDER, MODEL, output_dir=tmp_path)
        provider.complete(_make_request("a"))
        provider.complete(_make_request("a"))
        provider.complete(_make_request("b"))

        assert provider.pending == 2

    def test_flush_writes_the_export_line_format_plus_agent_model(self, tmp_path: Path) -> None:
        request = _make_request("flush me", metadata={"battle_slug": "x"})
        provider = OfflineProvider(PROVIDER, MODEL, output_dir=tmp_path)
        provider.complete(request)

        path = provider.flush("extract")

        assert path is not None
        assert path.parent == tmp_path
        assert path.name.startswith("extract_") and path.suffix == ".jsonl"
        line = json.loads(path.read_text(encoding="utf-8").strip())
        exported = export_pending(
            "extract", PROVIDER, MODEL, [request], output_dir=tmp_path / "ref"
        )
        reference = json.loads(exported.path.read_text(encoding="utf-8").strip())
        assert line == reference
        assert line["agent_model"] == "haiku"
        assert line["request_hash"] == compute_hash(request, PROVIDER, MODEL)
        assert provider.pending == 0

    def test_flush_returns_none_when_nothing_is_pending(self, tmp_path: Path) -> None:
        provider = OfflineProvider(PROVIDER, MODEL, output_dir=tmp_path)

        assert provider.flush("extract") is None
        assert list(tmp_path.iterdir()) == []

    def test_two_flushes_in_one_second_do_not_overwrite(self, tmp_path: Path) -> None:
        provider = OfflineProvider(PROVIDER, MODEL, output_dir=tmp_path)
        provider.complete(_make_request("one"))
        first = provider.flush("extract")
        provider.complete(_make_request("two"))
        second = provider.flush("extract")

        assert first != second
        assert len(list(tmp_path.glob("*.jsonl"))) == 2


class TestFactoryMode:
    def _spec(self, **params: Any) -> dict[str, Any]:
        return {
            "stage": "extract",
            "params": {"llm_provider": PROVIDER, "llm_model": MODEL, **params},
        }

    def test_defaults_to_offline_without_needing_an_sdk_or_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        provider = build_provider(self._spec())

        assert isinstance(provider, OfflineProvider)
        assert (provider.name, provider.model) == (PROVIDER, MODEL)
        assert llm_params(self._spec())["mode"] == "offline"

    def test_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(LLMConfigError, match="llm_mode"):
            build_provider(self._spec(llm_mode="batch"))

    def test_offline_still_requires_a_model(self) -> None:
        with pytest.raises(LLMConfigError, match="no llm_model"):
            build_provider({"stage": "extract", "params": {"llm_provider": PROVIDER}})

    def test_api_mode_builds_the_anthropic_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")

        provider = build_provider(self._spec(llm_mode="api"))

        assert not isinstance(provider, OfflineProvider)
        assert (provider.name, provider.model) == (PROVIDER, MODEL)


class TestServiceDeferred:
    def test_deferred_is_not_logged_or_counted_as_a_failure(self, tmp_path: Path) -> None:
        conn = _stub_conn()
        provider = OfflineProvider(PROVIDER, MODEL, output_dir=tmp_path)
        service = LLMService("extract", provider, db_conn=conn)

        response = service.complete(system="s", user="u", json_schema=SCHEMA)

        assert response.status is CallStatus.DEFERRED
        assert service.usage.deferred == 1
        assert service.usage.failures == 0
        assert service.usage.as_dict()["deferred"] == 1
        assert _inserted(conn) == []

    def test_flush_pending_writes_the_batch_file(self, tmp_path: Path) -> None:
        provider = OfflineProvider(PROVIDER, MODEL, output_dir=tmp_path)
        service = LLMService("extract", provider, db_conn=_stub_conn())
        service.complete(system="s", user="u", json_schema=SCHEMA)

        path = service.flush_pending()

        assert path is not None and path.exists()
        assert service.flush_pending() is None

    def test_flush_pending_is_a_no_op_for_an_api_style_provider(self) -> None:
        provider = MagicMock(spec=["name", "model", "complete"])
        service = LLMService("extract", provider, db_conn=_stub_conn())

        assert service.flush_pending() is None


class TestExportPending:
    def test_exports_uncached_requests(self, tmp_path: Path) -> None:
        requests = [_make_request("passage 1"), _make_request("passage 2")]
        result = export_pending(
            stage="extract",
            provider=PROVIDER,
            model=MODEL,
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
        assert first["provider"] == PROVIDER
        assert first["agent_model"] == "haiku"
        assert first["system"] == "You are a test extractor."

    def test_skips_cached_requests(self, tmp_path: Path) -> None:
        req = _make_request("cached passage")
        digest = compute_hash(req, PROVIDER, MODEL)
        conn = _stub_conn(cached_hashes={digest})

        result = export_pending(
            stage="extract",
            provider=PROVIDER,
            model=MODEL,
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
        digest = compute_hash(cached_req, PROVIDER, MODEL)
        conn = _stub_conn(cached_hashes={digest})

        result = export_pending(
            stage="extract",
            provider=PROVIDER,
            model=MODEL,
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
            provider=PROVIDER,
            model=MODEL,
            requests=[req],
            conn=None,
            output_dir=tmp_path / "run1",
        )
        result2 = export_pending(
            stage="extract",
            provider=PROVIDER,
            model=MODEL,
            requests=[req],
            conn=None,
            output_dir=tmp_path / "run2",
        )

        line1 = json.loads(result1.path.read_text(encoding="utf-8").strip())
        line2 = json.loads(result2.path.read_text(encoding="utf-8").strip())
        assert line1["request_hash"] == line2["request_hash"]


class TestImportResponses:
    @pytest.fixture
    def requests_file(self, tmp_path: Path) -> Path:
        return _write_requests(
            tmp_path / "requests", "extract_t1", [_make_request("p1"), _make_request("p2")]
        )

    def test_imports_valid_responses_with_claude_code_provider(
        self, tmp_path: Path, requests_file: Path
    ) -> None:
        digest = _hashes(requests_file)[0]
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(_answer(digest, "Napoleon") + "\n", encoding="utf-8")
        conn = _stub_conn()

        result = import_responses(response_file, conn, request_files=[requests_file])

        assert (result.imported, result.errors) == (1, 0)
        (row,) = _inserted(conn)
        assert row["provider"] == "claude_code"
        assert row["model"] == "claude-code-subagent"
        assert row["input_hash"] == digest
        assert row["status"] == "ok"
        assert json.loads(row["response_json"]) == {"name": "Napoleon"}

    def test_model_comes_from_the_line_then_the_cli_then_the_default(
        self, tmp_path: Path, requests_file: Path
    ) -> None:
        first, second = _hashes(requests_file)
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(
            _answer(first, model_used="claude-haiku-5-5") + "\n" + _answer(second) + "\n",
            encoding="utf-8",
        )
        conn = _stub_conn()

        import_responses(response_file, conn, model="cli-model", request_files=[requests_file])

        models = {row["input_hash"]: row["model"] for row in _inserted(conn)}
        assert models == {first: "claude-haiku-5-5", second: "cli-model"}

    def test_schema_invalid_data_is_an_error_and_not_imported(
        self, tmp_path: Path, requests_file: Path
    ) -> None:
        digest = _hashes(requests_file)[0]
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(
            json.dumps({"request_hash": digest, "data": {"name": 7}}) + "\n", encoding="utf-8"
        )
        conn = _stub_conn()

        result = import_responses(response_file, conn, request_files=[requests_file])

        assert (result.imported, result.errors) == (0, 1)
        assert _inserted(conn) == []

    def test_unknown_hash_is_an_error(self, tmp_path: Path, requests_file: Path) -> None:
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(_answer("f" * 64) + "\n", encoding="utf-8")
        conn = _stub_conn()

        result = import_responses(response_file, conn, request_files=[requests_file])

        assert (result.imported, result.errors) == (0, 1)

    def test_skips_already_cached(self, tmp_path: Path, requests_file: Path) -> None:
        digest = _hashes(requests_file)[0]
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(_answer(digest) + "\n", encoding="utf-8")

        result = import_responses(
            response_file, _stub_conn({digest}), request_files=[requests_file]
        )

        assert (result.imported, result.skipped) == (0, 1)

    def test_reports_malformed_lines(self, tmp_path: Path, requests_file: Path) -> None:
        digest = _hashes(requests_file)[0]
        response_file = tmp_path / "responses.jsonl"
        response_file.write_text(
            "not valid json\n" + json.dumps({"request_hash": "x"}) + "\n" + _answer(digest) + "\n",
            encoding="utf-8",
        )

        result = import_responses(response_file, _stub_conn(), request_files=[requests_file])

        assert result.errors == 2
        assert result.imported == 1

    def test_indexes_every_request_file_under_the_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "requests"
        file_a = _write_requests(directory, "extract_a", [_make_request("a")])
        file_b = _write_requests(directory, "classify_b", [_make_request("b")])
        monkeypatch.setattr("pipeline.llm.offline.REQUEST_DIR", directory)

        index = index_requests()

        assert set(index) == set(_hashes(file_a)) | set(_hashes(file_b))


class TestSplit:
    def _request_file(self, directory: Path) -> Path:
        reqs = [_make_request(f"p{i}") for i in range(5)]
        other = LLMRequest(
            system="A different prompt.", user="q", json_schema=SCHEMA, schema_name="test"
        )
        return _write_requests(directory, "extract_t1", [*reqs, other])

    def test_groups_by_prompt_and_respects_chunk_size(self, tmp_path: Path) -> None:
        request_file = self._request_file(tmp_path)

        chunks = build_chunks(request_file, chunk_size=2)

        sizes = [len(c["requests"]) for c in chunks]
        assert sizes == [2, 2, 1, 1]
        systems = [c["system"] for c in chunks]
        assert systems == ["You are a test extractor."] * 3 + ["A different prompt."]
        assert set(chunks[0]) == {
            "stage",
            "agent_model",
            "system",
            "json_schema",
            "schema_name",
            "requests",
        }
        assert set(chunks[0]["requests"][0]) == {"request_hash", "user"}
        assert chunks[0]["agent_model"] == "haiku"

    def test_rejects_a_non_positive_chunk_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            build_chunks(self._request_file(tmp_path), chunk_size=0)

    def test_writes_compact_numbered_chunk_files(self, tmp_path: Path) -> None:
        request_file = self._request_file(tmp_path)

        summary = run_split(
            request_file, chunk_size=3, skip_answered=False, response_dir=tmp_path / "none"
        )

        assert summary.chunk_dir == tmp_path / "chunks" / "extract_t1"
        files = sorted(summary.chunk_dir.glob("chunk_*.json"))
        assert [f.name for f in files] == ["chunk_001.json", "chunk_002.json", "chunk_003.json"]
        text = files[0].read_text(encoding="utf-8")
        assert "\n" not in text
        assert json.loads(text)["requests"]
        assert summary.requests == 6

    def test_skip_answered_uses_the_response_directory(self, tmp_path: Path) -> None:
        request_file = self._request_file(tmp_path)
        hashes = _hashes(request_file)
        response_dir = tmp_path / "responses"
        response_dir.mkdir()
        (response_dir / "chunk_001.jsonl").write_text(
            "\n".join(_answer(h) for h in hashes[:4]) + "\n", encoding="utf-8"
        )

        summary = run_split(
            request_file, chunk_size=15, skip_answered=True, response_dir=response_dir
        )

        assert summary.requests == 2

    def test_skip_answered_uses_the_llm_calls_cache(self, tmp_path: Path) -> None:
        request_file = self._request_file(tmp_path)
        hashes = _hashes(request_file)

        summary = run_split(
            request_file,
            chunk_size=15,
            skip_answered=True,
            response_dir=tmp_path / "none",
            conn=_stub_conn(set(hashes[:3])),
        )

        assert summary.requests == 3

    def test_resplit_removes_stale_chunks(self, tmp_path: Path) -> None:
        request_file = self._request_file(tmp_path)
        run_split(request_file, chunk_size=1, skip_answered=False, response_dir=tmp_path / "n")

        summary = run_split(
            request_file, chunk_size=15, skip_answered=False, response_dir=tmp_path / "n"
        )

        assert len(list(summary.chunk_dir.glob("chunk_*.json"))) == len(summary.chunks)


class TestValidate:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path, list[str]]:
        directory = tmp_path / "requests"
        request_file = _write_requests(
            directory, "extract_t1", [_make_request("a"), _make_request("b")]
        )
        responses = tmp_path / "responses" / "extract_t1"
        responses.mkdir(parents=True)
        return directory, responses, _hashes(request_file)

    def test_clean_responses_exit_zero(self, tmp_path: Path) -> None:
        directory, responses, hashes = self._setup(tmp_path)
        (responses / "chunk_001.jsonl").write_text(_answer(hashes[0]) + "\n", encoding="utf-8")

        assert run_validate(responses, request_dir=directory) == 0

    def test_invalid_and_unknown_lines_exit_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory, responses, hashes = self._setup(tmp_path)
        (responses / "chunk_001.jsonl").write_text(
            json.dumps({"request_hash": hashes[0], "data": {"name": 3}})
            + "\n"
            + _answer("e" * 64)
            + "\n",
            encoding="utf-8",
        )

        code = run_validate(responses, request_dir=directory)

        out = capsys.readouterr().out
        assert code == 1
        assert hashes[0][:12] in out
        assert "e" * 12 in out
        assert "no valid response yet: 2" in out

    def test_reports_requests_with_no_response(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory, responses, hashes = self._setup(tmp_path)
        (responses / "chunk_001.jsonl").write_text(_answer(hashes[0]) + "\n", encoding="utf-8")

        run_validate(responses, request_dir=directory)

        out = capsys.readouterr().out
        assert "no valid response yet: 1" in out
        assert hashes[1][:12] in out

    def test_validate_responses_does_not_need_a_connection(self, tmp_path: Path) -> None:
        directory, responses, hashes = self._setup(tmp_path)
        path = responses / "chunk_001.jsonl"
        path.write_text(_answer(hashes[0]) + "\n", encoding="utf-8")

        report = validate_responses(path, index=index_requests(None, directory))

        assert report.ok == 1 and report.errors == []
        assert report.answered == {hashes[0]}

    def test_missing_responses_path_exits_one(self, tmp_path: Path) -> None:
        assert run_validate(tmp_path / "nothing.jsonl", request_dir=tmp_path) == 1
