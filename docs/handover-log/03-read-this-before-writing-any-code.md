# Handover log §3. Read this before writing any code

Part of the session log; the index is `docs/handover-log/README.md`.

### 3.1 `pipeline/llm` is a package, not a module

There is **no `pipeline/llm.py`** and there must not be — a module and a package of that name cannot coexist. `.claude/commands/build-all.md` Task 1.4 says "implement pipeline/llm.py"; **that instruction is stale.** The package already does everything it asks for and more.

Public API:

```python
from pipeline.llm import LLMService

service = LLMService.from_spec(spec, db_conn=conn)
result = service.complete(
    system=spec["prompt"]["system"],
    user=article_text,
    json_schema=schema,
    metadata={"battle_id": battle_id},
)
if not result.ok:
    continue          # already logged and flagged for review
upsert(result.data)
service.log_summary()
```

It handles retries, JSON-schema validation, cost accounting, and logs every call to `llm_calls`. That table doubles as a **resume cache keyed on request hash**, so a re-run must not re-pay for work already done. Read `pipeline/llm/service.py` before using it.

Providers are routed per stage in the spec, not in code:

| Stage | `params.llm_provider` |
|---|---|
| extract | gemini |
| resolve | gemini |
| classify | anthropic |

Gemini was reached successfully on 2026-09-19 with the key already in the environment, whose prefix is `AQ.A...` -- **not** the `AIza...` AI Studio form this file previously insisted on. A real call to `gemini-3.8-flash` returned a schema-valid answer, cost $0.0014 and wrote its `llm_calls` row. Do not assume a key is wrong because of its prefix; make one call and read the error.

### 3.2 The quality gates are real now, and they bite

`pipeline/quality.py` executes the `quality_checks` SQL in each agent spec. Until recently `_run_single_check` was a stub returning `passed=True` unconditionally, so **every gate in every spec was inert**.

Two latent spec bugs surfaced the moment they started executing. Expect more as later stages run for the first time. When a gate fails, check whether the *spec* is wrong before assuming the *code* is.

Checks declaring a `method:` instead of SQL report as **not implemented** — deliberately failing rather than passing, so the gap stays visible. Handlers now exist for `diagnostics_json`, `imputation_gate` and `model_gate` (`METHOD_HANDLERS` in `pipeline/quality.py`); impute's old `per_round_check`/`range_check`/`statistical_test` were replaced by `imputation_gate`, and nothing uses `posterior_check` any more. Three are outstanding, all evaluate's:

| Method | Used by |
|---|---|
| `brier_score` | evaluate.calibration |
| `jaccard_overlap` | evaluate.ranking_stability |
| `held_out_accuracy` | evaluate.held_out_accuracy |

Implement each alongside the stage that produces the artefact it inspects, and register it in `METHOD_HANDLERS`.

### 3.3 `retry_policy.on_failure` is honoured

`continue_with_logging` (extract, classify) does not halt the pipeline; `pause_and_alert` does. This lives on `StageResult.halts_pipeline`. It was previously read and discarded.

### 3.4 Stage contract

Every stage module exposes `run(spec: dict) -> None`, optionally `run(spec, context=None)`. The orchestrator calls `check_conforms()` immediately after import, so a mistyped or mis-signed `run` fails **before** the pipeline starts rather than hours in. A contract violation returns immediately without consuming retries, because retrying cannot fix a defect.

---
