# Handover §3. Read before writing code

Part of the handover; the index is `handover.md` at the repo root.

### 3.1 `pipeline/llm` is a package

There is no `pipeline/llm.py` and must not be.

```python
from pipeline.llm import LLMService
service = LLMService.from_spec(spec, db_conn=conn)
result = service.complete(system=..., user=..., json_schema=schema, metadata={"battle_id": bid})
if not result.ok:
    continue          # already logged and flagged for review
```

Retries, schema validation, cost accounting, and a row in `llm_calls` per call.
`llm_calls` is also the **resume cache keyed on request hash**, and the offline
path correlates on the same hash. Providers are routed per stage in the spec
(`params.llm_model`, `params.llm_mode`); see §9.4. In offline mode
`complete()` returns `CallStatus.DEFERRED`, which is never written to
`llm_calls` and is not a failure. `params.candidate_source: local` resolves
with no LLM and leaves ambiguous groups `unresolved`.

### 3.2 Quality gates

`pipeline/quality.py` executes each spec's `quality_checks`. Each check runs in
its own SAVEPOINT. When a gate fails, **check whether the spec is wrong before
assuming the code is**: several gates have measured tables their stage never
writes (see §4). `method:` checks dispatch through `METHOD_HANDLERS`
(`diagnostics_json`, `imputation_gate`, `model_gate`, `evaluation_gate`); an
unregistered method reports "not implemented" rather than passing.
`evaluation_gate` reads `evaluation_report.json` and fails as stale when
the report's `model_run_id` is not the newest converged pooled run.

`pipeline/quality.py`'s `Connection` protocol has **two** test doubles:
`StubConnection` (tests/unit/test_quality.py) and `ScriptedConnection`
(tests/integration/test_orchestrator_gates.py). Change both.

### 3.3 Stage contract and retry policy

Every stage exposes `run(spec)` or `run(spec, context=None)`; the orchestrator
calls `check_conforms()` on import so a bad signature fails before the pipeline
starts. `retry_policy.on_failure: continue_with_logging` does not halt the
pipeline; `pause_and_alert` and `complete_with_warnings` do
(`StageResult.halts_pipeline`).

`--continue-past-deferred` runs later stages on deterministic data after an
offline LLM deferral (exit code 3 at the end; the result is provisional). A
gate failure retries only with `retry_policy.retry_on_gate_failure: true`
(crawl only); reconcile, impute and model have `max_stage_retries: 0` (fixed
seed; model retries non-convergence itself, reconcile and impute do not).
`--stages all` skips unbuilt stages. Gate thresholds may say
`{params.<name>}`; a missing param fails the check. `imputation_gate` reads
the run's `params.scale`.

### 3.4 Schema changes go in two places

Every schema change needs an Alembic migration **and** the same change in
`config/schema.sql`: integration fixtures build the database with
`apply_schema()`, which never runs Alembic. Guard the migration with
`IF NOT EXISTS` so both paths converge. `apply_schema()` and the baseline
migration use a raw driver cursor because psycopg reads the `%` in schema
comments as a placeholder, even through `exec_driver_sql`.

---
