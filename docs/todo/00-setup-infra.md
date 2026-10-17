## Setup and infrastructure

Built: schema, Alembic, orchestrator with a real DB connection, LLM client
(anthropic API or offline batch via Claude Code subagents, per stage), quality check runner, CI, pre-commit.

Standing rules worth knowing before changing these:
- Every schema change goes in both an Alembic migration and `config/schema.sql`:
  the integration fixtures call `apply_schema()` and never run Alembic.
- `llm_calls` doubles as the resume cache, keyed on request hash.
- Each quality check runs in its own SAVEPOINT; without it one failing check
  aborts the transaction and every later check reports a false failure.
- `logging_config.configure()` is process-global. A test that calls it needs the
  restore fixture in tests/unit/test_infrastructure.py (handover-log §15.6).
- CI fails if any test skips (a green run full of skips hid handover-log §4.3-4.4)
  and excludes the `live` marker. ruff-format is staged `manual`: applying it
  reformats the whole tree and belongs in its own commit.
- Postgres 15 runs natively; docker-compose.yml has never been run.

## Open
- [ ] **.env holds only a now-unused GEMINI_API_KEY**, no DATABASE_URL. Credentials
      file, user to fix.
- [ ] CI results not checked: `origin` exists (PR #1 merged as 5d11e7a), but no
      session has confirmed a green run on the current `main`
- [ ] `pre-commit install` has not been run on this machine
- [ ] Smoke-test the `llm_mode: api` path against a live Anthropic key (refusal +
      truncation branches). Harness: tests/integration/test_llm_live.py, gated on
      GENERAL_WAR_LIVE_LLM=1, never run. Either branch mistaken for success
      writes wrong data with an ok status. Optional while every stage is offline.
- [ ] Gate thresholds are proven only on fixtures and the one 211-battle corpus
      (every stage's gates ran against it 2026-10-12..17); untested beyond it
- [ ] tests/fixtures/gold/: 20 hand-labelled battles for /extraction-eval
      (also unblocks /test-stage extract and classify)
- [x] done 2026-10-15: evaluation_gate covers them; posterior_check removed: Implement evaluate's non-SQL quality check methods (brier_score,
      jaccard_overlap, held_out_accuracy). posterior_check is unused and can be dropped.
- [x] (pilot P1) fixed 2026-10-11: orchestrator `run_stage` calls `run(spec)` with no StageContext, so
      extract and resolve write nothing from the CLI. Pass a context with the
      connection; add a test that drives a stage through `run_pipeline`
- [x] fixed 2026-10-12: (pilot) stage retry reruns a fixed-seed sampler; a non-converged fit fails
      identically and doubles runtime. Skip retry for deterministic failures
- [x] fixed 2026-10-12: (gap review A13) `--stages all` halts at evaluate (module missing; the
      complete_with_warnings policy does not prevent the halt). Skip unbuilt
      stages loudly or leave them out of `all` until built
- [ ] (gap review) no end-to-end integration test chains crawl through model
- [x] done 2026-10-15: cmdstanpy, SPARQLWrapper, polars, click, tqdm removed; numpy declared; lock regenerated: (gap review A18) unused deps (cmdstanpy, polars, click, tqdm,
      SPARQLWrapper; matplotlib is used by evaluate and report); numpy undeclared; llm_fallback_* and
      log_all_llm_calls are read by nothing
