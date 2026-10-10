## Setup and infrastructure

Built: schema, Alembic, orchestrator with a real DB connection, LLM client
(anthropic + gemini, routed per stage), quality check runner, CI, pre-commit.

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
- [ ] **.env spells the key GEMIINI_API_KEY (two I's) and holds nothing else,
      no DATABASE_URL, no ANTHROPIC_API_KEY.** Credentials file, user to fix.
- [ ] CI has never run: there is no git remote
- [ ] `pre-commit install` has not been run on this machine
- [ ] Smoke-test the Gemini path against a live key (refusal + truncation
      branches). Harness: tests/integration/test_llm_live.py, gated on
      GENERAL_WAR_LIVE_LLM=1, never run, blocked on the free-tier quota. Either
      branch mistaken for success writes wrong data with an ok status.
- [ ] Quality gates have only run against an empty database: SQL proven valid,
      thresholds not
- [ ] tests/fixtures/gold/: 20 hand-labelled battles for /extraction-eval
      (also unblocks /test-stage extract and classify)
- [ ] Implement evaluate's non-SQL quality check methods (brier_score,
      jaccard_overlap, held_out_accuracy). posterior_check is unused and can be dropped.
- [ ] (gap review A13) `--stages all` halts at evaluate (module missing; the
      complete_with_warnings policy does not prevent the halt). Skip unbuilt
      stages loudly or leave them out of `all` until built
- [ ] (gap review) no end-to-end integration test chains crawl through model
- [ ] (gap review A18) unused deps (cmdstanpy, polars, click, tqdm, matplotlib,
      likely SPARQLWrapper); numpy undeclared; llm_fallback_* and
      log_all_llm_calls are read by nothing
