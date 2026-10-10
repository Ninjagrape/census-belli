## Project Setup
- [x] CLAUDE.md project spec
- [x] Database schema (config/schema.sql)
- [x] Agent specs for all 9 pipeline stages
- [x] Pipeline orchestrator skeleton
- [x] pyproject.toml with dependencies
- [x] Seed sources config
- [x] .gitignore
- [x] README.md (public-facing, not the CLAUDE.md)
      - states plainly that the statistical half does not exist and that there
        is no published ranking; lists the limitations bearing on believing one
- [x] LICENSE (GPL-3.0)
- [x] Alembic migration init from config/schema.sql
      - applied to a live database 2026-09-19; upgrade/downgrade/re-upgrade verified
- [x] Alembic migration 0002: battles.year_astronomical
      - also added to schema.sql, because the integration fixtures call
        apply_schema() and never run Alembic. Both paths need every future change.
- [x] .env.example with required env vars (DATABASE_URL, ANTHROPIC_API_KEY, GEMINI_API_KEY)
      - [x] pipeline.config.load_env() now reads it, and the orchestrator CLI
        calls it. An already-exported variable wins, so CI secrets and a
        deliberate export are never overridden.
      - [x] BOM sniffing: the .env on this machine is UTF-16-LE, so python-dotenv
        failed on its first byte. See handover.md 15.5.
      - [ ] **the key in .env is spelled GEMIINI_API_KEY (two I's), and the file
        holds nothing else — no DATABASE_URL, no ANTHROPIC_API_KEY.** Left alone
        deliberately: it is a credentials file. User to fix.
- [x] Postgres for local development
      - PostgreSQL 15.19 installed natively via winget; no container runtime on
        this machine. docker-compose.yml is written but has never been run.
- [x] CI config (GitHub Actions: lint, type check, test)
      - runs the suite against a real postgres:15 service and **fails the build
        if any test skips**, because a green run full of skips is the blind spot
        that hid 4.3 and 4.4. Excludes the `live` marker, so CI never touches
        Wikipedia. Also round-trips the Alembic migration.
      - [ ] never actually run: there is no git remote
- [x] Pre-commit hooks (ruff, mypy, and a hook refusing to commit a .env)
      - [ ] `pre-commit install` has not been run on this machine
      - ruff-format is registered but staged `manual`: applying it reformats the
        whole tree and belongs in its own commit


## Infrastructure
- [x] Database connection module (pipeline/db.py) using SQLAlchemy Core
- [x] Config loader that merges agent YAML specs with CLI overrides
- [x] Structured logging setup (pipeline/logging_config.py)
      - console renderer on a terminal, JSON otherwise; LOG_LEVEL honoured;
        httpx and sqlalchemy.engine held at WARNING
      - NOTE: configure() is process-global. A test that calls it needs the
        restore fixture in tests/unit/test_infrastructure.py, or it silences
        unrelated tests elsewhere in the suite. See handover.md 15.6.
- [x] LLM client wrapper (pipeline/llm/) that handles retries, token logging, cost tracking
      - provider-agnostic: anthropic + gemini, routed per stage via agents/<stage>.yaml
      - llm_calls table doubles as the resume cache, keyed on request hash
      - [ ] smoke-test the Gemini path against a live key (refusal + truncation
        branches). **Harness written: tests/integration/test_llm_live.py, 5 tests,
        gated on GENERAL_WAR_LIVE_LLM=1. Never run** — blocked on the free-tier
        quota below. Refusal and truncation are the branches worth paying for:
        either one mistaken for success writes wrong data with an ok status.
- [x] Quality check runner (actually execute the SQL checks in orchestrator.py)
      - each check runs in its own SAVEPOINT; without it one failing check
        aborted the transaction and every later check reported a false failure
      - all 22 SQL gates across all 9 specs execute without error
      - [ ] gates have only ever run against an empty database, so their SQL is
        proven valid, their thresholds are not
- [x] Stage runner base class / protocol that each pipeline/stages/*.py implements


## Tooling (.claude/)
- [x] bayesian-model-reviewer agent — inference correctness for the PyMC stages
- [x] historiography-reviewer agent — domain plausibility of extracted data
- [x] /data-audit command — data-level health check, distinct from /status
- [x] /extraction-eval command — prompt regression harness
- [ ] tests/fixtures/gold/ — 20 hand-labelled battles the eval harness scores against
      (the real work behind /extraction-eval; also unblocks /test-stage extract and classify)
- [ ] Implement evaluate's non-SQL quality check methods (brier_score,
      jaccard_overlap, held_out_accuracy). diagnostics_json, imputation_gate
      and model_gate are done; posterior_check is unused and can be dropped.
- [ ] (gap review A13) `--stages all` halts at evaluate (module missing; the
      complete_with_warnings policy does not prevent the halt). Skip unbuilt
      stages loudly or leave them out of `all` until built
- [ ] (gap review) no end-to-end integration test chains crawl through model
- [ ] (gap review A18) unused deps (cmdstanpy, polars, click, tqdm, matplotlib,
      likely SPARQLWrapper); numpy undeclared; CLAUDE.md names
      scripts.add_battle and scripts.quality_report, which do not exist;
      llm_fallback_* and log_all_llm_calls are read by nothing
- [x] **Wire a real DB connection into the orchestrator CLI.** Done. main() now
      loads .env, configures logging, opens a connection and passes it down.
      `--no-db` keeps the old behaviour explicitly; a missing DATABASE_URL exits 2
      rather than running nine stages and reporting nine identical gate failures.
      Verified: the CLI now returns real gate verdicts.
