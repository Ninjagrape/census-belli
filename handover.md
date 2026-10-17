# Handover (index)

The state of the project and where everything else lives. `CLAUDE.md` is the
design; this is the state. **Read this file only**, then open the one section
file the task needs (map below). Do not read every file.

Keep it short. Update §1, §7 and §10 here when the state changes. Put a
decision or footgun in the matching section file, and only if a future session
would otherwise get it wrong. Session narratives go in
`docs/handover-log/` (one file per section; index `docs/handover-log/README.md`).
Code comments cite "handover §9.3" (a file in `docs/handover/`) and
"handover-log §16" (a file in `docs/handover-log/`).

Last updated: 2026-10-17 (evening).

## 1. Current state

- **`main`** holds every stage plus Phase 1 (R1 outcomes, R2/R2c/R2d resolve
  lifespan guard and link hygiene, R3 campaign typing), each reviewed. Last
  verified: full non-model suite **2200 passed, 0 failed, 0 skipped**
  (`data/logs/main_nonmodel_suite_20261017e.log`), ruff and mypy clean, one
  Alembic head `c9d0e1f2a3b4`. Model test files last measured before Phase 1,
  which touches none of them: evaluate_cv 4, impute_recovery 5,
  model_recovery 7, reconcile_contamination 3, reconcile_neutral 5,
  reconcile_recovery 19 + 1 strict xfail.
- **Corpus** `general_war_corpus2` (236 battles, 420 sides): extract through
  impute have run on current `main`, every LLM request answered, every error
  gate passing. Reconcile run 82, impute run 83. Warnings: reconcile
  prior_predictive 0.062 (vs 0.05) and empty calibration set; resolve
  probable_duplicate_generals 6 (deliberate UNCERTAIN splits); impute
  distributional_check (early_modern, p=0.0016). 6 non-campaign battles have
  no outcome; 17 campaigns. Model, evaluate and report have **not** been re-run
  since Phase 1: there is no valid ranking.
- Backup before Phase 1: `data/backups/general_war_corpus2_20261017_pre_phase1.dump`.

**Verify before trusting this file:**

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"
ruff check pipeline/ tests/ scripts/ alembic/   # expect: All checks passed
python -m mypy pipeline/                         # expect: Success, 152 source files
python -m pytest tests/ -q -m "not live and not model"   # expect: 0 failed, 0 skipped (~2200)
PYTHONIOENCODING=utf-8 alembic heads             # expect: one head, c9d0e1f2a3b4
python -m pytest tests/model/<file>.py -m model  # one file per command; ~30 min in total
```

Without `DATABASE_URL` the integration tests skip rather than fail. **Any skip
under that marker means the database half is untested.** Live tests are gated
on `GENERAL_WAR_LIVE_CRAWL=1` / `GENERAL_WAR_LIVE_LLM=1`. The `asyncio_mode`
warning is expected (pytest-asyncio declared, not installed, on purpose).

## 6. What is missing

- The extraction gold set (`tests/fixtures/gold/` for `/extraction-eval`).
- Everything open in `docs/todo/` (index `TODO.md`).

## 7. Next steps, in order

1. **Phase 2: build `config/overrides.yaml`** to the design in
   `docs/handover/07-phase2-overrides-design.md` (loader, then the extract,
   resolve and classify kinds in parallel worktrees, one migration for
   `battle_type_source = 'manual'`, the `overrides_applied` gates; review and
   suite before merge).
2. **Draft the entries** with `historiography-reviewer` from its 2026-10-17
   findings (`docs/handover/07-record.md`, "Phase 1 status"), each with a
   citation; the user reads them before they go in. Includes the user's
   calls: per-battle cited outcomes for Borodino, Belmont, Bir Hakeim,
   Kasserine, Maloyaroslavets, Reims 1814 and Lobositz; `side_polity`
   relabels (Rommel on "Kingdom of Italy", Leipzig's coalition, the Bulge).
3. **Refit** in stage order: extract, resolve, reconcile, classify (answer any
   batch), impute, model, evaluate, report. Then `historiography-reviewer`
   before reading the ranking, and `bayesian-model-reviewer` on reconcile's
   prior_predictive warning and on shrinkage.
4. Open questions, decisions and the full Phase 1 record: `07-record.md`.

Say explicitly what was and was not verified against a database.

## 10. Honest status

Built and verified against a real Postgres: every stage, through the CLI on the
real corpus, crawl to impute (2026-10-17 on current `main`), LLM batches
answered offline and served from the cache. Recovery suites pass on synthetic
data.

The corpus data, not the model, is the binding constraint. Phase 1 fixed the
root causes a domain review found (missing outcomes, namesake links outside
lifetimes, stale links, BC years one late, coarse dates read as exact, wars
typed as battles); about 25 row-level errors remain for the overrides file.
Until Phase 2 lands and a refit is reviewed there is no valid ranking; the
2026-10-14 and run-79 numbers must not be cited.

Not proven: a publishable ranking from any corpus, any billed API call, CI on
the current `main` (`origin` exists; never checked), gate thresholds beyond
fixtures and this corpus. The gold sets are agent-built and unreviewed.

## Map: open only what the task needs

| Need | File (`docs/handover/`) |
|---|---|
| Env: Postgres, Python, `.env`, compiler | `02-environment.md` |
| Before writing code: `pipeline/llm`, gates, stage contract, schema changes | `03-read-before-writing-code.md` |
| Classes of past bugs (read before trusting a green suite) | `04-the-lesson.md` |
| Footguns (worktrees, DB use, Windows, hibernation, stage order) | `05-footguns.md` |
| Fixes A-E, Phase 1 record, review findings, open decisions | `07-record.md` |
| Phase 2 overrides design | `07-phase2-overrides-design.md` |
| Reviewer agents, slash commands, inspector | `08-project-tooling.md` |
| Standing design per stage: crawl, extract, resolve, LLM routing, Arsht set, reconcile, classify, impute, model, evaluate, report | `9.1-crawl.md` ... `9.11-report.md` |
