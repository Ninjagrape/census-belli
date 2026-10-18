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

Last updated: 2026-10-18 (late morning).

## 1. Current state

- **`main`** (74068f8, not pushed) holds every stage, Phase 1, and **Phase 2**:
  `config/overrides.yaml` (54 cited entries the user approved), the loader
  `pipeline/overrides.py`, the `overrides_applied` gate on extract, resolve and
  classify, and migration `d0e1f2a3b4c5` (`battle_type_source = 'manual'`).
  Each branch had code and DB review. Last verified on 2e31821 with the 54
  entries in place: non-model suite **2384 passed + 2 stale tests fixed**
  (`data/logs/main_nonmodel_suite_20261018a.log`; the two then passed in their
  file, the full suite was not re-run after), ruff and mypy clean, one Alembic
  head `d0e1f2a3b4c5`. 74068f8 adds only the unlabelled-entity fix (unit and
  resolve integration tests passed). Model test files not run this session.
- **Corpus** `general_war_corpus2` (236 battles, 420 sides), migrated to
  `d0e1f2a3b4c5`. Refit 2026-10-18: extract (23/23 overrides; its 14,705
  passage requests stay unanswered by the user's 2026-10-14 call, so its gates
  are always skipped), resolve (25/25, every gate passes, warning
  probable_duplicate_generals 6), reconcile run 84 (converged; same three
  warnings), classify (6/6, every gate passes; 18 batch answers imported).
  **Impute run 85 did not converge** (min ESS 302 < 400, rhat 1.009, 0
  divergences; run 83 had ESS 1038), so nothing was written and the pipeline
  halted. Model, evaluate, report not run: there is no valid ranking.
- Backups: `data/backups/general_war_corpus2_20261018_pre_phase2.dump` (before
  this refit), `..._20261017_pre_phase1.dump`.

**Verify before trusting this file:**

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"
ruff check pipeline/ tests/ scripts/ alembic/   # expect: All checks passed
python -m mypy pipeline/                         # expect: Success, 156 source files
python -m pytest tests/ -q -m "not live and not model"   # expect: 0 failed, 0 skipped (~2390)
PYTHONIOENCODING=utf-8 alembic heads             # expect: one head, d0e1f2a3b4c5
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

1. **Impute non-convergence (run 85)**: find the low-ESS parameter (a
   diagnosis fit was running at handoff), decide the fix, and gate it with
   `bayesian-model-reviewer` (any impute change). Suspect: the Phase 2 polity
   relabels added small levels (Axis 6 sides, Allies 2, Sixth Coalition 2,
   Seventh and Third Coalition 1 each) and tau_p fell 0.50 to 0.37. Polity
   keys on the `polity` string (no side has a polity Wikidata id), so the
   relabels do reach impute and the model.
2. **Finish the refit**: impute, model, evaluate, report (classify and
   everything before it are done on current data). Then `historiography-reviewer`
   before reading the ranking, including the Phase 2 entries and the classify
   calls flagged needs_review (Prague 1757: Schwerin at rank 0 over
   Frederick; Wagram's eight low-confidence weights; Waterloo co-command;
   Suez equal split), and `bayesian-model-reviewer` on reconcile's
   prior_predictive warning and on shrinkage.
3. Open questions, decisions and the Phase 1 and 2 records: `07-record.md`,
   `07-phase2-overrides-design.md`.

User decisions, 2026-10-17: build the overrides design **as written**; a
death-only or century-precision lifespan window stays generous and its
namesake cases (Philippikos Bardanes at Ajnadayn) go in the overrides file,
not a code change; **do not push** `main` until the refit is reviewed. This
session's worktrees, `wip/*` branches and scratch databases are removed;
older scratch databases from earlier sessions remain (`general_war_wt_{orch,
crawl,extract,resolve,classify,reconcile,model,infra}`, `general_war_p1`,
`_p23`, `_p4`, `_ci_repro`, `general_war_corpus`): ask before dropping.

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
