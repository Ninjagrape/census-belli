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

Last updated: 2026-10-18 (afternoon).

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
  Impute run 85 failed ESS (302, tau_p ridge with the Soviet Union offset);
  `mcmc_samples` doubled to 4000 (user decision) and run 86 passes every gate
  (rhat 1.006, ESS 685). Model run 108 (rounds 87-106, rhat 1.003, ESS 2038),
  evaluate 5/5 gates (held-out 0.751, Brier 0.176), report written (89
  headline, 643 under-determined). Warnings: PPC category, FMI (Zhukov),
  circularity gap 0.224, 41 of top 50 outside the largest component. Both
  reviewers: no blocker, internal reading only (see §7).
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

The Phase 2 refit is done and reviewed (handover-log §29 historiography,
§30 Bayesian): no blocker, readable internally with caveats, **not
publishable**. The user has not yet chosen what comes next; the candidates,
from the two reviews:
1. **Headline cut** (§30 MAJOR 1): ranks 16-89 are prior-dominated
   subordinates carrying their leader's WAR. Gate the headline on
   `effective_battles` or skill contraction, and/or a per-commander WAR
   counterfactual. Model-meaning change: user decision plus the model gate.
2. **Troop numbers** (§29 M1, §30 MAJOR 2): extract tags labelled modern
   estimates as branch "other" and reconcile drops them, so inflated ancient
   figures survive (Khalid #4 and Pyrrhus #14 rest on them); reconcile
   compresses ratios toward 1 in every era; Eastern Front sides are mostly
   imputed from the Soviet Union polity offset. Reviewer's top pick.
3. **Phase 2b overrides**: 34 cited entries drafted in §29 (ov-0055..ov-0088:
   ~20 wrong-person links, Won Kyun and Francis I, polity mislabels, Prague
   1757 roles), for the user to read.
4. **Polity canonicalisation map** (§29 M5: the Confederacy is 4 levels,
   Napoleonic France 7) and a 1914-1945 era bin (§30 MAJOR 2).
5. **Report disclosures** (§29 M2): rule_single, components, unanswered
   passage pass, corpus shape, fame bias; rank numbering over 732 rows.
6. Smaller: resolver age-window heuristic; PPC pooled over rounds; record the
   worst impute parameter and tail ESS; `.claude/agents/bayesian-model-reviewer.md`
   was not registered as an agent type in the 2026-10-18 session (run it
   through a general-purpose agent told to follow that file).
Open questions and the Phase 1 and 2 records: `07-record.md`,
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
real corpus, extract to report (2026-10-18 on `main` 5a0d27a, with the 54
Phase 2 overrides applied and every override gate passing), LLM batches
answered offline and served from the cache. Recovery suites pass on synthetic
data (not re-run this session).

Model run 108 is the first reviewed ranking. Both reviewers found no blocker
but it is **internal only**: ranks 1-15 rest on their own data; ranks 16-89
are mostly subordinates whose WAR is their leader's side performance; Khalid
(#4) and Pyrrhus (#14) rest on inflated ancient troop figures that extract
let survive; cross-era placement is assumed (41 of the top 50 outside the
largest component); report.md lacks the required disclosures. The corpus data
and the headline rule are the binding constraints. The 2026-10-14 and run-79
numbers must not be cited.

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
