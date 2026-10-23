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

Last updated: 2026-10-22 (night, mid-session; workstream A in progress).

## 0. In flight (2026-10-22): workstream A, troop numbers

Approved design and user decisions: `docs/handover/07-next-sessions-plan.md`
"A: approved design (2026-10-22)"; diagnosis: handover-log §31.
- **Merged to `main` (83cb0b7, not pushed):** extract keeps labelled modern
  estimates (supersede rule in reconcile load), scope wording (`population`
  scope, deployed/total/phase rules). Code + DB review, two fix rounds; suite
  2476 passed 0 skipped on the branch. Not yet run on `general_war_corpus2`.
- **`wip/troop-recon` (0c0cb98, worktree `w-recon`):** informative noise
  prior, recovery gates, reconcile exports per-side likelihood, impute uses it
  (no weak sides with a point report). Suite 2446 passed. **It fails every
  recovery gate on the corpus**, and so did all four noise variants in the
  experiment (worktree `w-exp`, branch `wip/noise-exp` = recon + main, DB
  `general_war_wt_exp_corpus`). Do not merge it as is.
- **NEXT SESSION STARTS HERE: the user decides on the redesign in
  handover-log §32** (bayesian-model-reviewer, 2026-10-23). Decisions to ask:
  1. Approve design P1: side prior with era means plus a shared battle effect
     (opposing armies' reported sizes correlate at 0.80), Normal likelihood,
     noise LogNormal(log 0.22, 0.2) / ancient (log 0.40, 0.2), no
     sigma_round/sigma_est for troops, mixed-predictive PPC. Ranked
     alternatives in §32.
  2. Accept that the wb >= 0.8 gate was wrong: a calibrated model shrinks
     reported force ratios by their reliability (~0.6; Kursk 2.45:1 to
     1.91:1). The gate becomes "wb consistent with model-implied reliability".
     This partly reverses the original goal (less compression), on purpose.
  3. Overrides for ~9 scope-conflated sides (Yom Kippur Egypt, Six-Day UAR,
     Lebanon 1982 Israel, Toulon royalists, Stalingrad x2, Smolensk x2; check
     Torch US, Hohenfriedberg), drafted by historiography-reviewer, plus a
     span guard (> log 1.0 warn, > 2.0 error).
  4. The level anchor (HIGH): g_unlabelled moves -0.13..-0.34 with era means,
     shifting published head counts 1.1-1.4x; tighter prior or explicit anchor.
  Then: implement P1 on a fresh branch from `wip/troop-recon` + main, the
  sensitivity runs (noise median 0.15/0.22/0.30/free), and only then the
  refit. Failing-first evidence was not collected for `wip/troop-recon`.
- **`wip/jena-split` (worktree `w-jena`):** `battle_split` override kind;
  ov-0029/0030 removed (moot); review fixes in (no silent line loss, retire
  only after children exist, failed retire raises `SplitRetireError`). Suite
  2518 passed 0 skipped. Corpus re-check (two extract runs) pending, then
  merge to main. The §29 drafts for Jena-Auerstedt (Schmettau, Hohenlohe,
  Kalckreuth) must target the child slugs `...#jena` / `...#auerstedt`.
- Backup before any of this: `data/backups/general_war_corpus2_20261022_pre_troopfix.dump`.
  Scratch DBs this session: `general_war_wt_{troop,troop_corpus,recon,
  recon_corpus,jena,jena_corpus,exp_corpus}`.
- Then: refit corpus2 from extract in order, both reviewers, report Khalid,
  Pyrrhus and force-ratio compression before (reconcile 0.60, impute 0.43).
- TODO to file: the generic "(<Adjective> sources)" ancient_claim rule would
  tag WWII "(German sources)"; no corpus hit today (only a naval row).

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
publishable**. **The plan is `docs/handover/07-next-sessions-plan.md`; read
it before starting.** Nothing in it has been started. In order:
1. **A. Troop numbers** (next session; its prompt is in the plan): keep
   labelled modern estimates in extract, read scope wording, then have the
   model reviewer diagnose reconcile's ratio compression and impute's
   weak-side rule before any change; Jena-Auerstedt is a user design call.
2. **B. Phase 2b overrides**: the user reads the 34 drafts (handover-log §29
   section 4; the plan has a review table with the calls to make); build in
   only what is signed off, after a twin check. Fold into A's refit.
3. **C. Report disclosures and a data-coverage figure set** (designed in the
   plan; report stage only, can run beside A).
4. **D.** The planned TODO fixes (polity map, era bin, resolver age window,
   PPC pooling, end-to-end test, named check), listed in the plan and filed in
   `docs/todo/`.

User decisions, 2026-10-18: keep the headline rule (`min_battles: 3`) and
disclose that WAR is side performance shared by weight; impute
`mcmc_samples` 4000; the 54 entries as drafted, with Borodino a French
pyrrhic win, per-battle Scipio links instead of the merge, Waterloo,
Austerlitz and Bautzen relabelled as coalitions, Kasserine attribution, and
Anderson linked (plus the Torch link added after review). The README now
carries the methodology and the provisional findings; keep it in step.
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
(troop figures, coverage) is the binding constraint; the headline rule stays by
the user's call and is disclosed. The 2026-10-14 and run-79 numbers must not
be cited.

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
| Phase 2 overrides design and build record | `07-phase2-overrides-design.md` |
| **Plan for the next sessions** (A troop numbers with its prompt, B Phase 2b review, C report disclosures and figures, D TODO fixes) | `07-next-sessions-plan.md` |
| Reviewer agents, slash commands, inspector | `08-project-tooling.md` |
| Standing design per stage: crawl, extract, resolve, LLM routing, Arsht set, reconcile, classify, impute, model, evaluate, report | `9.1-crawl.md` ... `9.11-report.md` |
