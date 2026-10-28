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

Last updated: 2026-10-29 (P1 merged; corpus2 refit running).

## 00. 2026-10-29: P1 fixes done, merged to main 490a5d1, corpus2 refit

**Merged to `main` 490a5d1 (not pushed)**: `wip/troop-p1` with 06cf2d0 (review
fixes), 8e75e0b (heavy-tail fixture at 0.99), 6e9ca4d (review follow-ups).
Verified on merged main: ruff, mypy (159 files) clean; one head d0e1f2a3b4c5;
non-model suite **2732 passed, 0 failed, 0 skipped**
(`data/logs/main_nonmodel_suite_20261029.log`). Model files at 8e75e0b/6e9ca4d:
p1_recovery 13, recovery 19 + 1 xfail, contamination 3, neutral 5 + 1 xfail,
impute_recovery 5.

Departures from the 2026-10-25 decisions, with evidence:
- **Centring (MAJOR 1) tried and reverted.** Centred `b` gave 2081 divergences
  on the main recovery corpus; centred `e` failed convergence on tau_e/e in
  recovery and contamination at 0.97-0.99. Non-centred stays; target_accept
  stays 0.99 (corpus at 0.9 had 0 divergences, but fixtures diverged at 0.97;
  the lead kept 0.99 to protect corpus2's 0-divergence write gate).
- **Cross-side learning (MAJOR 2): no cut.** Corpus copy, P1 vs
  `side_prior_independent`: unlabelled 0.00, staff_return 0.40, ancient_claim
  0.09 prior sd (threshold 0.5). Leak fixture: P1 cross-side pass-through
  0.44 vs prior-implied s_b^2/(s_b^2+s_d^2) 0.73 (xfail, measured);
  independent cross-side -0.11. Model reviewer: no BLOCKER/MAJOR; a cut would
  be worse. Re-raise only if a regime moves > 0.5 prior sd on corpus2.
- Casualties were sampled at nu floor 3 since P1; now back to 2, so casualty
  results move vs the last corpus2 run for that reason alone.
- TODO (reviewer, optional): scale-mixture battle effect to lower target_accept.

Corpus2 refit (backup `data/backups/general_war_corpus2_20261029_pre_p1_refit.dump`),
from `main` 490a5d1, `--continue-past-deferred`, logs `data/logs/refit20261029_*`:
extract (36/36 overrides, 237 battles, 422 sides), resolve (4 saved + 2 new
Haiku answers: Stalin at Berlin -> Q855, Wöhler -> Q64794; 50/50 links; warning
duplicate generals 6), reconcile (0 divergences, rhat 1.004, wb gap 0.045),
classify (25 saved + 2 new Sonnet answers: Gazala, Walaja), impute 110
(kept-pair 0.643), model 132 (0 div, rhat 1.003), evaluate 5/5 (held-out
0.780, Brier 0.164), report (93 headline). **Both reviewers: no blocker for an
internal reading; not publishable.** Compression fixed (reconcile wb 0.60 ->
0.63, impute 0.43 -> 0.64). Khalid #4 3.71 (small-sample, unbroken wins vs
one-off opponents); Pyrrhus #14 0.70, driven by Beneventum coded indecisive.
**Read handover-log §35**: record, and the open list (Beneventum outcome
override, Waterloo Blücher role, Yamama/Kalka figures, §30 subordinate MAJOR).
Next: those overrides (draft, user signs off), then corpus3 (decision 4).
Scratch DBs and worktrees from §0 still exist; ask before removing.

## 0. In flight (2026-10-25): P1 validated on a corpus copy, NOT merged

**Read handover-log §34 first.** `wip/troop-p1` at 07f529c (P1 + troop-scope
+ 39 approved overrides + manual_entry bias pin). On a corpus copy, the base
(noise 0.22/0.40) passes every reconcile error gate. wb 0.63, implied
reliability 0.67, impute kept-pair 0.67 (corpus2 had 0.60 / 0.43). The ranking
barely moves between noise 0.22 and a free prior. Khalid #4 and Pyrrhus #14
are unchanged; their cause is the §29 ancient figures, which P1 does not
address.

**User decisions, 2026-10-25 (all as recommended):**
1. **P1 approved at noise 0.22 / 0.40** as the default. Disclose the
   sensitivity table (§34) with the ranking.
2. **All three model-review MAJORs are fixed before the corpus2 refit:**
   - (a) centred `e` (ZeroSumNormal sigma=tau_e) and a centred battle
     effect, with zd non-centred; re-measure divergences at target_accept 0.9;
   - (b) store the g_regime posteriors (mean, sd, prior sd) in diagnostics,
     then measure cross-side learning per regime against an independent-sides
     fit; cut in two stages if any regime moves by more than 0.5 prior sd from
     across sides; replace the Gaugamela `abs < 0.25` test with a direct leak
     test;
   - (c) tests/model/conftest.py draws `is_ancient` per battle.
3. **The five python-review MEDIUMs are fixed in the same pass,** before the
   merge. Restore the casualty `student_t_nu_floor` to 2 (casualty-prefixed),
   so casualties are unchanged.
4. **Corpus3 waits until after the corpus2 refit,** one corpus change at a
   time.

Order: fixes on `wip/troop-p1` (tests first) -> P1 model tests plus a re-run
of the base reconcile on `general_war_wt_p1_corpus` (expect wb about 0.63,
0 divergences) -> both reviewers -> full non-model suite -> merge to main
(no suite running on the main checkout) -> back up corpus2 -> refit corpus2
in stage order with `--continue-past-deferred` -> both reviewers -> report.

Classify on corpus2 will need the same 25 Sonnet answers; they're in
`.claude/worktrees/w-p1run/data/llm_responses/classify_20261024T142520Z/`.
Re-import them with `python -m scripts.llm_offline import --dir ...` against
corpus2. The 4 resolve answers are in `.../resolve_20261024T134330Z/`.

Scratch DBs, all from this session: `general_war_wt_p1`, `general_war_wt_scope`,
`general_war_wt_p1_corpus`, `general_war_wt_p1_s015`, `general_war_wt_p1_s030`
and `general_war_wt_p1_sfree`. Ask the user before dropping any of them.

Worktrees, also from this session: `w-p1`, `w-scope`, `w-p1run`, `w-s015`,
`w-s030` and `w-sfree`. Each has a data/raw junction; remove it with
`cmd /c rmdir` before `git worktree remove`.

The 2026-10-22/23 notes below are kept for the record.

## 0a. Earlier (2026-10-22): workstream A, troop numbers

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
- **2026-10-23 session: the user approved all four as recommended** (P1; new
  wb gate; historiography-drafted `troop_strength` overrides + span guard,
  drafts shown to the user first; explicit level anchor = unmarked post-1800
  figures at face value, g_unlabelled and a_era zero there) and Phase 2b
  ov-0055..0081, 0085, 0086 (0072..0074 retargeted to the Jena/Auerstedt
  children); ov-0082..0084, 0087, 0088 held. Plan:
  `C:\Users\LeeMo\.claude\plans\pasted-content-id-ce35-resume-the-synthetic-scone.md`.
  Branches: `wip/troop-p1` (worktree `w-p1`, recon + main, P1 model, DB
  `general_war_wt_p1`), `wip/troop-scope` (worktree `w-scope`, from p1, the
  override kind + span guard, DB `general_war_wt_scope`).
- **Corpus widening (user, 2026-10-23):** the corpus is Arsht's 22 generals'
  battles (`config/sources_seed_arsht.yaml`), so anyone outside that roster
  (Montgomery: only Alam el Halfa, 2nd Alamein, Bulge, Overlord) is missing
  most battles. Decision: **complete the campaigns** (every battle sharing a
  Wikidata `P361` parent with a corpus battle, frozen into a seed file), into
  a new `general_war_corpus3`, **after P1 is validated** on corpus2. Size
  measurement in progress (scratchpad `campaigns\`).
- `wip/troop-scope` done (1c5c213): `troop_strength` kind + span guard; suite
  2630 passed, 1 failed (hard-coded check count, fixed, file re-run green;
  full suite not re-run). Approved entries staged in
  `docs/handover-log/33-troop-strength-drafts-approved-2026-10-23.yaml`.
  Override rows share source type `manual_entry`: ask the model reviewer
  whether that type's bias term should be pinned.
- Decisions that were asked (kept for the record):
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
- **Merged to `main` (c6c530f): Jena-Auerstedt split** (`battle_split`
  override ov-0089, children `...#jena` and `...#auerstedt`; ov-0029/0030
  removed as moot; 53 entries). Code review plus fixes; two extract runs on a
  corpus copy identical. **Verified on merged `main` c6c530f, 2026-10-23:**
  non-model suite 2518 passed, 0 failed, 0 skipped
  (`data/logs/main_nonmodel_suite_20261023.log`); ruff, mypy (157 files)
  clean; one head d0e1f2a3b4c5. Model tests not run. Neither merged change has
  been run on `general_war_corpus2` yet (the refit waits for the redesign).
  The §29 drafts for Jena-Auerstedt (Schmettau, Hohenlohe, Kalckreuth) must
  target the child slugs.
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
