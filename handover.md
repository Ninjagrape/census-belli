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

Last updated: 2026-11-10 second half (density-weighting design reviewed, §43; data-fix drafts; DB access fix pending; see §000 "2026-11-10 second half").

## 000. 2026-10-29/30: corpus3 seed, overrides ov-0101..0104, crawl

- **2026-11-10 second half (resume here; handover-log §43).**
  - **DB access first:** `DATABASE_URL` is now in the Windows User env, but Postgres rejected its password; the user re-sets it with the working one. Agents read it via `os.environ` then `winreg` HKCU\Environment and swap the DB name to `general_war_corpus3`; they must not read `.claude/settings.local.json` (the classifier blocks it).
  - Density weighting design written and reviewed: `docs/handover-log/43-design-density-weighting.md`, BMR GO with changes; section 8 amendments are binding (post-hoc routing, every `reduce_draws` caller, `cmd_method`, report readers, primary war by weighted battles, within-war 90% rank intervals from draws, `scope: global|component`, `war_per_battle` stays unweighted). Not implemented.
  - Drafts awaiting sign-off: ov-0700..0704 in `docs/handover-log/43-drafts/` (Actium side 125 = Octavian's, 19,000; Chester Station/Swift Creek attribution entries **cannot apply** as drafted, options in §43 section 10; Capt. Cleaver unlink; Brice's 3,500). Lone-branch: 30 of 72 promoted sides are cavalry, most whole cavalry forces; proposal: no offset for cavalry-only.
  - rule_single: cutting a sole commander below 1 needs a model change (W < 1 on ordinary sides, remainder `mu_unattributed`); user + BMR decide before any prompt change or LLM batch (§43 section 10).
  - **Next, in order:** (1) With DB access: Haiku runs the density measurement (§43 section 9) and the Chester/Swift side check; Sonnet drafts the nested-pair and campaign retypes (brief in §43 section 10). (2) User signs off ov-0700..0704 (choose an option for Chester/Swift) and the retypes; decides the cavalry-only offset and rule_single W < 1 design. (3) Sonnet implements density weighting per §43 (worktree, tests), BMR gate on the code. (4) rule_single prompt + batch (ask usage %). (5) Integration suite on a scratch DB, then refit 8 with `GENERAL_WAR_DATA_ROOT=data/corpus3`.

- **2026-11-10 (resume here; handover-log §42).**
  - Both reviewers ran on refit 7 (reconcile 29, model 52): no inference blocker; estimates fit to write. All §41 fixes verified except Brice's Cross Roads value (lone-branch offset lifts an all-cavalry 3,500 to about 6,700).
  - **ACW bias diagnosed** (§42 section 4): not Confederate, not outnumbered-win credit; the model rates ACW generals lowest per battle. Rank comes from density x volume x rule_single credit x prior-centred components with one global replacement level (BMR M3). WW2 is low from credit splitting.
  - **Decision pending with the user** (§42 section 5): aggregation density weighting + rule_single cut (BMR C+D, ACW 17 -> 9 of top 50) vs size tiers + sub-action collapse in likelihood and sum (Hist E, 17 -> 12), plus within-component headline ranks. No implementation before the choice.
  - Data fixes found, no design choice needed (draft overrides/changes for the user): Actium side 125 archers-as-total; 55 nested parent/child battle pairs; about 20-25 campaign/operation rows; Chester Station and Swift Creek Beauregard credit; cavalry-only lone-branch offset; "Capt. Cleaver".
  - Done: records.py docstring nit (w-c3seed). Data-root code review: no CRITICAL/HIGH; follow-up fixes done in agent-a512e2a4e3d0264a4 (root-named extract hint, `report_under_root` rejects `..`/absolute, override-precedence comments, 4 new tests): unit 2569, ruff, mypy 162 clean. Ready to commit.
  - **Merged 2026-11-10 (main, not pushed):** 86b64be (w-c3seed work) and 17144d4 (data root) merged into main (31a2c67, 37e3796); no conflicts. w-c3seed's two new report gates converted to `report_under_root`; `.gitignore` ignores `data/corpus*/`. Corpus3 data moved to `data/corpus3/` (run corpus3 with `GENERAL_WAR_DATA_ROOT=data/corpus3`; the refit-7 log is `data/corpus3/logs/corpus3_refit7_20261109.log`). w-c3seed and agent-adf22892058598c5e worktrees removed, branch wip/corpus3-seed deleted (an empty `.claude/worktrees/w-c3seed` folder may remain, file lock). Kept: agent-a512e2a4e3d0264a4 worktree (merged; remove when convenient) and branch worktree-agent-adf22892058598c5e (ee1c9de, its content is in main via cherry-pick). Verified on main: unit 2773, ruff, mypy 168, overrides 481 unique; no integration run since the merge.
  - **ACW design DECIDED** (§42 section 6): density weighting of the WAR sum (d_ref 29, power 0.5, likelihood unweighted); classify-level rule_single fix with a 0.5 evaluate variant meanwhile; global plus within-war ranks; nested pairs keep sub-battles (parent retyped campaign).
  - **Next, in order:** (1) Draft the §42 data fixes as overrides/changes for sign-off: Actium side 125, 55 nested parent/child pairs, about 20-25 campaign rows, Chester Station/Swift Creek, cavalry-only lone-branch offset. (2) Design brief for density weighting + within-war ranks (bayesian-model-reviewer on the brief), Sonnet implements. (3) rule_single classify change and LLM batch (ask usage %). (4) Implement (bayesian-model-reviewer gate), integration suite on a scratch DB, then refit 8 from main with `GENERAL_WAR_DATA_ROOT=data/corpus3`.

- **2026-11-08 (resume here; handover-log §41).**
  - Both reviewers ran on run 5 / ranking run 28: no model blocker; keep n_imputations 20. Historiography found two reconcile blockers (100-troop floor drops real small sides; single-arm figures always dropped) and campaign double counting (§41 section 2).
  - User approved and added to main + w-c3seed: ov-0607 (Rovereto) and ov-0608..0616 (Mesilla unlink, Fort Sumter 85, five campaign retypes, Fredericksburg date, Fort Wagner field).
  - w-c3seed (uncommitted, all verified and reviewer-gated GO): disclosures cherry-picked from ee1c9de plus the agreement fix; override floor exemption; lone-branch promotion; per-side prior-only `d_lone_branch[k] ~ TN(log 1.5, 0.6, lower 0)` (the shared scalar was rejected: it learned from the opponent gap); reports_honoured B1 fix; overrides ov-0607..0616. Unit 2563, reconcile integration and model files pass (§41 section 3).
  - **Refit 7 DONE** 2026-11-10 04:24 (reconcile run 29, model run 52, evaluate 5/5, report gates pass; all reconcile gates green, reports_honoured 0 misses, 62 sides with prior-dominated `d_lone_branch`). Gates read; details and top 20 in §41 section 3b. Log `w-c3seed/data/logs/corpus3_refit7_20261109.log`. Pre-refit backup `data/backups/general_war_corpus3_20261109_pre_rerun7.dump`.
  - Data-root setting built in worktree agent-a512e2a4e3d0264a4 (off main), uncommitted, not code-reviewed (§41 section 5).
  - **Next, in order:** (1) Gates are read (§41 3b); run both reviewers, both briefed on the user's ACW-bias concern (§41 3c: 17 of top 50 are ACW, WW2 credit split, volume effect) (bayesian on reconcile+model, historiography on the ranking: Burnside/Beauregard/Forrest, campaign rows gone, Fort Sumter, Brice's Cross Roads). (2) Nit: records.py:465 docstring still says "shared offset". (3) Code-review the data root, merge wip/corpus3-seed and the data-root branch into main (agents/report.yaml conflict expected), move w-c3seed `data/` to `data/corpus3/`, then remove w-c3seed and agent-adf22892058598c5e with the user's OK. (4) Open items from §41 section 2 (rule_single audit and variant, PPC/accuracy gates, date-parser bug, campaign rows sweep).

- **2026-11-07 (resume here; handover-log §40).**
  - **Corpus3 reconcile run 5 converged** with opt-in `battle_effect_centred=true` (C): troops rhat 1.006, ESS 1517, 0 divergences; noise 1.134. Troop estimates written. One gate miss: reports_honoured, Rovereto 1796 Austrians (scope inversion). Draft ov-0607 awaits the user's sign-off (§40 section 5).
  - Missingness batch done: 2,495 + 66 classify answers and 1 resolve imported into corpus3.
  - **First corpus3 ranking produced** (provisional; every stage succeeded, warning-level checks failed; §40 section 5b). Outputs in w-c3seed `data/processed/report/`. Unreviewed.
  - `era_effect: none` sensitivity: noise and ratio results unchanged, era effect carries level placement (§40 section 5).
  - Report disclosures (plan C) built and reviewed in worktree `agent-adf22892058598c5e`; all report gates pass on a corpus2 copy.
  - Uncommitted: w-c3seed (reconcile work + tests), the agent worktree (report), main (`config/overrides.yaml` Dürenstein slug, footguns, docs). The user commits; suggested messages in the session's final reply.
  - **Next:** (1) bayesian-model-reviewer on reconcile run 5 and the model run, historiography-reviewer on the ranking (Burnside, Beauregard, Forrest). (2) User signs off ov-0607; then the refit: extract..report with all answers cached. (3) Data-root layout (proposal in the session reply, awaiting OK), then commit and remove w-c3seed.

- **2026-11-06 (resume here; handover-log §39).**
  - **main** is 519e60d plus uncommitted, verified fixes:
    - extract: the head noun decides equipment counts; tilde ranges are read as ranges.
    - resolve: a wikilink becomes a commander's identity only when it names the mention; link-to-item attachment, decoding and fetch-failure counting are fixed.
    - overrides: ov-0035, 0105 and 0108 retired; ov-0126 retargeted to `United Kingdom`; ov-0600..0606 added (471 entries).
    - Verified: unit 2612, ruff, mypy. Integration on scratch `general_war_endash_it`: full suite 374/0/0 before the fixes; extract/resolve/override subset 142/0/0 with them (before the tilde fix).
    - User commits; suggested messages in §39 section 1. main is ahead of origin, and the user pushes from GitHub Desktop.
  - **w-c3seed (wip/corpus3-seed, 2808819)** holds uncommitted reconcile work:
    - the Kalka override-supersede fix (reviewed);
    - opt-in nutpie low-rank adaptation;
    - tune 4000 / draws 8000;
    - ov-0600..0606 (474 entries);
    - the A+B reparameterisation: built, unit 2455, reviewer no BLOCKER/MAJOR (§39 sections 8-9). Model and recovery tests are not run yet.
  - **Reconcile run 4 FAILED ESS (160, after 6 h with low-rank)**; the user stopped it during the neutral fit. Cause, per the model reviewer: the per-era-bin battle-effect mean duplicates the era level, and `s_d` vs infobox noise trade off on single-row sides. That is model geometry, which no sampler setting fixes.
  - **Approved fix, now built:** (A) centre `zb` within era bin, a small stated prior change; (B) marginalise `d` exactly on single-row point sides. Design: `docs/handover-log/39-design-reconcile-A-B.md`.
    - The 3-seed check at 1000 tune shows ESS about 200-260 per 4000 draws, against about 10-40 before.
    - One seed had a frozen chain, from a step-size collapse in short tuning.
    - `s_b` is the weak spot.
  - **Noise gate (about 1.49-1.51 against 1.5):** the audit found it driven by contaminated multi-row sides. Overrides ov-0600..0606 approved (Yonan tilde range, Odesa totals, Dürenstein engaged, First Badajoz garrison); Ratisbon held; Ajnadayn none. They apply at **extract**, so corpus3 needs extract, resolve and classify re-run (cache-served) before the next reconcile: ask the user first.
  - **Next, in order:**
    1. Follow §39 section 9, steps 1-4:
       - add a frozen-chain diagnostic plus MINORs 1 and 4;
       - run a 3-seed scratch check at tune 4000 / draws 2000 on `general_war_c3_diag`;
       - option C (centred b) or low-rank only if `s_b` misses;
       - model/recovery tests one file at a time, plus `test_reconcile_p1_gates.py` on a scratch DB.
    2. With user OK: back up corpus3, then extract,resolve,classify,reconcile from w-c3seed (`--set seed_path=config/sources_seed_corpus3.yaml --continue-past-deferred`). Read the gates; run the reviewer and `era_effect: none`.
    3. Classify plus the 2,488 missingness requests (ask usage %).
    4. Then `GENERAL_WAR_DATA_ROOT` plus `data/corpus3/` (user OK'd the direction; show the layout before building), commit, and remove w-c3seed.
  - Footgun added: scripts import main's `pipeline` (editable install) unless `PYTHONPATH` is pinned (05-footguns.md).

- **2026-11-03 night (handover-log §38).** §37 steps 1-5 ran.
  - **main = 40172bf (not pushed):** everything is fast-forwarded into it
    (corpus3-seed, extract-endash, the overrides, the cloud extract/resolve
    fixes). Unit 2547, mypy 161 and ruff are clean. No integration test or
    corpus run on main yet.
  - **wip/corpus3-seed (w-c3seed) = 2808819,** without the cloud commits. Its
    integration suites passed (reconcile/prior_check 82, extract 51).
  - **Corpus3:** extract, resolve and classify re-ran and every override gate
    applied (467). The 26 retypes land at classify, so classify ran before
    reconcile (footgun added).
  - **Reconcile run 3 FAILED, but close.** noise 3.30 -> 1.514 (gate 1.5),
    within-battle 0.10 -> 0.76, rhat 1.58 -> 1.041, **troop ESS 130 < 400**
    (neutral fit 116). No troop estimates written; casualties written.
  - **Cause found:** ov-0103 Kalka (`troop_strength`) did not replace the
    infobox `modern_scholarly` row, so it is the worst side (span 1.5). That
    is a load-logic bug. Otherwise only 35 multi-row sides identify the
    noise.
  - **Next, in order (user-approved):**
    (a) Sonnet fixes the Kalka replacement bug in reconcile load, on
    wip/corpus3-seed, tests first, then bayesian-model-reviewer;
    (b) raise reconcile troop tune/draws for ESS;
    (c) back up, re-run reconcile on corpus3, then bayesian-model-reviewer
    and an `era_effect: none` run;
    (d) only if the noise gate still fails, source Odesa/Dürenstein/Badajoz/
    Ratisbon;
    (e) classify re-run plus the 2,488 missingness requests
    (`classify_20261103T100314Z.jsonl`); ask the user's usage % first.
  - **Before main runs extract or resolve:** see the cloud items in §38
    section 6 (retire ov-0035/0105/0108, confirm three retargets, real-article
    passage check, wikilink network fetches).
  - **Worktrees:** the user wants all four removed once corpus3 reconcile is
    done. w-c3seed's 6.2 GB data/ needs its own folder, not a merge into
    data/ (§38 section 7).

- **2026-11-03 evening (resume here; handover-log §37).** Step 1 is done. Its
  fixes are approved but not applied, and no DB writes were made.
  - **48 overrides approved by the user:** ov-0500..0524 campaign retypes
    (Kursk and Berlin stay battles, because campaign removes a battle from the
    WAR model), ov-0530..0549 troop scope and polity, ov-0550 Seelow
    `theatre_strength`, ov-0551/0552 Okinawa amphibious and dated. They are in
    `docs/handover-log/37-corpus3-contamination-overrides-approved-2026-11-03.yaml`,
    validated (467 entries load, every target matches exactly once). Not yet
    in `config/overrides.yaml`.
  - **Extract fix**, branch `wip/extract-endash` (worktree `w-endash`, from
    288eee5), uncommitted. It fixes the `{{endash}}` range glue (infobox path
    only), "with <Name>" as a detachment, and Tournai's flag `size=20px`.
    Unit 2424 passed and ruff is clean. Passage text is byte-identical across
    3304 articles, so the LLM cache stays valid. Only the 8 expected battles
    change.
  - **mypy fails on wip/corpus3-seed** at `pipeline/reconcilers/model.py:765`
    (float * None), from the era-fix commits. Fix it before the merge.
  - **Next, in order:**
    1. The user commits w-endash (message in §37) and merges it into
       wip/corpus3-seed; fix the mypy error.
    2. Paste the 48 entries into w-c3seed `config/overrides.yaml` and load-check
       them.
    3. Run the integration suite `-k "reconcile or prior_check"` (plus the
       extract tests) on a scratch DB.
    4. Back up corpus3, then re-run extract onward with `--set
       seed_path=config/sources_seed_corpus3.yaml` and
       `--continue-past-deferred`. LLM answers should all come from the cache,
       and Tournai's sides change, so classify may re-request its roles.
    5. Reconcile; all gates must pass. Then bayesian-model-reviewer (as a
       general-purpose Opus agent with the spec file) and an `era_effect:
       none` sensitivity run.
    6. Then §36 step 4 (missingness requests: ask the user's usage % first).
  - Cloud sessions need the branch pushed. The user pushes from GitHub Desktop,
    with the worktree added as a repository; the CLI credential manager does
    not work here.

Branch `wip/corpus3-seed` (worktree `.claude/worktrees/w-c3seed`, created
this session), 9f9862f: `scripts/campaign_seed.py` (+11 unit tests),
`config/sources_seed_corpus3.yaml`, ov-0101..0104 (96 entries load), crawl UA
now names the repo. Not merged.
- **User decisions 2026-10-29:** the literal P361 rule (1782 new, mostly whole
  wars: ACW 485, WWII 251) is too broad, but keep WWII, Peninsular and
  Napoleonic. Rule approved: **shared commander** (keep a candidate whose
  infobox names a wikilinked commander who is in 2+ kept battles; corpus2
  always kept). Result: 1441 new + 211 corpus2 URLs (237 battles) = 1652
  URLs. Drops: 36 no infobox, 71 no linked commander, 234 no shared
  commander. 706 of 1441 kept have no commander in corpus2. Infobox counts
  corpus2 -> corpus3: Montgomery 4->12, Zhukov 10->18, Rommel 9->13,
  Wellington 1->33, Napoleon 43->77, Khalid 14->34, Pyrrhus 3->7; commanders
  with 3+ battles 115->621. Regenerate with the script (`--pool` re-runs only
  the filter).
- **Overrides approved (user, 2026-10-29):** ov-0101 Beneventum Roman victory,
  ov-0102 Waterloo Blücher co-field (weight 0.25 kept), ov-0103 Kalka Rus'
  10k-30k, ov-0104 Torch Vichy 105,000. Yamama declined (no modern estimate).
  Twin check clean. For corpus3's first run only; corpus2 is not refitted.
  corpus2 has a sideless Beneventum shell row (battle_id 32).
- **Crawl: citations on** (user). Corpus3 runs in the w-c3seed worktree with
  its own `data/` (a real copy of `data/raw`: extract reads every file under
  the raw dir), DB `general_war_corpus3` (schema + alembic_version + llm_calls
  copied from corpus2, so corpus2's answers are cached). Log
  `data/logs/corpus3_crawl_20261030.log` (started 2026-10-30 00:20, PID
  55884; corpus2's llm_calls loaded with battle_id NULL, which is only
  metadata: 919 rows, input_hash sets equal). If it died, re-run the same
  command from the worktree; crawl_state.json resumes it. Backup
  `data/backups/general_war_corpus3_20261030_empty_with_cache.dump`.
- **Done 2026-10-30:** crawl 1652/1652 (gates pass; Vimeiro and Zornoza
  failed on wake from hibernation and came in on a re-run). Extract and
  resolve done, all LLM answers in the cache (outcome 260+58 Sonnet,
  resolve 401+92 Haiku; 45,250 passages deferred as agreed). Backups
  `..._post_crawl.dump` (before extract) and `..._post_resolve.dump`.
- **Two bugs found and fixed on the branch** (not merged):
  25d72ad extract joined Wikidata sitelinks to articles by hashing the raw
  URL, so every percent-encoded or http seed URL got a sideless Wikidata-only
  shell battle and the real row lost its Q-id (327 shells on corpus3, ~26 on
  corpus2, e.g. the Beneventum shell 32). Model never double-counted (shells
  have no sides). Fix restored the DB to post_crawl and re-ran: 1653 battles,
  1652 joined, duplicate pairs 393 -> 33. b20fb6f resolve logged unresolved
  mentions by battle name only, so battles sharing a name (Akroinon 732/740,
  Beneventum 212/214/275 BC) lost them from missing_data_log (46 mentions);
  commander links were always slug-first. Same commit: ov-0105..0108
  (user-approved) unlink "Ming"/"Ming:" (the dynasty) from four Imjin sieges.
- **Review overrides (user-approved 2026-10-30), 888c8be:** ov-0109..0141
  (historiography review: merges, namesake splits, Perryville/Polotsk/
  Rhegium/Dunkirk 1944-45 indecisive, Ammunition Hill), ov-0160..0270 (111
  more split merges), ov-0300..0474 (175 wrong Wikidata links, e.g. G. H.
  Thomas on an engraver). 419 entries; extract 46/46 and resolve 366/366
  applied. Resolve after: resolution 0.984, link rate 0.758,
  probable_duplicate_generals 8 (was 36). Backup `..._post_resolve2.dump`.
  **Follow-up (code):** 242 of ~5,700 linked rows disagreed with the
  infobox's own wikilink; resolve should use that wikilink as the identity
  signal so future corpora do not need these overrides.
- **2026-11-03 state (resume here; record in handover-log §36).** Branch
  `wip/corpus3-seed` (w-c3seed) is now 288eee5: 8f86dad ov-0102 Q-ids,
  6eb2e10 **rotated era effect** (`era_effect: rotated|legacy|none`, fixed
  sigma_e, per-bin unlabelled levels sampled as coordinates; user chose it over
  dropping e, which the model review rejected because it would shift pre-1800
  sizes toward modern levels), 8ae2b27 **campaign exclusion** (`exclude_campaigns`,
  campaign sides logged to missing_data_log), 288eee5 identified-level and
  test fixes. Not merged, not pushed. Verified: unit 2393; integration
  `-k "reconcile or prior_check"` 82 passed (at 6eb2e10, not yet re-run with
  8ae2b27); reconcile model files pass after 288eee5. **corpus2 copy**
  (`general_war_c2_erarot`, run 133): rhat 1.008, 0 div, wb gap 0.045,
  Spearman 1.000 and median |dlog| 0.002 vs run 132's reconcile, so the fix
  changes nothing on corpus2. **corpus3 reconcile run 2 FAILED** (log
  `data/logs/corpus3_reconcile2_20261102.log`, backup
  `..._20261102_pre_reconcile2.dump`, no troop estimates written; casualties
  converged): era block now fine, but the noise leaks (modern infobox sigma
  0.72 vs prior 0.22, noise ratio 3.3, s_d 0.24, within-battle slope 0.10).
  Cause, from a read-only audit (scratchpad `c3audit/audit_out.txt`): only
  66 of 2136 sides have 2+ rows and the noise is learned from them; the worst
  are contaminated: (1) WWII strategic operations not typed campaign (Battle
  of France 150k vs 3.3M, Dnieper, Caucasus, Odessa Offensive, Dukla Pass,
  Debrecen, Nikopol, Rzhev, Belgorod-Kharkov, Donbass, Smolensk op, Voronezh,
  Vyazemsky op, Oder-Neisse); (2) an extract bug that glues ranges ("c. 450500"
  Roan's Tan Yard, "550800" Glasgow; a template dash is dropped; ~82
  candidate rows); (3) sub-detachment or mixed-scope totals (Corbach "400 to 500
  with Luckner", Fort Frontenac 2635/135, First Siege of Chaves two sieges,
  Tournai "20px" side); (4) broken side labels (Croix d'Orade, Ammunition Hill).
  **User decision 2026-11-02: fix the data first, no model change yet.**
  Classify roles done (573 imported; 2 lines had corrupted hashes and re-defer).
  The older diagnosis follows.
- **2026-10-31 state.** Reconcile on corpus3 FAILED to converge
  (run 1: max_rhat 2.54, min ESS 4.8, 0 divergences, 2107 sides fail MCSE;
  log `data/logs/corpus3_reconcile1_20261030.log`), wrote no estimates. Trace
  (scratch `recon_diag/a1.log` in the session scratchpad): chains stuck in
  different modes (tau_e per-chain 0.97/0.74/0.87/1.75, m0 chain 4 at 9.26),
  i.e. multimodality, not slow mixing. Nine troop_span_guard sides are
  campaign-scale entries (Battle of France, Caucasus, Dnieper, Debrecen,
  Okinawa). A Sonnet diagnosis worker runs on scratch DB
  `general_war_corpus3_recon` (copy of corpus3 after resolve, created
  2026-10-31; drop when done) testing longer tuning / excluding campaigns.
  User chose: diagnose first, no code change before its report.
  **Diagnosis done (troops fit, scratch `recon_diag/fit.py`, logs a1-d1):**
  baseline rhat 2.54; campaigns excluded 2.14; tune 4000 2.03; campaigns
  excluded + era_effect_prior_sd 0.3 1.75 (one chain frozen); 0 divergences
  throughout. Only the global era block fails (e[era_bin], z_era, tau_e,
  a_era, g_unlabelled, g_regime, offset_bar, m0, corpus_level). **Cause:**
  the P1 side prior `_troop_side_prior` (model.py ~918) adds `e = tau_e *
  z_era`, which on unlabelled rows (96%) is not separable from `a_era*era`
  + g_unlabelled, exactly what the module docstring N3 (~line 215) says must
  not be added. Corpus2 was too small to expose it. **Proposed fix
  (untested, needs bayesian-model-reviewer):** keep one era channel, drop
  e[era] for troops when a_era is fitted (or fix a_era at 0 and keep e).
  Validate on corpus2 (rhat 1.004 baseline), then corpus3 with campaigns
  excluded (battle_type=campaign filter in load_reports, secondary).
- **Classify (partial).** `classify --continue-past-deferred` on corpus3
  wrote 3,840 requests (`data/llm_requests/classify_20261030T132012Z.jsonl`):
  1,355 command-role (do not depend on reconcile) and 2,485 missingness
  (their prompt includes whether reconcile gave the side a troop estimate,
  so answer them only AFTER reconcile converges and classify is re-run).
  Role-only file `classify_roles_20261030T132012Z.jsonl`, 91 chunks under
  `data/llm_requests/chunks/classify_roles_20261030T132012Z/`; responses in
  `data/llm_responses/classify_roles_20261030T132012Z/`: chunks 1-52 done,
  validated and imported (780 answers, 0 errors); 575 requests left
  (chunks 53-91, ~39 Sonnet chunks). Resume with `split --skip-answered` on
  the roles file, answer, validate, import. Many answers are needs_review (thin excerpts,
  equal-split "unknown" defaults); classify reviewer should check Chaeronea
  (Philip/Alexander got equal "unknown"), Napoleon-credited battles, Khalid.
  ov-0102 (Waterloo) now names Uxbridge and Bülow by Q-id (Q335133, Q63023)
  because the merges changed their linked names; uncommitted on the branch.
- Dropped 2026-10-30 (user: "drop if not needed"), dumps kept in
  `data/backups/*_20261030_before_drop.dump`: `general_war_corpus` (pilot),
  `general_war_wt_p1_corpus`, `general_war_wt_exp_corpus`.
- **Next (2026-11-03):** (a) draft fixes for the contamination above and show
  the user before building any: retype the strategic operations as campaign
  (battle_type override), a Sonnet fix for the extract dash-glue bug (then
  re-run extract onward, so ask first), troop-scope overrides for the
  mixed-scope sides; twin-check overrides. (b) integration suite on 8ae2b27.
  (c) re-run reconcile on corpus3 (backup first), then bayesian-model-reviewer
  (the agent type was not registered this session: run general-purpose Opus
  with `.claude/agents/bayesian-model-reviewer.md`), plus one `era_effect:
  none` sensitivity run. (d) classify `--continue-past-deferred`, answer the
  ~2,485 missingness requests (ask usage %), re-run classify,
  historiography-reviewer (leads in handover-log §36), then impute onward.
  Scratch DBs created this session: `general_war_c2_erarot`,
  `general_war_erarot_it`; worktrees `w-erarot`, `w-campx` (branches
  merged into wip/corpus3-seed by cherry-pick/ff); `general_war_corpus3_recon`
  from last session. Ask before dropping any.
- Earlier next list: historiography review of extract+resolve (running; leads: placeholder
  side labels "side 1"/"side 2", Jean Moreau/David Baird/Jan de Winter
  unlinked, Blücher on an Austria side, Perryville strategic coding, Rhegium,
  non-battles like Clydebank Blitz) -> user approves overrides -> re-run
  resolve if any -> reconcile -> classify (~190 Sonnet chunks: **ask the
  user's usage % first**) -> impute -> model -> evaluate -> report -> both
  reviewers -> compare with run 132 (top 20, Montgomery, Khalid, Pyrrhus,
  largest component, compression).

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
Cleanup 2026-10-29: all `.claude/worktrees/*` removed (branches kept; only
`wip/noise-exp` is unmerged) and old session scratchpads deleted. The 26
scratch DBs (everything but `general_war` and `general_war_corpus2`) are
still there: the user asked to drop them, the permission classifier blocked
DROP DATABASE, so the user drops them or allows it.

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
