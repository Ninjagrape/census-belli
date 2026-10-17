# Handover log §27. "Fix all outstanding items", 2026-10-14 to 2026-10-16 (was handover §0)

Part of the session log; the index is `docs/handover-log/README.md`.

Worked on throwaway local branches (`wip/*`) in worktrees, squashed onto `main` as
commits 3c4b22a..015bed7 on 2026-10-16. Branch and worktree instructions below are
historical.

Scope chosen by the user: every code item that needs no user decision, plus
the resolve (146) and classify (~254) LLM batches, then a corpus refit.
Extract's 14,944 requests, cost/design decisions and stretch goals wait.

**Latest status (2026-10-16 ~00:30; read this first, the rest of §0 is
detail). Next session, in order:**
1. **DONE 2026-10-16, merged as 87952b4.** Worker-verified: ruff, mypy,
   unit 1736, integration (reconcile/impute/model/evaluate) 93 passed 0
   skipped, all six model files pass (recovery 20 incl. 1 pre-existing
   strict xfail pinned to `censored`). Neutral gap 0.027 under the original
   0.06 tolerance. ExGaussian is now the default: RMSE 0.304 vs censored
   2.143 on lower bounds, 0.329 vs 1.924 on upper; `lambda_lower`
   contraction 0.73. `test_unmerged_range_pairs_inflate_sigma` pins
   `censored` (it is a positive control for the censored-pair pathology).
   The original brief, for reference:
   IG(3, 4/3) noise prior (mode-preserving; IG(3,2) over-shrank two-report
   sides and is what moved the neutral test: rewrite that test's comment, set
   tolerance back to 0.06 if the gap allows); Latin-hypercube theta
   selection; **ExGaussian bounds as the default** only with a lower-bound
   recovery case, a logp-direction unit test and `lambda_*` diagnostics;
   per-side casualty ratio random effect; realised-WAR split
   `w_i*(y-E_model) + (w_i/W)*xwar_side` for partial camps;
   `designs_equivalent` share arrays; precision-weighted `_side_offsets`;
   disclosures (camp share fill from imputed troops, era table
   approximations, S4 partial pooling). Read `git -C ... diff` against that
   list, finish, then acceptance: ruff, mypy, unit, integration
   `-k "reconcile or impute or model or evaluate"` on `general_war_wt_reconcile`,
   and all five model files one per command. Commit on `wip/recon2`, merge
   into `wip/outstanding`.
2. **Corpus (2026-10-16 progress)**. Resolve fixes merged into
   `wip/outstanding` (ed99498, 7cd257c): candidates were capped at 8 in raw
   Q-id order before any ranking, so Julius Caesar (Q1048, 11th of 11) and
   Hannibal Barca (Q36456, 15th) never reached the LLM; now date-gated and
   ranked (exact name, date-confirmed, score, sitelinks, Q-id) before the cap.
   Template braces `{{` leaked into resolve prompts (fixed). Resolve did not
   converge across re-runs because SPARQL `GROUP_CONCAT` order is arbitrary
   and occupations/citizenship were printed unsorted into hashed prompts
   (fixed: sorted; test `tests/integration/test_resolve_rerun.py`). After
   answering the stable batch (`resolve_20261016T003031Z.jsonl`, 88, imported)
   resolve **converges: 0 deferred, 716 linked, 168 new, 42 unresolved**
   (was 614 / 182); Caesar, Hannibal Barca, Napoleon, Alexander linked.
   Earlier batch files (`resolve_20261015T125258Z`, `..001100Z`, `..001555Z`,
   `..001928Z`, `..002241Z`) are superseded; their answers are in `llm_calls`
   but most hashes no longer occur.
   **Fixed and merged (8bf6aba)**: the wikitext path (first real use since
   crawl began saving `.wikitext`) dropped whole infobox fields wrapped in
   unknown templates (`blist`, `ubli`, `ublist`, `unbulletedlist`,
   `Indented plainlist`, `bdl|wrap=`, `udl`, `center`, `ill`, `lang`,
   `approximately`, `DOW`, `Collapsible list bullets=yes`) and leaked
   `[[File:..|border|25px]]` as "border|25px Robert E. Lee" (and parsed 25px
   as a headcount). Now File/Image links are removed, flag templates render
   the country or nothing; over 211 battles wikitext equals HTML on troops
   (1323) and commanders (1617), better on 4 casualty splits. Resolve marks
   UNCERTAIN as method `llm_uncertain` + note `UNCERTAIN_SPLIT_NOTE`, and
   `no_duplicate_generals` excludes those pairs (counted in the
   `probable_duplicate_generals` warning). Worker-verified: unit 1773,
   integration extract/resolve 71, 0 skipped.
   **Last corpus run (2026-10-16 01:37 UTC)**: `--stages
   extract,resolve,classify --continue-past-deferred`, log
   `data/logs/corpus2_extract_resolve_classify_20261016.log`, EXIT=3.
   Extract: 236 battles, **1305 troop reports** (regression gone), commander
   mentions 1613 (was 1479: list templates recovered), 14,705 extract
   requests deferred (not in scope). Resolve: 1002 groups, 836 identities,
   727 linked, 163 new, 112 unresolved, **70 new requests**
   (`data/llm_requests/resolve_20261016T013713Z.jsonl`, from the recovered
   commanders). Classify: 579 sides, 562 requests deferred
   (`classify_20261016T013735Z.jsonl`), but these depend on resolve's links,
   so they are **stale once the 70 resolve answers land**.
   **Next session, in order**: (1) `/process-llm-batch` the 70 resolve
   requests (5 Haiku chunks; the llm-batch-worker can stall on a slow API:
   relaunch a chunk whose worker reports "no progress for 600s"); (2)
   `--stages resolve,classify --continue-past-deferred` (resolve should defer
   0; if the duplicate gate fails, the new UNCERTAIN note only exists on rows
   written after 8bf6aba, so one resolve re-run clears it); (3)
   `/process-llm-batch` the new classify file on Sonnet (~38 chunks, the
   largest remaining spend: check the user's usage first; the user caps a
   session at 70%); (4) classify, then reconcile, impute, model, evaluate,
   report; (5) investigate classify's `battles_typed: 0` on the corpus
   (campaign typing should fire on wars like the Six-Day War; an existing
   non-unknown type is never rewritten, so check whether old types block it). Classify reported `battles_typed: 0`: check whether campaign
   typing fires on the corpus. Run from the `wip/outstanding` worktree with
   `DATABASE_URL=...general_war_corpus2`, logs to `data/logs/`.
   Also seen: resolve groups carry stray modern years (Caesar's includes
   1948, Hannibal's 2015; likely access dates), which keeps modern namesakes
   through the date gate; and Lefebvre, Moreau, Bernadotte, Bülow, Wŏn Kyun
   are not returned by exact label lookup at all (label search, deferred by
   §9.3).
3. Squash `wip/outstanding` onto `main` uncommitted (procedure below), tidy
   this file and `docs/todo/*`, hand the user a commit message.

**State verified by the lead on 2026-10-15/16:**
- `wip/outstanding` (79d2c9a, every wave-1, wave-2 and review fix except the
  w-recon2 cleanup): ruff clean, mypy 151 files, non-model suite **1984
  passed, 0 failed, 0 skipped** (17 min, DB `general_war_wt_infra`); model
  suites all pass: evaluate_cv 4, impute_recovery 5, model_recovery 7,
  reconcile_contamination 3, reconcile_neutral 5, reconcile_recovery 18 + 1
  expected xfail (13 min).
- Corpus run on `general_war_corpus2` from the `wip/outstanding` code:
  `--stages crawl,extract,resolve --continue-past-deferred --set
  seed_path=config/sources_seed_arsht.yaml --set citation_depth=0`
  (citations not re-crawled; crawl back-filled `.wikitext`). Log
  `data/logs/corpus2_crawl_extract_resolve_20261015.log`, EXIT=3 (deferred,
  expected), 683 s. Extract: 236 battles, 1179 troop reports, 14,721 LLM
  requests deferred (`extract_20261015T125033Z.jsonl`). Resolve: 926 groups,
  692 identities, 614 linked, 130 new, 182 unresolved, 11 non-person
  filtered, 21 duplicate battle pairs reported, 140 LLM requests deferred.
- `general_war_corpus2` had **no alembic_version** (built by
  `apply_schema()`); `alembic upgrade head` ran every migration from the
  baseline. All are IF-NOT-EXISTS guarded, data intact (236 battles, 420
  sides, 54 runs), now stamped `b8c9d0e1f2a3`.
- The `wip/outstanding` worktree's `data` is a **directory junction** to the
  main checkout's `data/`. Remove it with `cmd /c rmdir <worktree>\data`
  BEFORE `git worktree remove`, or the removal may delete the real data.
- Bayesian reviewer findings left open by choice: L8 (z_skill carries the
  common shift raw skill was dropped from the gate for); L-h (count sides
  whose weak classification differs between raw and residual covariance on
  the corpus).

**Baseline on main before any change**: ruff clean, mypy 147 files clean,
1645 passed 0 skipped (17.5 min) against live Postgres, alembic head
a7b8c9d0e1f2.

**Where the work is (local branches, never pushed; the user approved a
throwaway WIP branch).** Auto mode blocked workers from syncing uncommitted
work between worktrees, so:

- `wip/outstanding` (worktree `.claude/worktrees/agent-a2c6c6ca032e26c6c`) is
  the integration branch, based on local `main` 6c91f1a. Merged so far:
  - 2318f53 migration `b8c9d0e1f2a3` (+ schema.sql): `battles.date_circa`,
    `battle_commanders.infobox_order`, `battle_type` value `campaign`,
    `general_skill_estimates.war_ci_lo/war_ci_hi/p_above_next_rank`;
    `campaign` added to `BATTLE_TYPES`.
  - 27281af extract: circa dates written to `date_circa` ("late 1250s BC" ->
    decade, 1250 BC; "X or Y" -> earlier year; "1-3 May c. 1450 BC" fixed);
    `part_of` -> `wars`/`campaigns` rows (`pipeline/extractors/wars.py`, first
    war-like / first campaign-like entry, select-then-insert, COALESCE links);
    `infobox_order` from extract through resolve's upsert (lowest order per
    person and side). Worker-verified: unit 1439, integration 66, 0 skipped.
    Open: `resolvers/mentions.py` uses `part_of[0]` (can be a Q-id) as war
    context; no test of the circa merge rule.
  - f6832d9 infra: removed cmdstanpy, SPARQLWrapper, polars, click, tqdm
    (nothing imported them; **CLAUDE.md's tech stack and the CmdStanPy hint
    at `agents/model.yaml:225` are now stale**), declared numpy, lock
    regenerated; `log_all_llm_calls` and `posterior_check` removed; evaluate
    writes gate `severity` (report already preferred it); `citation_depth`
    must be 0 or 1; `external_datasets` stanza removed; crawl fetches
    `https://en.wikipedia.org/wiki/<title>?action=raw` behind
    `fetch_wikitext` (robots disallows `/w/` and `/api/`, allows `/wiki/`;
    goes through robots-checked `fetch`), saved as `<stem>.wikitext`, back-
    fills earlier HTML-only articles. Worker-verified: unit 1426,
    integration 75, 0 skipped. No live crawl run.
  - `citation_line_malformed` was already fixed by ca9d804 (cause:
    `splitlines()` splits on NEL/LS/PS inside JSON; 11,337 = 11,170 NEL + 37
    LS + 30 PS).
  - Also merged (2026-10-15), each worker-verified (unit + integration on its
    own `general_war_wt_*` DB, 0 skipped):
    - classify (7b59e56): infobox order + unresolved count in the prompt,
      per-side `is_defender` from the LLM, 7 few-shots, LLM missingness path,
      casualty and post-1914 suppression rules, `campaign`/amphibious/
      guerrilla, no `siege_defensive`, Badr/Donelson, terrain, proportional
      unknown gate.
    - model (998f8e6): `war_ci_lo/hi`, `p_above_next_rank`, `war_draws` in
      `pooled_draws.npz`, report uses them (normal approx only as fallback),
      convergence gated on z_skill/centred skill. Recovery corpus doubled;
      worst Spearman across 4 seeds 0.825 vs 0.8 (thin). At 400 battles/era
      the joint-latent comparator's beta_force sits ~0.15 below truth.
    - reconcile (0524f1e): npz `g_unlabelled`, `a_era`, `offset_bar`,
      `side_era`, `side_unlabelled_share`; identified-width gates;
      `n_identifying_sides` via union-find; prior predictive check.
    - evaluate (d91dcbc): `g_unlabelled_plus/minus_1sd` variants
      (`regime_shift`), `not_evaluable` status.
    - impute (7dd7392): shared reconcile error propagated per round.
    - 2f0f4da orchestrator gate tests updated for classify's 4th gate (two
      more tests in that file pass only via the stub's default 0: tidy).
    - model2 (269df79): campaigns excluded, defender term, force slope by
      group, two-camp merge of 3+ side battles.
  - Merged-branch suite before model2/evaluate/impute: 1811 passed, 3 failed
    (the gate tests, since fixed). Ruff: two E501 in
    tests/unit/test_reconcile_summarise.py (being fixed in w-recon2).
- **Reviews (2026-10-15) and fixes in progress**, each in its own worktree
  fast-forwarded to `wip/outstanding`, uncommitted (check `git -C
  .claude/worktrees/w-<name> status`, re-run acceptance, commit, merge):
  - `w-recon2`: wave-2 reconcile structure (S3, S4, N1, IG alpha 3, casualty
    ratio variance, `bound_likelihood: exgaussian` option, N3 docstring,
    the two E501s).
  - `w-classify`: historiography review fixes. Few-shots: Mutina (Hirtius,
    Octavian, D. Brutus; Pansa was at Forum Gallorum), Cannae consuls equal
    co-field commanders, Salamis 0.6/0.4 and is_defender null, Mohi Batu
    supreme_commander, Königgrätz wording + Herwarth + scoped staff-officer
    rule, Munda 0.9/0.05/0.05. Rules: campaign span exempts "Battle of"/
    "Siege of" and rises to 180 days (Verdun, Stalingrad were being dropped),
    Invasion/Conquest titles; density-based guerrilla/amphibious; regime list
    anchored on polity and extended; terrain false positives.
  - `w-impute`: Bayesian review H1 (HIGH): the loading assumed reconcile's
    m0 fixed, but m0 = corpus_level - offset_bar moves with g, so the shift
    must be B + P^-1 (B - 1 c^T)/s0^2 with c = dm0/dtheta (export m0 draws);
    otherwise unequally documented sides get a spurious shared ratio shift
    that attenuates beta_force. Also M1 imputed sides co-move via an era
    loading table, M2 per-side identified offset, L4 stratified theta draws,
    L5 weakness on raw covariance, L6 prior check on the neutral fit, L7.
  - `w-evaluate`: M4 report recomputes p_above_next_rank over the eligible
    (headline) ordering from `war_draws`; L1-L3; M3 evaluate variant gives
    sides without a slope the era-mean slope, wording fixed.
  - `w-model2`: model-terms review. H1 (blocker): the 2-group force
    hierarchy made beta_force prior-dominated and silently loosened the
    circularity gate; replace with land slope `beta_force` + fixed
    `delta_water_air`. M1 restore defender complement-fill when one flag is
    set. M2 coalition camps carry the unnamed share as a latent component.
    M3 beta_def diagnostics; L1 campaigns not "uncredited"; L2; L4; L6.
  - Not yet decided: L8 (z_skill carries the same common shift raw skill
    was dropped for).

**Merge procedure**: in a finished worktree `git add -A && git commit` on
`wip/<name>`, then `git merge wip/<name>` inside the `wip/outstanding`
worktree. At the end bring everything to `main` uncommitted (`git merge
--squash wip/outstanding` on main, then `git reset` so nothing is staged),
remove the wip worktrees and branches, and the user commits.

**Remaining plan** (designs decided by the lead, not built):

- Wave 2: the impute, model and evaluate designs are merged (list above);
  the reconcile structure items (S3, S4, N1, InverseGamma alpha 3, casualty
  ratio variance, `bound_likelihood`) are in `w-recon2`. N3 (a period mean and
  `a_era` coincide on unlabelled rows) needs no code; record it as a
  disclosure.
- Wave 3:
  - `bayesian-model-reviewer` on reconcile, impute and model.
  - `historiography-reviewer` on the classify few-shots and rules.
  - An end-to-end fixture test from crawl through model.
  - The full suite run by a Haiku verifier on the merged tree.
- Wave 4, corpus `general_war_corpus2`:
  - Extract and resolve, then `/process-llm-batch` on the new resolve file,
    then resolve again.
  - Classify, its batch, then classify again.
  - Reconcile, impute, model, evaluate and report.
  - Ask the user if auto mode blocks DB access.

**Gotchas found this session**:
- `origin` exists. Agent `isolation: worktree` branches from `origin/main`,
  which lags local `main`. Create worktrees yourself from the right base and
  point workers at them, as this session did.
- Auto mode refuses cross-worktree patch copying and some compound
  `git ... && createdb` commands. The `general_war_wt_*` databases already
  exist.
- A full non-model suite takes ~17.5 min now.

**Left for the user** (out of scope): `.env`; CI on the existing `origin`
remote; the billed API smoke test; the
hand-labelled gold set; web_secondary keep/cut; which arm the calibration
gate sits on; `pre-commit install`; the stretch goals.
