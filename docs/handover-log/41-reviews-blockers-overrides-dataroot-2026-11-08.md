# §41. Reviews of the first corpus3 ranking, reconcile floor and lone-branch fixes, ov-0607..0616, data root (2026-11-08)

## 1. State at the end of the session

- **main** = 9ee03a4 (user committed §40 work). Uncommitted: `config/overrides.yaml` (ov-0607..0616, 481 entries), this file, `handover.md`.
- **w-c3seed (wip/corpus3-seed, 5199c34)**, uncommitted:
  - Report disclosures cherry-picked from ee1c9de (`git cherry-pick -n`, then unstaged) plus the verb-agreement fix (section 4). Verified on w-c3seed: unit 2549, ruff, mypy 165, report integration 24/24 on `general_war_wt_disclose`.
  - Reconcile: overrides exempt from the troop floor; lone-branch promotion; lone-branch offset (section 3; handover §000 says whether the offset build finished and was gated).
  - `config/overrides.yaml` ov-0607..0616 (484 entries; each target matches one corpus3 row).
  - `tests/integration/test_reconcile.py` diagnostics key set updated for 5199c34's chain-health keys (it failed on the committed code). Integration 34/34 on `general_war_endash_it`.
- **worktree-agent-adf22892058598c5e** (ee1c9de) holds the agreement fix uncommitted too; it is now redundant with w-c3seed.
- **worktree-agent-a512e2a4e3d0264a4** (off main 9ee03a4): the data-root setting (section 5), uncommitted, 27 files.
- The refit has NOT run yet.

## 2. Reviews of reconcile run 5 and ranking run 28

**bayesian-model-reviewer** (general Opus agent with the spec): no BLOCKER. Run-5 troop estimates fit for purpose; ranking fit to refit. Keep `n_imputations` 20 (FMI fires only on Yi Sun-sin; MC SE 0.14 vs 2.72). MAJORs:
1. rule_single gives w = 1 to a sole listed commander even when he was theatre/department level (Beauregard's Charleston and Bermuda Hundred actions, Burnside at Fort Sanders); 72-94% of Forrest/Beauregard/Burnside xWAR is rule_single, and no sensitivity variant touches rule_single. Needs a top-50 audit and a rule_single-downweight variant.
2. Every component's skill level is prior-set (component means about 0); 29 of top 50 outside the main component. Headline caveat; show within-component ranks.
MINORs: PPC category check measures listing order (side A wins 0.676; fold onto favourite); held-out accuracy gate 0.55 is below the trivial side-A baseline; FMI and distributional gates miscalibrated (make informational); add an ancient_claim +-1 sd variant; xWAR positive in defeats needs explorer wording; reconcile `reports_honoured` sees only 23 sides; about 110 classify answers rejected for rank-0 conflicts, 526 sides need review.

**historiography-reviewer**: Burnside, Beauregard, Forrest placements are method consequences, not data errors, except: Forrest wrongly linked at Second Mesilla ("Colonel. Steel"), Fort Sumter's 85 defenders dropped by the 100 floor and imputed at 1-3k. BLOCKERs (both reconcile): the floor drops 27 real small sides; single-arm figures are always dropped even when they are the whole force (Brice's Cross Roads 3,500 cavalry). MAJORs: campaign rows double count constituent battles (Peninsula, Valley, Chattanooga, Seven Days, Stoneman's raid credits Lee 1.0; about 70 campaign/operation/war rows overall); one comparison per battle regardless of size (Burnside above Grant and Lee); Khalid's record is a one-sided futuh tradition (disclose); Marcellus rests on Livy's Nola sorties. MINORs: 229 battles carry a 1 January year-precision dummy date (extract date-parser bug; Fredericksburg among them); Fort Wagner and Roanoke Island typed naval; "James pp" junk commander at the Crater; Chaeronea credits Alexander 0.5; Antietam has no Burnside row (no override kind can add a commander); Knoxville/Fort Sanders count twice. Rovereto ov-0607 endorsed from the article (Boycott-Brown pp. 418-419).

## 3. User decisions and the reconcile changes

- ov-0607 approved; drafts ov-0608 (Mesilla unlink), ov-0609 (Fort Sumter 85), ov-0610..0614 (campaign retypes), ov-0615 (Fredericksburg date), ov-0616 (Fort Wagner field) approved. Added to main and w-c3seed.
- Disclosures reach corpus3 by cherry-pick only (not a branch merge), so the refit does not also pick up main's cloud extract/resolve fixes.
- Fix both blockers before one refit:
  - troop_strength overrides bypass `min_plausible_troops` (raw rows keep the floor; an info log notes an override below it).
  - A side with no total report whose non-total rows name one personnel branch (infantry, cavalry, irregular) has them promoted to the total (55 corpus3 sides, about 47 above the floor). artillery/armour/air rows (equipment counts) do not block promotion; `other` does; two personnel branches do not promote. Counter `promoted_lone_branch`.
  - Model reviewer on the promotion: no BLOCKER; MAJOR that a lone branch figure runs below the total (corpus3 sides with both: median ln(total/branch) 1.08, cavalry 2.39, irregular 0.62, infantry 0.24). User chose an upward offset: shared `d_lone_branch ~ TruncatedNormal(log 1.5, 0.6, lower 0)` subtracted from promoted rows' location, prior-only, created only when a row is flagged. "At least" bounds were rejected because a bound-only side has no likelihood in the export.

- **Offset form changed after the gate review.** The shared scalar was identified by the data through the battle effect shared with the opponent (synthetic: posterior 1.10 +- 0.10 against prior 0.405 +- 0.6), which pulls promoted sides toward opponent size and partly assumes the force ratio being measured. User chose per-side prior-only: `d_lone_branch[k]`, dims `lone_side`, one per promoted side. Synthetic (200 battles, 40 flagged, true d 0.8, centred b, nutpie): 0 divergences, min ESS 540 tail, rhat <= 1.005, posterior mean 0.725 vs prior mean 0.659 (contraction 0.15, from the hierarchical side prior, not a new channel), estimate - raw log 0.751, 95% width 1.505 flagged vs 0.867 unflagged. Script: session scratchpad `lone_fit_per_side.py`.
- Gate review B1 (fixed): `reports_honoured` rebuilt row locations without the offset; it now subtracts the side's d per draw, as do `_row_eta_and_var`, `_side_offsets` and `_identified_level_stats`. `offset_bar` subtracts the row mean of d_k * flag (unit-Jacobian change of coordinates, reviewer verified). Diagnostics: `promoted_lone_branch_sides` and `d_lone_branch` {n_sides, mean_of_means, mean_sd, mean_contraction}. Re-gate: GO.
- Also: `_unfillable_reason` says "only implausible or non-personnel figures" when a promoted side is floored; `reconcile_override_below_floor` info log; an override below the floor is kept, not an error (behaviour change).
- Verified: unit 2563; integration test_reconcile_load 20, p1_gates 20, clearing 4, test_reconcile 34 (`general_war_endash_it`); model p1_recovery 13, contamination 3, neutral 5+1 xfail and recovery 19+1 xfail (the last two on the shared-scalar version; the per-side change leaves unflagged fits unchanged). ruff, mypy 165.

## 3b. Refit 7

Backup `general_war_corpus3_20261109_pre_rerun7.dump` (3.4 MB, 17 table-data entries). Started 2026-11-09 from w-c3seed via scratchpad `refit7.sh` (three chained orchestrator calls, classify before and after reconcile). Log `w-c3seed/data/logs/corpus3_refit7_20261109.log`. ALL DONE 2026-11-10 04:24 (chains rc 3, 0, 3; the 3s are deferrals: extract 45,250, resolve 2, classify 83 then 89). Paused about 1h on 2026-11-09 by NtSuspendProcess for the user; no effect on results.

- Reconcile run 29: 0 divergences, max rhat 1.009, min ESS bulk 942; noise_not_leaking 1.16 (<= 1.5); reports_honoured 0 misses (Rovereto fixed). promoted_lone_branch 87 reports, 62 troop sides carry `d_lone_branch` (mean of means 0.659, mean sd 0.398, mean contraction 0.093: prior-dominated, as designed). excluded_implausible_small 85; 2 troop-span warnings (Ruspina Pompeians 1.6k-9.6k, Shepherdstown CSA 600-2,000); 2 low-precision casualty sides (731, 1551).
- Model run 52 (imputation run 30, 20 rounds, 1,535 battles, 2,874 generals): gate warnings unchanged in kind: FMI over 0.3 for 1 top general, 29 of top 50 outside the main component, PPC category check (defeat, victory). 234 classify answers rejected.
- Evaluate: 5/5 gates; ranking_stability top-20 overlap 15-20 across all variants (observed_only lowest at 15).
- Report: 496 headline rows; disclosures_match_db, coverage_figures_written, evaluation_current, top_rankings_have_intervals all pass.
- Top 20: Napoleon, Khalid, Caesar, Wellington, Hannibal, Forrest (6), Frederick II, Alexander, Soult, Beauregard (10), Massena, Lee (12), Suchet, Sheridan, Yi Sun-sin, Stuart, Ney, Jackson, Burnside (19), Marcellus. Grant 25. Forrest/Beauregard/Burnside barely move: expected, since their placement is rule_single (MAJOR 1 in section 2), which this refit did not touch.
- Not yet done: both reviewers on run 29/52; records.py:465 docstring.

### 3c. User concern: ACW bias (2026-11-10)

The user does not believe Confederate (and ACW) generals are that much better than the rest of history and suspects the ranking is biased. Evidence: 17 of the top 50 are ACW (first_year 1855-1865), 1 is WW2 (Montgomery 42). ACW is 472 of 1,651 credited battles (29%) from 5 years; WW2 234; all BCE 116. WW2 credit is split (Zhukov 8 battles, 2.95 effective; Eisenhower 0.05); Agrippa 4 battles, rank 60. Likely mechanisms: total WAR sums over battles (volume), dense Wikipedia coverage of small ACW actions (raids, skirmishes, cavalry fights counted as battles), rule_single full credit, and possibly the troop-ratio expectation crediting outnumbered wins. Both reviewers must address this next session, and the fix should be a design decision, not a narrative caveat. Candidate diagnostics: ACW battles by size/type (how many are under ~5k a side or raids), rank by war_per_battle, a variant down-weighting battles by size or capping per-era battle counts, and whether ACW outnumbered-wins are over-credited relative to other eras.

## 4. Report disclosures agreement nit

`_verb` helper in `pipeline/reporters/disclosures.py`; fixed "1 side rest/carry", "1 have/rest", plus three more clauses of the same kind (in-scope battles excluded, ranks in gaps, undated generals). Tests for counts 1 and 2.

## 5. Data root

User approved: `GENERAL_WAR_DATA_ROOT` (default `data`); `pipeline/paths.py` `data_root()` / `data_path()`, read at call time; every `data/...` default derives from it; YAML path defaults removed from agents/*.yaml and config/model_default.yaml; two report gates use `report_under_root`; `data/backups/` stays fixed. Corpus3 will live at `data/corpus3/` and run with `GENERAL_WAR_DATA_ROOT=data/corpus3`. Built in worktree agent-a512e2a4e3d0264a4 off main: unit 2578, ruff, mypy 162 (8 formatting-only files reverted with the user's OK). Not yet code-reviewed. When merged with wip/corpus3-seed, expect a conflict on agents/report.yaml (disclosures also touch it).

## 6. Known test-harness issue

`tests/integration/test_resolve_overrides.py` (5) and `test_resolve_rerun.py` (1) error instead of skipping when DATABASE_URL is unset (their `engine` fixture raises DatabaseConfigError). Pre-existing; fix the fixture to skip.
