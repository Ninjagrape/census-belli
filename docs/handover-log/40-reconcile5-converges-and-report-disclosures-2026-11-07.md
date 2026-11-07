# §40. Corpus3 reconcile run 5 converges (centred battle effect), report disclosures built, missingness batch answered (2026-11-07)

## 1. State at the end of the session

- **main** = 18a6b4e (the user's commits of the extract/resolve/override fixes and §39 docs). Uncommitted on main: `config/overrides.yaml` (Dürenstein slug fix, section 5), `docs/handover/05-footguns.md`, this file, `handover.md`.
- **w-c3seed (wip/corpus3-seed, 2808819)**, uncommitted, all reviewed:
  - §39 work: Kalka supersede fix, opt-in low-rank, tune 4000 / draws 8000, A+B, ov-0600..0606.
  - New: frozen-chain diagnostic (`summarise.chain_health`, gate `write_min_chain_zd_sd` 0.5); reviewer MINORs 1 and 4; option C, opt-in `battle_effect_centred` (default false); rhat gate excludes the raw battle offsets `zb`/`b_raw` (`RHAT_GATE_EXCLUDED`, informational `max_rhat_raw_battle`); docstring on chain_health's gap; Jacobian test at two s_b; p1_recovery test mode-aware; contamination test runs with `era_effect: none`; Dürenstein override slug fix.
  - Verified: unit 2482 (after the gate fix), ruff, mypy. Model tests default mode: recovery 19 + 1 xfail, p1_recovery 13, contamination 3. C mode: recovery 19 + 1 xfail, p1_recovery 12 + the zb-name test since fixed. Integration `test_reconcile_p1_gates.py` 20 passed on `general_war_endash_it`.
- **worktree-agent-adf22892058598c5e** (off main 18a6b4e), uncommitted: report disclosures plan C phases 1 and 2 plus review fixes (section 7). Unit 2599, report integration 48 + 1 skip on `general_war_wt_disclose`, ruff, mypy. Rendered on a corpus2 copy: all 11 report gates pass.

## 2. Frozen-chain diagnostic and the 3-seed stage-tuning check (§39 steps 1-2)

Scratch `general_war_c3_diag`, 4 x 2000 after 4000 tune, ta 0.99, low-rank off. Scripts and pickles: session scratchpad `stagecheck/`.

| | s2 | s1 | s20260923 |
|---|---|---|---|
| divergences | 0 | 0 | 0 |
| frozen chain (zd sd) | none (0.997-0.999) | none | none |
| worst global rhat | 1.022 s_b | 1.054 s_b | 1.033 u_c_free |
| s_b ESS / 8000 (need 100) | 184 | 99 | 90 |

L_k agree across seeds within 0.001. Step 2 missed on s_b, so step 3 ran.

## 3. Option C (centred b, opt-in) and the reviewer

- `b_raw ~ StudentT(nu, 0, s_b)`, centred within era bin by the same projector as A; `b` deterministic; no `zb`. An exact reparameterisation of A (reviewer verified; logp test with the `n log s_b` Jacobian at two s_b).
- C on 3 seeds: 0 divergences, no frozen chain, s_b ESS 1357 / 1035 / 530, L_k spread <= 0.0016, means match default to about 0.002. Weakest then log(s_d/sigma_ibx), ESS 190-467 (corr -0.65, a mixing-speed issue, chains agree).
- C-mode recovery suite 19 + 1 xfail: the 2081-divergence history of centred b does not recur with A+B.
- bayesian-model-reviewer (run as a general Opus agent following `.claude/agents/bayesian-model-reviewer.md`; the agent type was not registered in this session): no BLOCKER/MAJOR. Use C for corpus3. MINORs, all fixed: (1) the rhat gate saw the prior-only per-bin mean of `zb`/`b_raw` (rhat up to 1.09), now excluded and logged separately; (2) chain_health is silent where no sides are marginalised (documented); (3) Jacobian test at two s_b.
- Low-rank comparison fit (default form, seed 1) ran but was not analysed; C made it moot.

## 4. Contamination model test

`test_merged_ranges_keep_sigma_near_truth_and_singletons_unshrunk` errored at the write gate (ESS 218-273). Pre-A+B code was already marginal (388 at seed 20261013); tune 4000 did not help (190-223). Cause: the synthetic corpus dates every battle 1800, one era bin, so corpus_level vs u_c is a prior-only split; A sharpens the identified sum and leaves a longer ridge. Dating across three bins did not help (two bins still empty). Fix, per reviewer: the test runs with `era_effect: none`. 3 passed.

## 5. Corpus3 re-run (step 5, user-approved)

- Backup `data/backups/general_war_corpus3_20261107_pre_rerun5.dump` (main checkout).
- extract, resolve, classify from w-c3seed. First pass: ov-0604/0605 (Dürenstein) unmatched, "battle not found": the stored slug is `Battle_of_D_renstein-c5194ffa8f`. Retargeted in both main and w-c3seed `config/overrides.yaml`, chain re-run: all 474 overrides apply (75 extract, 366 resolve, 33 classify), none unmatched. Footgun added.
- Exit 3 (awaiting LLM): 45,250 extract passages (accepted, infobox-only), 1 resolve (Sebastian at Thannuris, answered NEW_ENTITY), 2,495 classify.
- **Reconcile run 5** (`--set battle_effect_centred=true`), 1 h 40 min: troops max rhat 1.006, min ESS 1517, 0 divergences; casualties rhat 1.006, ESS 3891. Gates: noise 1.134 (<= 1.5; run 3 was 1.514), within-battle 0.85, side deviation 0.64, wb vs reliability 0.008, estimates populated 1.0, prior predictive 0.045. **One failure: reports_honoured = 1**, Rovereto (1796) Austrian side 1822: infobox "20,000 total" was tagged engaged and "13,695 available" tagged available (inverted), estimate 9,642 [5,411, 17,336]. Troop estimates were written (run 5).
- **`era_effect: none` sensitivity** (copy `general_war_c3_eranone`, run 6, C on): noise 1.131 (run 5 1.134), within-battle 0.850 (0.850), side deviation 0.643 (0.641), reports_honoured the same Rovereto side. Troop ESS 338 < 400, so nothing written there. `neutral_all_sides_level_in_band` fails (1.0) without the era effect, so the era effect carries the level placement; the noise and ratio results do not depend on it.
- User call: produce the ranking on run 5 now and fold a Rovereto fix into the next refit. Historiography-reviewer draft (not applied, needs sign-off): ov-0607 `troop_strength`, target `{battle: Battle_of_Rovereto-6add7f8553, side: Austria}`, `value: {point: 13695}`, `scope: available`; the 20,000 is Davidovich's whole corps (19,555 in the article body). Smith 1998 pp. 122-123 and Chandler p. 97 not checked.

## 5b. First corpus3 ranking (provisional)

classify, impute, model, evaluate, report on reconcile run 5 (log `data/logs/corpus3_rank1_classify_to_report_20261107.log` in w-c3seed; outputs in w-c3seed `data/processed/report/`). Every stage succeeded (impute 35 min, model 1.7 h, evaluate 5 h). Warning-level checks failed: distributional_check (4 eras), posterior_predictive_category_check (defeat, victory), fmi_top_generals (1 general > 0.3; raise n_imputations to 40), top_generals_cross_component (29 of 50). Top 10: Napoleon, Khalid, Caesar, Wellington, Hannibal, Forrest, Alexander, Lee, Frederick II, Beauregard. Not reviewed yet; Burnside #20 and Beauregard #10 look suspect. Provisional: 66 classify answers were imported after the chain started, Rovereto unfixed, and w-c3seed's report code predates the disclosures work.

## 6. Missingness batch (step 6)

- 2,495 classify requests answered by Sonnet llm-batch-workers, 5 or 10 chunks per worker (fewer coordinator turns than one per chunk). A 5-hour usage limit stopped 11 workers mid-batch; they were resumed with SendMessage after the reset.
- Validation found one missing hash (chunk 054) and a worker reported marking the US side at Okinawa and the Bulge mnar because the opponent was authoritarian. Chunks 054, 161 and 165 were re-run fresh with "classify the row's own side from the row text"; both rows are now mar. Imported 2,495 + 1 resolve, 0 errors.
- Answers are overwhelmingly mar; mnar on Axis/Vichy sides after 1914 and a few defeated sides; a handful of mcar.
- The post-reconcile classify deferred 66 new requests; answered and imported (0 errors). The ranking chain started before they were imported, so the run-5 ranking lacks them.

## 7. Report disclosures (plan C), worktree agent-adf22892058598c5e

- Phase 1: data-driven disclosure sentences (`reporters/coverage.py`, `disclosures.py`), headline rank 1..N with the stored position kept, gate `disclosures_match_db`.
- Phase 2: six "Data coverage and caveats" panels (`coverage_series.py`, `coverage_panels.py`, `coverage_figures.py`, `explorer_coverage.py`), gate `coverage_figures_written`; report eras `REPORT_ERAS` (500/1500/1789/1816/1914) are separate from the model's era bins.
- Sonnet review: 4 HIGH (tautological explorer check, gate only re-ran the loader, rank inconsistency across forest plot and explorer, no SQL-error degradation) and 6 MEDIUM; all fixed. The gate now recounts credited sides, rule_single sides, headline generals and credited battles with independent SQL. `bound_only` sides (15 on corpus2) get their own clause (the old code counted them as multi-source: corpus2 has 14 multi-source sides, not 29).
- corpus2 copy `general_war_c2_report`: all 11 report gates pass. Known nit: "1 side rest" agreement in the bound-only and other-method sentences.

## 8. Scratch DBs created this session

`general_war_wt_disclose` (report integration), `general_war_c2_report` (corpus2 copy for report renders), `general_war_c3_eranone` (corpus3 copy, era_effect none sensitivity run, from scratch cwd `eranone/` in the session scratchpad). Ask before dropping.
