# 34. P1 implemented, reviewed and run on a corpus copy (2026-10-24/25, overnight)

Branch `wip/troop-p1` (worktree `w-p1`), local only, not merged:
e9bf6ef (failing tests) -> a5c017b (P1) -> b671615 (merge `wip/troop-scope`,
with the 39 approved overrides f86e464) -> 07f529c (manual_entry bias pinned
to 0, impute pair gate fails without a reference slope, duplicate
`side_prior_nu` removed). Raw numbers:
`C:\Users\LeeMo\AppData\Local\Temp\claude\c--Users-LeeMo-Documents-GitHub-census-belli\fdd389e4-6c51-463f-8950-8d99329bc30d\scratchpad\sensitivity_raw.md`
(session scratchpad, may be cleaned; the tables below are the record).

## Verification

- b671615: non-model suite 2705 passed, 0 failed; ruff, mypy (159 files)
  clean; one head d0e1f2a3b4c5; model files test_reconcile_p1_recovery (10),
  test_reconcile_recovery (17 + 1 xfail), contamination (3), neutral (5),
  test_impute_recovery (5) green.
- 07f529c: unit 2341 passed; test_reconcile (34), test_reconcile_p1_gates
  (18), test_troop_strength_override (11), test_impute (19), p1 recovery (10)
  green. **Full suite not re-run on 07f529c.** test_impute's fixture got a
  guessed `within_battle_slope: 0.9`.
- Corpus copy `general_war_wt_p1_corpus` (restore of corpus2, backup
  `data/backups/general_war_corpus2_20261024_pre_p1.dump`): extract (36
  overrides incl. 10 troop_strength, unmatched 0; Jena split -> battles 237,
  238), resolve (50 commander_link, 4 new LLM answers, gates pass, duplicate
  warning 6 as before), reconcile, classify (25 new Sonnet answers, 9 flagged
  needs_review: Dresden AUT, Leipzig RUS, Waterloo UK, Arcole, Austerlitz,
  Kursk, Granicus, Toulon, Bautzen), impute, model. Variants in
  `general_war_wt_p1_{s015,s030,sfree}` (worktrees `w-s015/w-s030/w-sfree`,
  `w-p1run` for the base). All stages `--continue-past-deferred`.

## Sensitivity (noise median modern/ancient)

| | 0.15/0.27 | **0.22/0.40 (base)** | 0.30/0.55 | free (inv-gamma) |
|---|---|---|---|---|
| divergences / max rhat | 0 / 1.003 | 0 / 1.005 | 0 / 1.002 | 0 / 1.005 |
| reconcile wb | 0.716 | 0.627 | 0.532 | 0.340 |
| implied reliability | 0.772 | 0.673 | 0.557 | 0.304 |
| singleton slope / spread | .936/.942 | .921/.929 | .903/.914 | .858/.879 |
| s_d / s_b | .46/.68 | .43/.68 | .40/.68 | .33/.67 |
| mixed PPC tail p | 0.91 | 0.92 | 0.90 | 0.58 |
| impute kept-pair (ref wb) | .748 (.718) pass | .668 (.632) pass | .597 (.535) pass | .448 (.345) FAIL |
| beta_force (Rubin) | - | 0.925 sd 0.43 | - | 0.975 sd 0.62 |

- Reconcile errors: base none (warning calibration set 10 < 20, as before);
  0.15 fails reports_honoured (1) and singleton honest-width (0.586); free
  fails wb 0.34 and prior predictive 0.053.
- **Compression, corpus2 before -> base after:** reconcile wb 0.60 -> 0.63;
  impute kept-pair 0.43 -> 0.67. Impute no longer compresses past reconcile.
- **Ranking is insensitive to the noise prior.** Top 20 at base and free hold
  the same names; at most 4 swap adjacent ranks (Belisarius/McClellan/Yi,
  Montgomery/Agrippa/Zhukov/Rommel/Lannes). Napoleon 9.27, Caesar 5.87,
  Hannibal 5.18, Khalid 3.75, Frederick 3.74, Lee 3.55, Alexander 2.07.
- **Khalid #4 and Pyrrhus #14, unchanged** from corpus2 and across variants
  (Khalid 3.75 [1.92, 5.52]; Pyrrhus 0.75 [-0.45, 1.88]). P1 does not touch
  their cause, the inflated opposing figures in §29 (Islamic tradition at
  Yamama/Chains/Ullais; inverted Beneventum).
- Model gate failures (cross-component 38/50, PPC category) and impute
  distributional_check predate P1. Free also fails circularity (0.285 > 0.217).

## Reviews

- **bayesian-model-reviewer: approve with changes.** BLOCKER (fixed in
  07f529c): manual_entry bias unpinned. MAJOR, to fix before the corpus2
  refit:
  1. Divergences need target_accept 0.99: try a centred `e` (ZeroSumNormal
     sigma=tau_e) and a centred battle effect, keep zd non-centred.
  2. Regime offsets with few rows learn from the other side of a battle
     through the shared effect (Gaugamela fixture: published below neutral).
     Measure g_regime P1 vs independent sides per regime. g_regime posteriors
     are not stored today, only contraction, so storing them comes first. The
     rewritten Gaugamela test (`abs < 0.25`) tests nothing; replace it with a
     direct leak test.
  3. tests/model/conftest.py draws `is_ancient` per side, so battles mix eras.
     Draw it per battle.
  MINOR: era column not recentred after the anchor mask; singleton gate 0.85
  has no margin (tie to reliability); add e/s_b recovery and mu_side coverage
  tests; README must say force ratios are shrunk by reliability.
- **python-reviewer: no CRITICAL/HIGH.** MEDIUM:
  1. The span-guard exemption keys on source type, not the `override:` url.
  2. `student_t_nu_floor` is not casualty-prefixed (casualty floor moved
     2 -> 3).
  3. Override values below min_plausible_troops are silently dropped.
  4. The 1800 cutoff is hard-coded 3x.
  5. Skipped gates and a missing span key pass silently.
- Classify oddity: one missingness request listed troop_total as both missing
  and known; 14 `classify_llm_answer_rejected` (several rank-0 commanders not
  marked co-field) per DB, same as each other.

## User decisions (2026-10-25, all as recommended)

1. P1 approved at noise 0.22/0.40; sensitivity table disclosed with the ranking.
2. Model-review MAJORs (centred e/battle effect; g_regime stored and
   cross-side learning measured, cut if > 0.5 prior sd; conftest `is_ancient`
   per battle) fixed before the corpus2 refit.
3. Python-review MEDIUMs fixed in the same pass; the casualty nu floor goes
   back to 2.
4. Corpus3 (seed writer, crawl) after the corpus2 refit.
