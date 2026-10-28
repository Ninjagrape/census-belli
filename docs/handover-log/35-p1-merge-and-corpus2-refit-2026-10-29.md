# 35. P1 review fixes, merge and the corpus2 refit (2026-10-28/29)

## Fixes on `wip/troop-p1`, merged to `main` 490a5d1 (not pushed)

06cf2d0 (review fixes), 8e75e0b (heavy-tail fixture at target_accept 0.99),
6e9ca4d (review follow-ups: an `override:` row the loader would drop raises;
span guard fails with no completed run; nu floor consistent for casualty
construction; leak test reworked).

- **Centring tried, reverted.** Centred `b`: 2081 divergences on the main
  recovery corpus. Centred `e` (ZeroSumNormal sigma=tau_e): convergence
  failures on tau_e/e in test_reconcile_recovery (1 failed + 11 errors) and
  contamination at 0.97-0.99. Non-centred stays at 0.99. The corpus copy at
  0.9 had 0 divergences; 0.99 kept because fixtures diverge at 0.97.
- **Cross-side learning, corpus copy, P1 vs `side_prior_independent`:**
  unlabelled 0.00, staff_return 0.40, ancient_claim 0.09 prior sd. No cut
  (threshold 0.5). Leak fixture with an unpaired control: P1 cross-side
  pass-through 0.44 of log(5), prior-implied s_b^2/(s_b^2+s_d^2) 0.73
  (measured xfail, gap 0.29 > 0.25); independent cross-side -0.11.
- Reviews: python (no CRITICAL/HIGH, five MEDIUMs fixed), model (no
  BLOCKER/MAJOR; optional TODO: scale-mixture battle effect to lower
  target_accept).
- Verified on merged main: ruff, mypy (159) clean, one head d0e1f2a3b4c5,
  non-model suite 2732 passed 0 failed 0 skipped. Model files: p1_recovery 13,
  recovery 19 + 1 xfail, contamination 3, neutral 5 + 1 xfail, impute 5.

## Corpus2 refit

Backup `data/backups/general_war_corpus2_20261029_pre_p1_refit.dump`. Logs
`data/logs/refit20261029_*`. All stages `--continue-past-deferred`.

| stage | result |
|---|---|
| extract | 36/36 overrides, 237 battles, 422 sides; 14,676 passages deferred (accepted) |
| resolve | 4 saved + 2 new Haiku answers (Stalin at Berlin Q855, Wöhler Q64794); 50/50 links; warning duplicate generals 6 |
| reconcile 109 | 0 divergences, rhat 1.004, wb 0.630, reliability 0.675, gap 0.045 |
| classify | 25 saved + 2 new Sonnet answers (Gazala, Walaja); all gates pass; 113 needs_review |
| impute 110 | rhat 1.006, ESS 1081; kept-pair 0.643 (ref 0.630), kept-side 0.943; warning distributional early_modern |
| model 132 | rhat 1.003, ESS 2418, 0 div; warnings PPC category, circularity 0.194 > 0.185 (108: 0.224 > 0.183), 38/50 cross-component |
| evaluate | 5/5; held-out 0.780 (0.751), Brier 0.164 (0.176) |
| report | 93 headline, 632 under-determined |

**Compression:** reconcile wb 0.60 -> 0.63; impute kept-pair 0.43 -> 0.64.
beta_force (Rubin) 0.834 -> 1.027, FMI 0.69 -> 0.41. g_regime on corpus2
matches the copy to three decimals.

## Khalid and Pyrrhus

- Khalid #4, 3.71 [1.86, 5.47] (run 108: 4.20 [2.08, 6.19]). Hazir, Chains,
  Ullais now near modern estimates; Yamama (Banu Hanifa 51k on one
  "40,000-100,000" report) still inflated. His rank now rests on 11 wins in
  12 against opponents who appear once (prior-identified; component 2 is
  disconnected from Napoleon's). Small-sample placement.
- Pyrrhus #14, 0.70 [-0.51, 1.84]. Troop inversion at Beneventum fixed, but
  Beneventum is coded **indecisive** (scored 0.5 vs baseline 0.34, +0.27).
  As a Roman victory he drops to about 0.09, out of the top 15. Outcome
  override needed.

## Open, from both reviewers (none blocks an internal reading; not publishable)

1. Beneventum outcome -> Roman victory (outcome override).
2. Waterloo UK: Blücher `subordinate` rank 1; should be co-field (weight kept).
3. Yamama (176) and Kalka (187) opposing figures probably inflated.
4. Torch Vichy side 376 (127,728 from all of French North Africa), LOW-MEDIUM.
5. §30 MAJOR still open: subordinates inherit side xWAR (Ney #15, skill sd
   1.95 = prior); ACW volume effect (Meade #10, McClellan #12).
6. Stalin at Berlin 0.10, Hitler at Berlin 0.15: defensible, LOW.
7. One casualty reports_honoured violation, side 375 (minor).
