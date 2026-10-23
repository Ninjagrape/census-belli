# Handover-log §32: troop-number redesign review, 2026-10-23

Part of the handover log; the index is `docs/handover-log/README.md`.

Why this exists: the §31 fix (informative noise prior, impute uses an exported
likelihood) was built on `wip/troop-recon` and failed every recovery gate on
the corpus. An experiment on the corrected extract (worktree `w-exp`, branch
`wip/noise-exp`, DB `general_war_wt_exp_corpus`, reconcile runs 109/111/114/116)
compared four noise treatments; none met the targets:

| | V1 log-sd 0.3, t | V2 0.3, contaminated | V3 0.1, t | V4 0.1, contaminated |
|---|---|---|---|---|
| sigma modern | 0.642 | 0.443 | 0.264 | 0.247 |
| singleton slope | 0.536 | 0.388 | 0.828 | 0.439 |
| within-battle slope (wb) | 0.573 | 0.433 | 0.761 | 0.472 |
| post-1800 > 50k ratio | 0.47x | 0.46x | 0.61x | 0.48x |
| impute kept-pair slope | 0.463 | 0.675 | 0.452 | 0.682 |

The user chose a reviewer redesign before any merge. Reviewer:
bayesian-model-reviewer instructions via an Opus agent, read-only, in-memory
refits with a pluggable side prior (`scratchpad\reviewer2\harness.py`,
`run_cf.py`). Its V3 refit reproduced run 114.

## Findings (computed unless marked)

Three defects, none of them the noise median:
1. **Noise leaks around the prior.** V3 posterior sigma_type_modern 0.314
   (3.5 prior sds above 0.22 even at log-sd 0.1); sigma_round escapes
   HalfNormal(0.1) to 0.22 (0.67 under a Normal likelihood); median per-row sd
   0.37, times the t(3.4) variance factor 2.4. (Run 114's diagnostics say 0.264
   for sigma_type: probably a different summary, unchecked.)
2. **About 9 scope-conflated sides drive the leak**: Yom Kippur Egypt (707 to
   988k), Six-Day UAR, Lebanon 1982 Israel, Toulon royalists, Stalingrad x2,
   Smolensk 1812 x2, Jena-Auerstedt France (being split). With them in, a
   Normal fit puts sigma at 0.60. Torch US and Hohenfriedberg are borderline.
3. **The side prior is wrong twice**: one corpus mean (the 20th-century mean is
   12.2 against m0 10.4), and a battle's two sides drawn independently while
   reported log sizes correlate at 0.80 (ICC; 0.71 after era means; ancient
   0.44, 19th c. 0.87, 20th c. 0.83). An independent prior implies ratio
   variance ~2.9 against 0.598 observed: that is why the PPC fails everywhere.

Also: the shipped PPC (`recovery.between_battle_ppc`) replicates around the
posterior mu_side and cannot pass for any non-zero noise. Replace it with a
mixed predictive (new side deviations, kept offsets, new noise).

Impute kept-pair slope 0.45 is not a bug: lik_mean is right (report - 0.154,
slope 0.995) but lik_sd has median 0.449 (fitted scale 0.31, sigma_round 0.22
plus range variance, t Fisher factor 1.21), so a reported ratio's reliability
is ~0.28-0.45. Shared-error loadings cancel within a battle.

**A wb >= 0.8 gate cannot be met by a calibrated model unless noise <= ~0.2.**
wb measures the reliability of a reported ratio, (V - 2 sigma_eff^2)/V, about
0.65-0.73 at sigma 0.28. Models that pass it do so by believing true ratios
vary 4x more than they do. For the force-ratio regression, the calibrated
posterior is the correct regression-calibration input; near-raw ratios
attenuate beta_force.

## Counterfactuals (V3 noise prior unless stated; "pinned" = sigma_round and
sigma_est HalfNormal(0.02); "excl" = the 9 sides dropped)

| Variant | sing / spread / wb | post1800 / anc / <10k / all | mixed PPC (obs ~0.58) | rhat / ESS |
|---|---|---|---|---|
| V3 baseline | .83 / .84 / .77 | .67 / .72 / 1.30 / .85 | 3.04, p 1 | 1.01 / 442 |
| era means, t | .80 / .83 / .68 | .97 / .96 / 2.09 / 1.19 | 2.31, p 1 | 1.04 / 123 |
| era + battle, t (all data) | .80 / .85 / .07 | 1.17 / 1.09 / 2.57 / 1.40 | .69, p .71 | 1.02 / 254 |
| era + battle, contaminated, pinned, excl | .85 / .88 / .30 | | .38, p .01 | 1.07 / 73 |
| P2: era, independent, Normal, pinned, excl | .94 / .94 / .88 | 1.04 / .96 / 1.30 / 1.12 | 2.25, p 1 | 1.04 / 166 |
| **P1: era + battle, Normal, pinned, excl** | **.93 / .94 / .59** | 1.08 / .88 / 1.38 / 1.12 | **.55, p .45** | 1.01 / 510 |
| P1, t(4) side deviations | .94 / .95 / .66 | 1.27 / 1.13 / 1.56 / 1.34 | .39, p .015 | 1.01 / 488 |
| P1 at noise median .15 / .30 / free | wb .72 / .54 / .43 | | p .44 / .52 / .53 | |

P1 named sides (neutral): Leipzig 348k / 229k, Kursk 1.81M / 948k, Uhud
1.1k / 2.9k. Earlier "battle effect collapses" was the leak plus the
conflated sides, not the parameterisation (s_d 0.42 once both are fixed).

## Proposed design P1 (awaiting the user's approval)

```
mu_s   = m0 + e[era(s)] + s_b*zb[battle(s)] + s_d*zd[s]
e      = tau_e * ZeroSumNormal(5); tau_e ~ HalfNormal(1)   # <500, 500-1500, 1500-1800, 1800-1900, >=1900
zb ~ StudentT(4,0,1); zd ~ Normal(0,1); s_b ~ HalfNormal(1); s_d ~ HalfNormal(0.5)   # non-centred; s0 removed
log y_r ~ Normal(mu_s + offsets, sigma_r)
sigma_modern ~ LogNormal(log .22, .2); sigma_ancient ~ LogNormal(log .40, .2); no sigma_round/sigma_est for troops
```

Files: `reconcilers/model.py` side-prior block (~608-614); `design.py` (battle
index, era bin per side); `ModelPriors` + `agents/reconcile.yaml`
(`likelihood_family: normal`, drop troop dispersions); `recovery.py` (mixed
PPC); `config/overrides.yaml` (the conflated sides, drafted by
historiography-reviewer). `likelihood_export` and impute code unchanged.

Gates: singleton slope and spread >= 0.85 (error); s_d >= 0.25 (error,
collapse guard); posterior noise <= 1.5x prior median (error, leak guard);
mixed-PPC tail p >= 0.05 (error); wb >= 0.5 (error) and |wb - model-implied
reliability| <= 0.1 (warning), replacing wb >= 0.8; neutral post-1800 ratio /
all-sides ratio in [0.85, 1.15] (warning); neutral all-sides ratio in
[0.9, 1.1] (warning; P1 1.12 fails, see level); impute kept-pair slope within
+/- 0.1 of reconcile wb (replacing >= 0.8); kept-side slope >= 0.9. Extract or
reconcile guard: kept point rows on one side spanning > log 1.0 without an
override warn, > 2.0 error.

Sensitivity required: noise median 0.15 / 0.22 / 0.30 / free, reporting wb,
impute slope, beta_force and the top 20. Human spot-check of 10 gold cases
before publication.

**Level issue (HIGH, head counts not ratios):** g_unlabelled is +0.18 in V3
and -0.13 (neutral) / -0.34 (published) with era means; 96% of rows are
unlabelled and only 10 modern_scholarly rows anchor the level, so published
counts shift 1.1-1.4x. Needs a tighter g_unlabelled prior or an explicit level
anchor: a design decision.

What P1's numbers mean: head counts near reports (singleton slope 0.93, large
modern ~0.97x relative); sides under 10k pulled up ~1.23x; large ancient
~0.79x (shrinkage to battle and era, not inflation correction); force ratios
shrunk toward parity by their reliability ~0.6 (Kursk 2.45:1 to 1.91:1,
Leipzig 1.84 to 1.52, Uhud 1:4 to 1:2.6); beta_force is per unit of true log
ratio. Impute still conditions on outcome (circular): keep evaluate's
`observed_only` variant.

Ranked: P1; P1 with t(4) deviations; P2 (passes old gates, miscalibrated,
attenuates beta_force); status quo rejected.

Inferred, not computed: posterior noise under the 0.2 log-sd prior (~0.28-0.30
expected); effect on beta_force and the ranking; published-scale P1 after a
level fix.
