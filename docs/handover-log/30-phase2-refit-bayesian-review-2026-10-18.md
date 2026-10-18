# Handover-log §30: Bayesian review of the Phase 2 refit, 2026-10-18

Part of the handover log; the index is `docs/handover-log/README.md`.

Reviewer: the `bayesian-model-reviewer` instructions (`.claude/agents/bayesian-model-reviewer.md`),
run through an Opus general-purpose agent because the project agent was not
registered in that session. Read-only, on reconcile 84, impute 86, model rounds
87-106 pooled as run 108, evaluate on 108. Artefacts it used:
`data/processed/model_diagnostics/run108/{gate_report.json,rubin_per_general.csv,rounds_summary.json}`,
`data/imputed/neutral/round_*/sides.csv`, `data/processed/evaluation_report.json`.

**Verdict: no BLOCKER.** The sampler, pooling, ordinal link and WAR arithmetic
are correct. Readable with caveats, and only at the top: ranks 1-15 rest on
their own data (skill contraction >= 0.25, every row overlaps the next,
P(above next) 0.45-0.88); cross-era placement is assumed.

Checked and passing: anchoring by the shared prior with no era term
(`pipeline/modelers/model.py:239-241`); non-centred skill; ordered cutpoints
(`model.py:296-299`); weights fixed data, every side sums to 1; Rubin's rules;
replacement level per draw (`war.py:85-99`); WAR decomposition reconciles.

**MAJOR**
1. The 89 headline / 643 under-determined split is not honest. WAR shares a
   side's xWAR out by weight (`war.py:182-184`), so subordinates inherit the
   leader's credit. Ney (rank 16): skill -0.80 [-4.92, 2.96] (the prior,
   sigma_skill ~1.95) but WAR 0.61 [0.28, 0.95]; same for Lannes, Massena,
   Marmont, Davout, Murat, Soult. Of 87 generals with >= 3 credited battles,
   57 have skill contraction 1-(sd/sigma)^2 < 0.2 (36 in Napoleon's
   component); 26 have WAR intervals above zero with unidentified skill;
   median effective battles 0.74. Rank 16 is the first prior-dominated one.
   Fix options: gate the headline on `effective_battles` (19 generals >= 3,
   27 >= 2) or skill contraction >= 0.25; a per-commander counterfactual
   (replace only w_i*s_i with w_i*s_rep); at minimum disclose that WAR is side
   performance shared by weight.
2. Eastern Front force ratios are dominated by impute. Seven of 10 Soviet
   sides have reconcile sd (0.72-1.6) at or above reconcile's prior s0 (0.69),
   so the weak-side rule treats them as missing and impute fills them with the
   Soviet Union polity offset (+1.06, learned from 3 strong sides, all Soviet
   victories) plus the outcome effect. Stalingrad reconcile x_diff -0.36 to
   imputed +1.29; Barbarossa -0.81 to +1.09; Moscow +0.04 to +1.43; Kursk
   +0.55 to +1.46. Corpus-wide the outcome shift is small (winner minus loser
   +0.05 to +0.11 over 149 battles). 75 of 341 estimated sides are discarded
   this way. Check whether weakness is judged on a covariance inflated by the
   shared regime error (`pipeline/imputers/likelihood.py:261-316`); consider a
   1914-1945 era bin so polity stops carrying a total-war scale effect.
   Zhukov is the only FMI warning (0.311).

**Disclose, not defects**
3. PPC category check: victory 0.718 vs upper 0.710, defeat 0.210 vs lower
   0.226; computed on one round (`len(rounds)//2`, `model.py:179`) whose
   beta_force is 0.38 against pooled 0.83, with an orientation-dependent
   statistic. Fix: pool over rounds, use an orientation-free statistic.
4. 41 of the top 50 outside the largest component (Napoleonic, 237 generals):
   cross-era placement rests on the shared N(0, sigma) prior. Unfixable
   without bridging battles.
5. FMI: m=40 will not lower it; relative efficiency at m=20 is 98.5%.
   beta_force FMI 0.69 (not published); skill FMI median 0.003, max 0.10.
6. Circularity: gap 0.224 is about 0.3 of the combined sd; direction matches
   outcome leakage; Spearman 0.989. Compare the gap to the combined sd.
7. Reconcile prior_predictive 0.062: the prior is slightly too wide, which is
   conservative; it does not drive the ranking. Empty calibration set means
   regime biases come from the 45-case inflation prior: disclose.
8. No decisive_* outcome exists, so the link is effectively three-level; c2
   rests on the prior (harmless: expected score uses c1 only).
9. The `agrippa_above_augustus` named check is no longer independent evidence:
   the Actium weights are now hand-set.

**Impute draws change (run 85 to 86):** sound. ESS per draw unchanged
(3.8% vs 4.3%); four chains agree. The ridge is a parameterisation issue
(non-centred tau*z with one data-dominated level; centring the polity block
or a Student-t effect would fix mixing) with the modelling smell in item 2.
Run 86 stores only the worst rhat/ESS: record the parameter and tail ESS.

**Overrides:** pyrrhic_victory maps to 3 like victory (`design.py:101-110`);
indecisive pairs map to 2; manual weights are fixed data; Zumail's empty side
enters as a mu_unknown latent (all 4 unknown sides are defeats, so mu_unknown
leans on its N(0,1) prior; disclose).
