# Handover-log §31: troop numbers, reviewer diagnosis, 2026-10-22

Part of the handover log; the index is `docs/handover-log/README.md`.

Reviewer: `bayesian-model-reviewer` instructions run through an Opus
general-purpose agent (the type is still not registered). Read-only on
`general_war_corpus2` (reconcile 84, impute 86, model 108). It rebuilt the
troops design with the pipeline's `load_reports`/`build_design` (389 rows,
341 sides, 378 unlabelled, one source type) and refit it in memory with
nutpie; the baseline refit reproduced run 84 (s0 0.692, m0 10.658, singleton
slope 0.61; run 84's own diagnostics: singleton_slope 0.605, spread_ratio 0.65).

## Q1: reconcile shrinkage (computed)

| Group | n | Published | Neutral |
|---|---|---|---|
| Post-1800, report > 50k | 98 | 0.71x | 0.62x |
| Ancient, report > 50k | 21 | 0.83x | 0.73x |
| Report < 10k | 30 | 2.68x | 2.35x |
| Singleton slope | 291 | 0.605 | 0.573 |
| Within-battle log-ratio slope | 150 | 0.60 | 0.61 |

The model stage reads impute's rounds (`config/model_default.yaml:119`
`scale: neutral`), which divide the side prior back out: for the 266 kept
sides imputed-vs-report slope is 0.93. The shrinkage reaches the model through
the weak-side rule and through impute's kept-pair ratio slope of 0.43.

Mechanism: the side prior `mu_side = m0 + s0 * z`, `z ~ StudentT(4)`
(`pipeline/reconcilers/model.py:538-544`) against per-row noise ~0.51 (t,
nu 3.4): shrinkage factor 0.58-0.63, observed slope 0.61. Regime and era
offsets are uniform shifts (g_unlabelled -0.08 +/- 0.37 from a prior mean of
0.92; a_era -0.11 +/- 0.14). sigma is identified only by 31 multi-report sides
(pooled within-side sd 1.16, median 0.39), prior IG(3, 4/3) with mean 0.67;
the gold set's faithful_sd is 0.22. A Normal likelihood (slope 0.56) or a
Normal side prior (0.47) is worse.

Counterfactuals (fixed sigma constants as stand-ins for an informative prior):

| Variant | Singleton slope | Ratio slope | Post-1800 > 50k | < 10k | Weak (neutral) |
|---|---|---|---|---|---|
| Baseline | 0.61 | 0.60 | 0.71x | 2.68x | 76 |
| sigma 0.25 | 0.76 | 0.71 | 0.83x | 2.04x | |
| sigma 0.2, sigma_round = sigma_est = 0.1 | 0.905 | 0.845 | 0.98x | 1.42x | |
| same, neutral fit | 0.915 | 0.856 | 0.88x | 1.24x | 26 |
| period means only | 0.55 | 0.41 | 1.14x | 6.0x | |
| battle random effect | 0.76 | 0.03 | | | |

Period means alone are harmful; a battle random effect with independent
errors reads every within-battle difference as noise. Both rejected.

## Q2: impute weak-side rule (computed)

`pipeline/imputers/likelihood.py:261-279, 314-316`: P_lik = S_post^-1 - I/s0^2
per battle, weak when diag <= 0.5 (`agents/impute.yaml` `min_lik_precision`),
on the raw covariance (`:619-622`). For a singleton this is posterior sd >=
0.619. Shared parameters explain a median 13% of side variance; a residual
rule leaves 63 weak (rescues 12, no Soviet side). corr(|report - m0|,
posterior sd) = 0.94: weak sides sit a median 2.0 log units from m0 (kept
0.5); 30 weak sides report < 10k, 17 > 200k, 13 are bound-only. A t prior
against a t likelihood gives a wide, near-bimodal posterior in the tails;
dividing it out as N(m0, s0^2) mislabels it, though the t likelihood's Fisher
sd is ~0.63 wherever the report sits.

Not verified by the reviewer: impute's ratio compression mechanism (inferred
from sigma_lik), the bound likelihood's share, the real imputation model.

Decisions taken on this: `docs/handover/07-next-sessions-plan.md`, "A:
approved design (2026-10-22)".
