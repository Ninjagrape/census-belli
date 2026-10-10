## Stage 4: Reconcile (pipeline/stages/reconcile.py)
Built and writing estimates against live Postgres; see handover.md §19-§20.
- [x] Source-bias model specification in PyMC (claim regime, source type + key, lineage, scope, era)
- [x] Ancient/claim-regime inflation prior, fitted to a 45-case gold set
- [x] Likelihood with censored bounds and StudentT noise
- [x] Compute best estimates + 95% CI per side (exp of posterior median)
- [x] Write estimates to battle_sides, biases to sources, one model_runs row per run
- [x] Convergence diagnostics + the diagnostics_json gate
- [x] Parameter-recovery suite (tests/model, `-m model`): 13 pass, 1 xfail (censoring)
- [x] bayesian-model-reviewer pass (handover §20.10)
- [x] Regime-neutral troops fit, cross-regime duplicate merge, neutral columns + npz (handover §21.1)
- [ ] est_troops_method counts lineages the cross-regime merge dropped (§21.7)
- [ ] Parameterise gate thresholds from the spec so integration can pass neutral_fit_converged (§21.5)
- [ ] Width gates on the identified scale; n_identifying_sides (§20.10)
- [x] bound_only label; [ ] ExGaussian bound likelihood later (§20.10)
- [x] Write estimates only after convergence (§20.11)
- [ ] Prior predictive check; per-side casualty ratio; InverseGamma(2,1) on variance
- [ ] Run on a real corpus (needs crawl/extract/resolve on config/sources_seed_arsht.yaml first)
- [ ] (gap review A11) data_era is centred over all dated rows but applied only to
      unlabelled rows (design.py / model.py); centre within the unlabelled rows
- [ ] (gap review A12) a fit failing write thresholds leaves sides NULL with no
      missing_data_log row
- [ ] (gap review) dead spec params: bias_prior_mu, regime_deviation_prior_sd
